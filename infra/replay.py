"""Drive the pipeline from a logged run instead of from a camera.

Two modes:

* ``replay_run`` -- re-run a recorded flight through the same pipeline code,
  either with the config the run was flown with or with a substitute.
* ``ab_compare`` -- run the same log through two configs and diff the resulting
  PlannerCommand streams, so a tuning change can be judged by what it would
  have *done* rather than by staring at the numbers.

The pipeline is injected as a factory rather than imported, for two reasons.
The perception and planning packages do not exist yet when this module is
written, and more importantly replay must exercise the same object the live
stack builds -- if replay constructed its own, it would be testing a
lookalike. ``ReplayPipeline`` is the whole contract.

Command streams are compared as canonical bytes. Two runs of the same config
over the same log must produce identical bytes; anything else means state
leaked across the run boundary or something non-deterministic crept into the
decision path, which is exactly the bug this is here to catch.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from infra.flightlog import (
    FlightLogReader,
    FrameRecord,
    canonical_json,
    encode_planner_command,
)
from infra.metrics import Metrics
from sources.types import FcState, FrameBundle, PlannerCommand

if TYPE_CHECKING:
    from config import Config


class ReplayError(RuntimeError):
    """Raised when a run cannot be replayed."""


class ReplayPipeline(Protocol):
    """What replay needs from a pipeline. Implemented by the real stack."""

    def reset(self) -> None:
        """Drop all accumulated state. Called once before the first frame."""
        ...

    def tick(self, bundle: FrameBundle, state: FcState | None) -> PlannerCommand | None:
        """Process one frame. Returning None means "no command this tick"."""
        ...


PipelineFactory = Callable[["Config"], ReplayPipeline]


def split_side_by_side(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper around the live source's split rule."""
    from sources.stereo_uvc import CaptureError
    from sources.stereo_uvc import split_side_by_side as split

    try:
        return split(frame)
    except CaptureError as exc:
        raise ReplayError(str(exc)) from exc


def decode_frame(record: FrameRecord) -> FrameBundle:
    from sources.stereo_uvc import CaptureError, decode_mjpg

    try:
        return decode_mjpg(record.mjpg, record.seq, record.t_ns)
    except CaptureError as exc:
        raise ReplayError(str(exc)) from exc


class TelemetryTrack:
    """Telemetry indexed for "what did the FC report at or before this frame".

    Zero-order hold, not interpolation: FcState carries discrete fields (mode,
    armed, rc) that cannot be averaged, so the honest answer for a frame is the
    last state actually received before it.
    """

    def __init__(self, states: Iterable[FcState]) -> None:
        self._states = sorted(states, key=lambda s: s.t_ns)
        self._times = [s.t_ns for s in self._states]

    def __len__(self) -> int:
        return len(self._states)

    def at(self, t_ns: int) -> FcState | None:
        """Latest state at or before ``t_ns``; None if the log starts later."""
        if not self._states:
            return None
        index = bisect.bisect_right(self._times, t_ns) - 1
        if index < 0:
            return None
        return self._states[index]


def command_stream_bytes(commands: Iterable[PlannerCommand]) -> bytes:
    """Canonical byte serialisation of a command stream, for exact comparison."""
    lines = [canonical_json(encode_planner_command(cmd)) for cmd in commands]
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


@dataclass(frozen=True)
class ReplayResult:
    run_dir: Path
    config_label: str
    commands: tuple[PlannerCommand, ...]
    frames_seen: int
    telemetry_seen: int
    metrics: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def stream(self) -> bytes:
        return command_stream_bytes(self.commands)

    def __len__(self) -> int:
        return len(self.commands)


@dataclass(frozen=True)
class CommandDiff:
    """One differing field at one index of two command streams."""

    index: int
    field_name: str
    a: Any
    b: Any

    def __str__(self) -> str:
        return f"[{self.index}] {self.field_name}: {self.a!r} != {self.b!r}"


@dataclass(frozen=True)
class ABResult:
    a: ReplayResult
    b: ReplayResult
    diffs: tuple[CommandDiff, ...]

    @property
    def identical(self) -> bool:
        return self.a.stream == self.b.stream

    @property
    def first_divergence(self) -> int | None:
        """Index of the first differing command, or None if the streams match."""
        return self.diffs[0].index if self.diffs else None

    def summary(self) -> str:
        if self.identical:
            return (
                f"identical: {len(self.a)} commands, "
                f"{self.a.config_label} vs {self.b.config_label}"
            )
        head = "\n".join(f"  {d}" for d in self.diffs[:10])
        more = f"\n  ... {len(self.diffs) - 10} more" if len(self.diffs) > 10 else ""
        return (
            f"{len(self.diffs)} differences between {self.a.config_label} and "
            f"{self.b.config_label} (first at command {self.first_divergence})\n{head}{more}"
        )


def diff_command_streams(
    a: Iterable[PlannerCommand], b: Iterable[PlannerCommand]
) -> tuple[CommandDiff, ...]:
    """Field-level diff of two command streams, including a length mismatch."""
    list_a, list_b = list(a), list(b)
    diffs: list[CommandDiff] = []
    for index in range(min(len(list_a), len(list_b))):
        enc_a = encode_planner_command(list_a[index])
        enc_b = encode_planner_command(list_b[index])
        for key in sorted(enc_a):
            if enc_a[key] != enc_b[key]:
                diffs.append(CommandDiff(index=index, field_name=key, a=enc_a[key], b=enc_b[key]))
    if len(list_a) != len(list_b):
        diffs.append(
            CommandDiff(
                index=min(len(list_a), len(list_b)),
                field_name="<stream length>",
                a=len(list_a),
                b=len(list_b),
            )
        )
    return tuple(diffs)


def iter_replay_inputs(
    reader: FlightLogReader, limit: int | None = None
) -> Iterator[tuple[FrameBundle, FcState | None]]:
    """Frame/telemetry pairs in logged order, as the live pipeline would see them."""
    track = TelemetryTrack(reader.telemetry())
    for count, record in enumerate(reader.frames()):
        if limit is not None and count >= limit:
            return
        bundle = decode_frame(record)
        yield bundle, track.at(bundle.t_ns)


def replay_run(
    run_dir: str | Path,
    factory: PipelineFactory,
    cfg: Config | None = None,
    *,
    config_label: str | None = None,
    limit: int | None = None,
    metrics: Metrics | None = None,
) -> ReplayResult:
    """Re-run a logged flight through the pipeline.

    ``cfg`` defaults to the config the run was flown with, read back out of the
    run directory. Pass one to replay the same log under different tuning.
    """
    reader = FlightLogReader(run_dir)
    config = cfg if cfg is not None else reader.config()
    label = config_label or ("logged" if cfg is None else "override")

    pipeline = factory(config)
    pipeline.reset()

    tracker = metrics if metrics is not None else Metrics.from_config(config)
    commands: list[PlannerCommand] = []
    frames_seen = 0
    telemetry_seen = len(TelemetryTrack(reader.telemetry()))

    for bundle, state in iter_replay_inputs(reader, limit=limit):
        frames_seen += 1
        with tracker.stage("replay.tick"):
            command = pipeline.tick(bundle, state)
        if command is not None:
            commands.append(command)

    return ReplayResult(
        run_dir=Path(run_dir),
        config_label=label,
        commands=tuple(commands),
        frames_seen=frames_seen,
        telemetry_seen=telemetry_seen,
        metrics=tracker.to_dict(),
    )


def ab_compare(
    run_dir: str | Path,
    factory: PipelineFactory,
    cfg_a: Config,
    cfg_b: Config,
    *,
    label_a: str = "A",
    label_b: str = "B",
    limit: int | None = None,
) -> ABResult:
    """Replay one log under two configs and diff the command streams.

    Each side gets a freshly built pipeline from the factory, so state from the
    A run cannot colour the B run.
    """
    result_a = replay_run(run_dir, factory, cfg_a, config_label=label_a, limit=limit)
    result_b = replay_run(run_dir, factory, cfg_b, config_label=label_b, limit=limit)
    return ABResult(
        a=result_a,
        b=result_b,
        diffs=diff_command_streams(result_a.commands, result_b.commands),
    )
