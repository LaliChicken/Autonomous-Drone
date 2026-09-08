"""One directory per run: frames, telemetry, planner decisions, config, git hash.

Layout of a run directory::

    <root>/<run_id>/
        meta.json        run id, git revision, start/end markers, record counts
        config.yaml      verbatim YAML the run used -- loads straight back
        config.json      resolved view (radians, derived) for eyeballing
        frames.mjpg      concatenated MJPG bitstreams, exactly as received
        frames.jsonl     index: seq, t_ns, byte offset, byte length
        telemetry.jsonl  one FcState per record
        planner.jsonl    one planner tick per record: what it saw, what it chose
        metrics.json     stage rollup, written at close

Frames are stored as the bitstream the camera handed us. No decode, no
re-encode: a re-encode would silently change the pixels that every downstream
result was computed from, which makes a "replay" of the run a different run.

Wall-clock time appears exactly twice -- the run directory name and the
human-readable ``created_utc`` in meta.json -- and never in a record the
pipeline consumes. CLOCK_MONOTONIC is meaningless across boots, so it cannot
name a directory; everything the pipeline reads back is monotonic ns.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import numpy as np

from sources.types import FcState, OccupancySnapshot, PlannerCommand

if TYPE_CHECKING:
    from config import Config
    from infra.metrics import Metrics

META_NAME = "meta.json"
CONFIG_YAML_NAME = "config.yaml"
CONFIG_JSON_NAME = "config.json"
FRAMES_BLOB_NAME = "frames.mjpg"
FRAMES_INDEX_NAME = "frames.jsonl"
TELEMETRY_NAME = "telemetry.jsonl"
PLANNER_NAME = "planner.jsonl"
METRICS_NAME = "metrics.json"

REPO_ROOT = Path(__file__).resolve().parent.parent


class FlightLogError(RuntimeError):
    """Raised when a run directory is unusable or malformed."""


def canonical_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, no incidental whitespace.

    Determinism matters here -- replay compares command streams byte for byte,
    so two logically equal records must serialise identically.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def git_revision(repo_root: Path | None = None) -> dict[str, Any]:
    """Current commit, branch, and dirty flag.

    Never raises: a run must still be loggable outside a git checkout, or with
    git missing entirely. The failure is recorded in the returned mapping so
    the log says *why* it has no revision rather than silently claiming none.
    """
    root = repo_root or REPO_ROOT

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise FlightLogError(str(exc)) from exc
        if out.returncode != 0:
            return None
        return out.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        if commit is None:
            return {"commit": None, "branch": None, "dirty": None, "error": "not a git repository"}
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        status = run("status", "--porcelain")
        return {
            "commit": commit,
            "branch": branch,
            "dirty": bool(status),
            "error": None,
        }
    except FlightLogError as exc:
        return {"commit": None, "branch": None, "dirty": None, "error": str(exc)}


# --------------------------------------------------------------------------
# Record encoding. Kept as free functions so replay can decode a run written
# by a different process without constructing a writer.
# --------------------------------------------------------------------------


def encode_fc_state(state: FcState) -> dict[str, Any]:
    return {
        "t_ns": int(state.t_ns),
        "roll": float(state.roll),
        "pitch": float(state.pitch),
        "yaw": float(state.yaw),
        "vel_ned": [float(v) for v in state.vel_ned],
        "agl_m": None if state.agl_m is None else float(state.agl_m),
        "mode": str(state.mode),
        "armed": bool(state.armed),
        "ekf_ok": bool(state.ekf_ok),
        # JSON object keys are strings; decode_fc_state puts them back to int.
        "rc": {str(k): int(v) for k, v in sorted(state.rc.items())},
    }


def decode_fc_state(record: dict[str, Any]) -> FcState:
    vel = record["vel_ned"]
    return FcState(
        t_ns=int(record["t_ns"]),
        roll=float(record["roll"]),
        pitch=float(record["pitch"]),
        yaw=float(record["yaw"]),
        vel_ned=(float(vel[0]), float(vel[1]), float(vel[2])),
        agl_m=None if record["agl_m"] is None else float(record["agl_m"]),
        mode=str(record["mode"]),
        armed=bool(record["armed"]),
        ekf_ok=bool(record["ekf_ok"]),
        rc={int(k): int(v) for k, v in record["rc"].items()},
    )


def encode_planner_command(cmd: PlannerCommand) -> dict[str, Any]:
    return {
        "t_ns": int(cmd.t_ns),
        "vx": float(cmd.vx),
        "vy": float(cmd.vy),
        "vz": float(cmd.vz),
        "yaw_rate": float(cmd.yaw_rate),
        "reason": str(cmd.reason),
    }


def decode_planner_command(record: dict[str, Any]) -> PlannerCommand:
    return PlannerCommand(
        vx=float(record["vx"]),
        vy=float(record["vy"]),
        vz=float(record["vz"]),
        yaw_rate=float(record["yaw_rate"]),
        reason=str(record["reason"]),
        t_ns=int(record["t_ns"]),
    )


def encode_occupancy(snapshot: OccupancySnapshot) -> dict[str, Any]:
    return {
        "t_ns": int(snapshot.t_ns),
        "bearings": [float(v) for v in snapshot.bearings],
        "distances": [float(v) for v in snapshot.distances],
        "confidence": [float(v) for v in snapshot.confidence],
        "unknown": [bool(v) for v in snapshot.unknown],
        "danger": [bool(v) for v in snapshot.danger],
    }


def decode_occupancy(record: dict[str, Any]) -> OccupancySnapshot:
    return OccupancySnapshot(
        t_ns=int(record["t_ns"]),
        bearings=np.asarray(record["bearings"], dtype=np.float64),
        distances=np.asarray(record["distances"], dtype=np.float32),
        confidence=np.asarray(record["confidence"], dtype=np.float32),
        unknown=np.asarray(record["unknown"], dtype=bool),
        danger=np.asarray(record["danger"], dtype=bool),
    )


@dataclass(frozen=True)
class FrameRecord:
    """One logged frame: index entry plus the MJPG bytes it points at."""

    seq: int
    t_ns: int
    offset: int
    length: int
    mjpg: bytes


class _JsonlWriter:
    """Append-only JSONL sink with optional periodic fsync."""

    def __init__(self, path: Path, fsync_every: int) -> None:
        self._path = path
        self._fsync_every = fsync_every
        self._handle: IO[str] = path.open("w", encoding="utf-8")
        self._since_sync = 0
        self.count = 0

    def write(self, record: dict[str, Any]) -> None:
        self._handle.write(canonical_json(record))
        self._handle.write("\n")
        self.count += 1
        if self._fsync_every:
            self._since_sync += 1
            if self._since_sync >= self._fsync_every:
                self.flush()
                self._since_sync = 0

    def flush(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()


def new_run_id(pid: int | None = None) -> str:
    """A sortable, collision-resistant run id.

    Wall clock is correct here: this names a directory a human will look for
    later, and a monotonic counter would restart at every boot.
    """
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{stamp}-{pid if pid is not None else os.getpid():d}"


class FlightLog:
    """Writer for one run directory. Use as a context manager."""

    def __init__(self, run_dir: Path, cfg: Config) -> None:
        self.run_dir = run_dir
        self.cfg = cfg
        self._closed = False
        self._frames_blob: IO[bytes] | None = None
        self._frames_offset = 0

        run_dir.mkdir(parents=True, exist_ok=False)
        fsync_every = cfg.flightlog.fsync_every

        # config.yaml is the re-loadable snapshot; config.json is the resolved
        # view. Both, because each answers a question the other cannot.
        calibration_hash = None
        raw = copy.deepcopy(cfg.raw)
        if cfg.camera.calibration_npz is not None:
            calibration_copy = run_dir / "calibration.npz"
            shutil.copyfile(cfg.camera.calibration_npz, calibration_copy)
            calibration_hash = hashlib.sha256(calibration_copy.read_bytes()).hexdigest()
            if raw:
                raw["camera"]["calibration_npz"] = "calibration.npz"
        if raw:
            import yaml

            (run_dir / CONFIG_YAML_NAME).write_text(
                yaml.safe_dump(raw, sort_keys=True, default_flow_style=False),
                encoding="utf-8",
            )
        (run_dir / CONFIG_JSON_NAME).write_text(
            json.dumps(cfg.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

        self._frames_index = _JsonlWriter(run_dir / FRAMES_INDEX_NAME, fsync_every)
        self._telemetry = _JsonlWriter(run_dir / TELEMETRY_NAME, fsync_every)
        self._planner = _JsonlWriter(run_dir / PLANNER_NAME, fsync_every)
        if cfg.flightlog.write_frames:
            self._frames_blob = (run_dir / FRAMES_BLOB_NAME).open("wb")

        self._meta: dict[str, Any] = {
            "run_id": run_dir.name,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "t_start_ns": time.monotonic_ns(),
            "git": git_revision(),
            "config_source": cfg.source_path,
            "calibration_sha256": calibration_hash,
            "write_frames": cfg.flightlog.write_frames,
        }
        self._write_meta()

    @classmethod
    def create(
        cls, cfg: Config, root: str | Path | None = None, run_id: str | None = None
    ) -> FlightLog:
        base = Path(root) if root is not None else Path(cfg.flightlog.root)
        return cls(base / (run_id or new_run_id()), cfg)

    def __enter__(self) -> FlightLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _write_meta(self) -> None:
        (self.run_dir / META_NAME).write_text(
            json.dumps(self._meta, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    def _check_open(self) -> None:
        if self._closed:
            raise FlightLogError("flight log is closed")

    def write_frame(self, seq: int, t_ns: int, mjpg: bytes) -> None:
        """Append a frame's MJPG bitstream and index it.

        Byte-for-byte what the camera produced; nothing is decoded here.
        """
        self._check_open()
        if self._frames_blob is None:
            return
        self._frames_blob.write(mjpg)
        self._frames_index.write(
            {
                "seq": int(seq),
                "t_ns": int(t_ns),
                "offset": self._frames_offset,
                "length": len(mjpg),
            }
        )
        self._frames_offset += len(mjpg)

    def write_telemetry(self, state: FcState) -> None:
        self._check_open()
        self._telemetry.write(encode_fc_state(state))

    def write_planner_tick(
        self,
        t_ns: int,
        command: PlannerCommand,
        snapshot: OccupancySnapshot | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log one planner decision: inputs, decision, and why.

        CLAUDE.md requires every decision-making module to log inputs ->
        decision, so ``snapshot`` (what the planner saw) and ``command.reason``
        (why it chose that) are both part of the record, not optional colour.
        """
        self._check_open()
        record: dict[str, Any] = {
            "t_ns": int(t_ns),
            "command": encode_planner_command(command),
            "snapshot": None if snapshot is None else encode_occupancy(snapshot),
        }
        if extra:
            record["extra"] = extra
        self._planner.write(record)

    def write_metrics(self, metrics: Metrics) -> None:
        self._check_open()
        (self.run_dir / METRICS_NAME).write_text(
            json.dumps(metrics.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    def close(self) -> None:
        if self._closed:
            return
        self._meta["t_end_ns"] = time.monotonic_ns()
        self._meta["closed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._meta["counts"] = {
            "frames": self._frames_index.count,
            "telemetry": self._telemetry.count,
            "planner": self._planner.count,
        }
        for writer in (self._frames_index, self._telemetry, self._planner):
            writer.close()
        if self._frames_blob is not None:
            self._frames_blob.flush()
            self._frames_blob.close()
        self._write_meta()
        self._closed = True


class FlightLogReader:
    """Reader for a run directory written by FlightLog."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_dir():
            raise FlightLogError(f"run directory not found: {self.run_dir}")
        meta_path = self.run_dir / META_NAME
        if not meta_path.is_file():
            raise FlightLogError(f"{self.run_dir} has no {META_NAME}; not a run directory")
        self.meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))

    @property
    def run_id(self) -> str:
        return str(self.meta.get("run_id", self.run_dir.name))

    def raw_config(self) -> dict[str, Any]:
        """The YAML-shaped config mapping this run used."""
        path = self.run_dir / CONFIG_YAML_NAME
        if not path.is_file():
            raise FlightLogError(f"{self.run_dir} has no {CONFIG_YAML_NAME}")
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def config(self) -> Config:
        """Rebuild the exact Config the run was flown with."""
        from config import load_config_from_dict

        raw = self.raw_config()
        calibration = raw.get("camera", {}).get("calibration_npz")
        if calibration is not None and not Path(calibration).is_absolute():
            raw["camera"]["calibration_npz"] = str((self.run_dir / calibration).resolve())
        return load_config_from_dict(raw, str(self.run_dir / CONFIG_YAML_NAME))

    def _iter_jsonl(self, name: str) -> Iterator[dict[str, Any]]:
        path = self.run_dir / name
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def frames(self) -> Iterator[FrameRecord]:
        """Frames in logged order, each with its MJPG bytes.

        Reads the blob with explicit seeks off the index rather than assuming
        the index is contiguous, so a truncated or reordered log still yields
        the frames it can address.
        """
        blob_path = self.run_dir / FRAMES_BLOB_NAME
        if not blob_path.is_file():
            return
        with blob_path.open("rb") as blob:
            for record in self._iter_jsonl(FRAMES_INDEX_NAME):
                offset = int(record["offset"])
                length = int(record["length"])
                blob.seek(offset)
                payload = blob.read(length)
                if len(payload) != length:
                    raise FlightLogError(
                        f"frame seq={record['seq']} truncated: wanted {length} bytes at "
                        f"offset {offset}, got {len(payload)}"
                    )
                yield FrameRecord(
                    seq=int(record["seq"]),
                    t_ns=int(record["t_ns"]),
                    offset=offset,
                    length=length,
                    mjpg=payload,
                )

    def telemetry(self) -> Iterator[FcState]:
        for record in self._iter_jsonl(TELEMETRY_NAME):
            yield decode_fc_state(record)

    def planner_ticks(self) -> Iterator[dict[str, Any]]:
        yield from self._iter_jsonl(PLANNER_NAME)

    def planner_commands(self) -> Iterator[PlannerCommand]:
        for record in self.planner_ticks():
            yield decode_planner_command(record["command"])

    def metrics(self) -> dict[str, Any]:
        path = self.run_dir / METRICS_NAME
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
