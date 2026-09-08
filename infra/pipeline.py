"""The same CPU perception/planning object for live recording and replay.

No command emitter is constructed here. Proposed commands are diagnostic output;
calibration validation, authority enforcement and stopping are owned dependencies.
"""
from __future__ import annotations

import math
from dataclasses import asdict, replace

import numpy as np

from calib.rectify import Rectifier
from config import Config
from depth.sgbm_cpu import SgbmCpuBackend
from infra.flightlog import FlightLog
from infra.metrics import Metrics
from perception.obstacles import scan_from_depth
from perception.red_box import best_detection
from planner.behaviours import BehaviourInputs, BehaviourMachine
from sources.types import FcState, FrameBundle, PlannerCommand
from world.occupancy import OccupancyMap


class RecordPipeline:
    def __init__(self, cfg: Config, log: FlightLog | None = None,
                 metrics: Metrics | None = None) -> None:
        self.cfg = cfg
        self.log = log
        self.metrics = metrics or Metrics.from_config(cfg)
        self.rectifier = (Rectifier(cfg.camera.calibration_npz, cfg)
                          if cfg.camera.calibration_npz is not None else None)
        self.q = self.rectifier.q if self.rectifier else cfg.camera.nominal_q()
        self.rotation = (self.rectifier.rotation_body_from_rectified if self.rectifier
                         else cfg.mount.rotation_body_from_cam())
        self.backend = SgbmCpuBackend(cfg, q=self.q, metrics=self.metrics)
        self.world = OccupancyMap(cfg)
        self.machine = BehaviourMachine(cfg)
        self._last_t_ns: int | None = None
        self._last_seq: int | None = None
        self._previous_fc: FcState | None = None

    def reset(self) -> None:
        self.world.reset()
        self.machine.reset()
        self.metrics.reset()
        self._last_t_ns = None
        self._last_seq = None
        self._previous_fc = None

    def tick(self, bundle: FrameBundle, state: FcState | None) -> PlannerCommand:
        if self._last_t_ns is not None and bundle.t_ns <= self._last_t_ns:
            raise ValueError('frame timestamps must strictly increase')
        if self._last_seq is not None and bundle.seq <= self._last_seq:
            raise ValueError('frame sequence must strictly increase')
        expected = (self.cfg.camera.eye_height, self.cfg.camera.eye_width, 3)
        if bundle.left.shape != expected or bundle.right.shape != expected:
            raise ValueError('frame dimensions do not match configured geometry')
        if state is not None:
            if not 0 <= (bundle.t_ns - state.t_ns) / 1e9 <= self.cfg.runtime.telemetry_timeout_s:
                state = replace(state, mode='UNKNOWN', armed=False, ekf_ok=False, agl_m=None)
        dt_s = (self.cfg.offboard.period_s if self._last_t_ns is None
                else (bundle.t_ns - self._last_t_ns) / 1e9)
        with self.metrics.stage('pipeline.tick'):
            with self.metrics.stage('rectification'):
                left, right = (self.rectifier.apply(bundle.left, bundle.right)
                               if self.rectifier else (bundle.left, bundle.right))
            depth = self.backend.infer_rectified(left, right, bundle.t_ns)
            if self.rectifier:
                depth.valid &= self.rectifier.valid
                depth.depth_m[~depth.valid] = np.nan
                if depth.conf is not None:
                    depth.conf[~depth.valid] = 0
            with self.metrics.stage('obstacles'):
                scan = scan_from_depth(depth, self.cfg, q=self.q,
                                       rotation_body_from_cam=self.rotation)
            with self.metrics.stage('target'):
                target = best_detection(left, self.cfg, bundle.t_ns, depth, self.q,
                                        rotation_body_from_cam=self.rotation)
            # Occupancy bins are body-relative and have no ego-motion transform.
            # Retain history only when telemetry establishes an unchanged, stationary
            # pose; otherwise old evidence belongs to a different bearing/origin.
            previous = self._previous_fc
            retain = (state is not None and previous is not None
                      and state.ekf_ok and previous.ekf_ok
                      and (state.roll, state.pitch, state.yaw)
                      == (previous.roll, previous.pitch, previous.yaw)
                      and all(v == 0 for v in (*state.vel_ned, *previous.vel_ned)))
            if not retain:
                self.world.reset()
            with self.metrics.stage('planning'):
                self.world.update(scan)
                speed = 0.0 if state is None else math.sqrt(sum(v*v for v in state.vel_ned))
                snapshot = self.world.snapshot(bundle.t_ns, speed_mps=speed)
                command = self.machine.tick(BehaviourInputs(
                    t_ns=bundle.t_ns, snapshot=snapshot, target=target, fc=state, dt_s=dt_s,
                ))
            if self.log:
                self.log.write_planner_tick(bundle.t_ns, command, snapshot, extra={
                    'seq': bundle.seq, 'mode': 'record_only',
                    'geometry': 'candidate_calibration' if self.rectifier else 'nominal_offline',
                    'target': None if target is None else asdict(target),
                    'state': self.machine.state.value,
                    'transitions': [t.to_dict() for t in self.machine.transitions],
                    'occupancy_history_retained': retain,
                })
            self.machine.transitions.clear()
        self._last_t_ns = bundle.t_ns
        self._last_seq = bundle.seq
        self._previous_fc = state
        return command
