"""
smoother.py
-----------
All temporal filtering for the fused 6-DOF pose:
  1. Median filter over a rolling window    (removes outlier spikes)
  2. Step clamping                          (prevents unrealistic jumps)
  3. Exponential moving average (EMA)       (low-pass, removes jitter)
  4. Deadzone zeroing                       (suppresses noise at rest)
"""

import numpy as np
from collections import deque


class PoseSmoother:
    def __init__(self):
        # Median window
        self.MEDIAN_WIN   = 9
        self._history     = deque(maxlen=self.MEDIAN_WIN)

        # EMA
        self.alpha        = 0.08        # lower = smoother, more lag

        # Step clamp
        self.MAX_POS_STEP   = 8.0       # mm per frame
        self.MAX_ANGLE_STEP = 3.0       # degrees per frame
        self.max_confidence = 20

        # State
        self._filtered    = None        # last EMA value
        self._confidence  = 0

    # ─────────────────────────────────────────────────────────────────────────

    @property
    def confidence(self):
        return self._confidence

    def update(self, raw: np.ndarray) -> np.ndarray:
        """
        Feed a new raw pose, get back the smoothed pose.
        raw: np.array([X, Y, Z, roll, pitch, yaw])
        """
        self._history.append(raw)

        # 1. Median
        if len(self._history) >= 3:
            median = np.median(self._history, axis=0)
        else:
            median = raw.copy()

        # 2. Step clamp
        if self._filtered is not None:
            median = self._clamp_step(median, self._filtered)

        # 3. EMA
        if self._filtered is None:
            self._filtered = median.copy()
        else:
            self._filtered = self.alpha * median + (1 - self.alpha) * self._filtered

        # 4. Confidence ramp
        self._confidence = min(self._confidence + 1, self.max_confidence)

        # 5. Deadzone
        return self._deadzone(self._filtered)

    def decay(self) -> np.ndarray | None:
        """
        Call when no valid pose was computed this frame (markers lost).
        Reduces confidence; returns last filtered pose or None.
        """
        self._confidence = max(0, self._confidence - 1)
        if self._filtered is not None:
            return self._deadzone(self._filtered)
        return None

    def reset(self):
        self._history.clear()
        self._filtered   = None
        self._confidence = 0

    # ─────────────────────────────────────────────────────────────────────────

    def _clamp_step(self, new_pose, prev_pose):
        scale = 1.0 + (self._confidence / self.max_confidence) * 3.0
        pos_lim = self.MAX_POS_STEP   * scale
        ang_lim = self.MAX_ANGLE_STEP * scale
        out = new_pose.copy()
        delta = new_pose - prev_pose
        for i, lim in enumerate([pos_lim]*3 + [ang_lim]*3):
            if abs(delta[i]) > lim:
                out[i] = prev_pose[i] + np.sign(delta[i]) * lim
        return out

    def _deadzone(self, pose):
        out = pose.copy()
        out[:3][np.abs(out[:3]) < 0.4]  = 0.0   # < 0.4 mm → zero
        out[3:][np.abs(out[3:]) < 0.25] = 0.0   # < 0.25° → zero
        return out