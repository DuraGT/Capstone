"""
imu_worker.py
-------------
BNO055 IMU worker — exact same protocol as the tested BNO055Tracker code.
Parses:  QUAT:w,x,y,z|CAL:sys,gyro,accel,mag

Key behaviours (matching original code):
  - set_zero()  : records current orientation as origin (like pressing O)
  - All euler angles reported RELATIVE to zeroed origin
  - 5-frame smoothing buffer
  - Auto-detects /dev/ttyUSB* or /dev/ttyACM*
"""

import serial
import numpy as np
import threading
import time
from collections import deque
import glob


class IMUWorker:

    def __init__(self, port: str = None, baudrate: int = 115200):

        # Auto-detect port
        if port is None:
            candidates = sorted(
                glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
            )
            if not candidates:
                raise RuntimeError(
                    "IMU: no serial port found — plug in USB-UART adapter "
                    "then check: ls /dev/ttyUSB* /dev/ttyACM*"
                )
            port = candidates[0]

        self._port = port
        print(f"  [IMU] opening {port} @ {baudrate} baud")

        self._ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)                       # BNO055 boot wait
        self._ser.reset_input_buffer()

        # Sensor state (protected by _lock)
        self._lock        = threading.Lock()
        self._cur_q       = np.array([1., 0., 0., 0.])
        self._cal         = np.zeros(4, dtype=int)

        # Zero/origin
        self._zeroed       = False
        self._origin_q_inv = np.array([1., 0., 0., 0.])

        # Smoothing — 5-frame mean (same as original)
        self._eh = deque(maxlen=5)

        # Yaw drift correction offset (applied on top of IMU yaw)
        self._yaw_correction = 0.0

        # FPS
        self._fps = 0.0
        self._fc  = 0
        self._ft  = time.perf_counter()

        # Published result dict
        self._result_lock = threading.Lock()
        self._result = {
            'roll':          0.0,
            'pitch':         0.0,
            'yaw':           0.0,
            'corrected_yaw': 0.0,
            'quaternion':    (1.0, 0.0, 0.0, 0.0),
            'cal':           (0, 0, 0, 0),
            'zeroed':        False,
            'timestamp':     time.perf_counter(),
            'fps':           0.0,
            'alive':         False,
        }

        self._running = True
        self._thread  = threading.Thread(
            target=self._serial_loop, daemon=True, name="imu-serial"
        )
        self._thread.start()

        # Wait up to 3 s for first packet
        for _ in range(60):
            with self._result_lock:
                if self._result['alive']:
                    print(f"  [IMU] first packet OK  (port:{port})")
                    return
            time.sleep(0.05)
        print(f"  [IMU] WARNING: no data in 3 s — check wiring/port")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_result(self) -> dict:
        with self._result_lock:
            return dict(self._result)

    def set_zero(self):
        """Record current orientation as origin. Call when IMU is at marker-0."""
        with self._lock:
            q = self._cur_q.copy()
        self._origin_q_inv = self._qconj(q)
        self._zeroed       = True
        self._eh.clear()
        self._yaw_correction = 0.0
        print("  [IMU] zeroed — origin set to current orientation")

    def clear_zero(self):
        self._zeroed = False
        print("  [IMU] zero cleared")

    def is_zeroed(self) -> bool:
        return self._zeroed

    def correct_yaw_drift(self, camera_yaw_deg: float):
        """
        Gently nudge IMU yaw toward camera yaw to prevent drift accumulation.
        Called by main.py whenever cameras are confident (confidence >= 15).
        Uses alpha=0.02 low-pass — very gradual, won't cause visible jumps.
        """
        with self._result_lock:
            raw_yaw = self._result['yaw']
        diff = camera_yaw_deg - raw_yaw
        diff = (diff + 180) % 360 - 180      # wrap to ±180
        self._yaw_correction += 0.02 * diff

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)
        if self._ser.is_open:
            self._ser.close()
        print("  [IMU] stopped.")

    # ── Serial loop ───────────────────────────────────────────────────────────

    def _serial_loop(self):
        while self._running:
            try:
                raw  = self._ser.readline()
                line = raw.decode(errors='ignore').strip()
                if 'QUAT:' in line or 'EULER:' in line:
                    self._parse(line)
            except serial.SerialException as e:
                print(f"  [IMU] serial error: {e}")
                time.sleep(0.1)
            except Exception:
                pass

    def _parse(self, line: str):
        """Parse QUAT:w,x,y,z|CAL:s,g,a,m — identical to original BNO055Tracker."""
        with self._lock:
            for part in line.split('|'):
                part = part.strip()
                try:
                    if part.startswith('QUAT:'):
                        vals = [float(v) for v in part[5:].split(',')]
                        if len(vals) == 4:
                            q = np.array(vals)
                            n = np.linalg.norm(q)
                            if n > 0.01:
                                self._cur_q = q / n
                    elif part.startswith('CAL:'):
                        vals = [int(float(v)) for v in part[4:].split(',')]
                        if len(vals) == 4:
                            self._cal = np.array(vals)
                except Exception:
                    pass

        self._publish()

    def _publish(self):
        with self._lock:
            q   = self._cur_q.copy()
            cal = self._cal.copy()

        # Relative rotation from zeroed origin (same logic as original rel_euler)
        if self._zeroed:
            qr = self._qmul(self._origin_q_inv, q)
            if qr[0] < 0:
                qr = -qr
        else:
            qr = q.copy()
            if qr[0] < 0:
                qr = -qr

        euler = self._q2euler(qr)

        # 5-frame smoothing
        self._eh.append(euler)
        smoothed = np.mean(self._eh, axis=0) if len(self._eh) >= 3 else euler

        # FPS counter
        self._fc += 1
        if self._fc >= 30:
            now       = time.perf_counter()
            self._fps = 30 / max(now - self._ft, 1e-6)
            self._ft  = now
            self._fc  = 0

        with self._result_lock:
            self._result['roll']          = float(smoothed[0])
            self._result['pitch']         = float(smoothed[1])
            self._result['yaw']           = float(smoothed[2])
            self._result['corrected_yaw'] = float(smoothed[2] + self._yaw_correction)
            self._result['quaternion']    = tuple(qr.tolist())
            self._result['cal']           = tuple(cal.tolist())
            self._result['zeroed']        = self._zeroed
            self._result['timestamp']     = time.perf_counter()
            self._result['fps']           = self._fps
            self._result['alive']         = True

    # ── Quaternion math (identical to original code) ──────────────────────────

    @staticmethod
    def _qmul(q1, q2):
        w1,x1,y1,z1 = q1;  w2,x2,y2,z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    @staticmethod
    def _qconj(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def _q2euler(q):
        w, x, y, z = q
        r  = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
        p  = np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1, 1)))
        y_ = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
        return np.array([r, p, y_])