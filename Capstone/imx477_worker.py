"""
imx477_worker.py
----------------
CameraWorker-compatible class for the Arducam IMX477 connected via CSI
to the Raspberry Pi 5 CAM/DISP port 1.

Uses picamera2 (the correct Pi 5 native stack) instead of OpenCV V4L2.
Captures at 1332x990 @ 120fps (the sensor's fastest native mode).
Detection runs on a 333x248 downscaled image (1/4 pixels = ~4x faster).

Public interface is IDENTICAL to CameraWorker so main.py needs no changes:
    get_result()       → dict with rvecs, tvecs, frame, timestamp, fps, alive
    set_draw_axes(bool)
    get_draw_axes()
    stop()

The result dict also carries 'priority_weight' = 3.0 so pose_fusion.py
can give this camera higher influence than the USB backup cameras.
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import threading
import time
import json
import os

try:
    from picamera2 import Picamera2
    import libcamera
    _PICAMERA2_OK = True
except ImportError:
    _PICAMERA2_OK = False


# Native 120fps mode for IMX477 on Pi 5
_CAPTURE_W = 1332
_CAPTURE_H = 990
_CAPTURE_FPS = 120


class IMX477Worker:
    """
    Primary camera worker using the IMX477 CSI sensor at 120fps.
    Drop-in replacement for CameraWorker in the cameras list.
    """

    PRIORITY_WEIGHT = 3.0   # per detected marker, vs 1.0 for USB cams

    def __init__(self, camera_id: str = "imx477", calibration_file: str = None):
        if not _PICAMERA2_OK:
            raise RuntimeError("picamera2 not available — cannot use IMX477")

        self.camera_id = camera_id

        # ── Marker geometry ──────────────────────────────────────────────────
        self.marker_size = 0.015
        self.valid_ids   = {0, 1, 2, 3}
        s = self.marker_size / 2
        self.obj_pts = np.array(
            [[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32
        )

        # ── Calibration ──────────────────────────────────────────────────────
        self.cam_mat, self.dist = self._load_calibration(calibration_file)

        # ── ArUco detector ───────────────────────────────────────────────────
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        params = aruco.DetectorParameters()
        params.cornerRefinementMethod        = aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementMaxIterations = 10
        params.cornerRefinementMinAccuracy   = 0.1
        params.adaptiveThreshWinSizeMin      = 7
        params.adaptiveThreshWinSizeMax      = 23
        params.adaptiveThreshWinSizeStep     = 10
        params.minMarkerPerimeterRate        = 0.06
        params.maxMarkerPerimeterRate        = 4.0
        params.polygonalApproxAccuracyRate   = 0.04
        self.detector = aruco.ArucoDetector(self.aruco_dict, params)

        # ── picamera2 setup ───────────────────────────────────────────────────
        # Camera index 0 = the IMX477 (only CSI camera present)
        self._picam = Picamera2(0)

        # Use the 1332x990 @ 120fps sensor mode (mode 0 = SRGGB10_CSI2P)
        # Request BGR888 output so OpenCV can use it directly (no conversion)
        sensor_mode = self._picam.sensor_modes[0]   # 1332x990 120fps mode
        config = self._picam.create_video_configuration(
            main={"size": (_CAPTURE_W, _CAPTURE_H), "format": "RGB888"},  # RGB; we convert to BGR after capture
            sensor={"output_size": sensor_mode["size"],
                    "bit_depth":   sensor_mode["bit_depth"]},
            controls={
                "FrameDurationLimits": (
                    int(1e6 / _CAPTURE_FPS),   # min frame duration µs
                    int(1e6 / _CAPTURE_FPS),   # max frame duration µs
                ),
                "NoiseReductionMode": 0,   # off — fastest ISP path
                "Sharpness":          1.0,
                "ExposureTime":       4000,  # 4ms fixed — avoids AE hunting
                "AnalogueGain":       4.0,
            },
            buffer_count=4,    # 4 buffers: ISP fills ahead, we drain latest
        )
        self._picam.configure(config)
        self._picam.start()

        # Let AE/AWB settle for a few frames then lock exposure
        time.sleep(0.3)

        actual_fps = self._picam.capture_metadata().get("FrameDuration", 0)
        if actual_fps:
            actual_fps = 1e6 / actual_fps
        print(f"  [{camera_id}] IMX477 {_CAPTURE_W}x{_CAPTURE_H} "
              f"@ {actual_fps:.0f}fps  (CSI native)")

        # ── Detection downscale ───────────────────────────────────────────────
        # Detect on 1/4 area (half linear): 666x495
        self.DETECT_SCALE = 0.5
        self._detect_w    = int(_CAPTURE_W * self.DETECT_SCALE)
        self._detect_h    = int(_CAPTURE_H * self.DETECT_SCALE)

        # Scaled camera matrix for downscaled detection
        self._cam_mat_small = self.cam_mat.copy()
        self._cam_mat_small[0, 0] *= self.DETECT_SCALE
        self._cam_mat_small[1, 1] *= self.DETECT_SCALE
        self._cam_mat_small[0, 2] *= self.DETECT_SCALE
        self._cam_mat_small[1, 2] *= self.DETECT_SCALE

        # ── Shared state ─────────────────────────────────────────────────────
        self._latest_frame = None
        self._frame_lock   = threading.Lock()
        self._new_frame    = threading.Event()

        self._result_lock  = threading.Lock()
        self._result = {
            'rvecs':           {},
            'tvecs':           {},
            'frame':           None,
            'timestamp':       time.perf_counter(),
            'fps':             0.0,
            'alive':           False,
            'priority_weight': self.PRIORITY_WEIGHT,
        }

        self._running    = True
        self._draw_axes  = True

        self._grab_thread = threading.Thread(
            target=self._grabber,   daemon=True, name=f"grab-{camera_id}")
        self._proc_thread = threading.Thread(
            target=self._processor, daemon=True, name=f"proc-{camera_id}")
        self._grab_thread.start()
        self._proc_thread.start()

        if not self._new_frame.wait(timeout=3.0):
            print(f"  [{camera_id}] WARNING: no frames in 3s")
        else:
            print(f"  [{camera_id}] first frame received OK")

    # ── Public API (identical to CameraWorker) ────────────────────────────────

    def get_result(self) -> dict:
        with self._result_lock:
            return dict(self._result)

    def set_draw_axes(self, val: bool):
        self._draw_axes = val

    def get_draw_axes(self) -> bool:
        return self._draw_axes

    def stop(self):
        self._running = False
        self._new_frame.set()
        self._grab_thread.join(timeout=2)
        self._proc_thread.join(timeout=2)
        self._picam.stop()
        self._picam.close()
        print(f"  [{self.camera_id}] stopped.")

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_calibration(self, path):
        if path and os.path.exists(path):
            with open(path) as f:
                c = json.load(f)
            cam_mat = np.array(c['camera_matrix'],           dtype=np.float32)
            dist    = np.array(c['distortion_coefficients'], dtype=np.float32).flatten()
            print(f"  [{self.camera_id}] calibration loaded from {path} "
                  f"(err: {c.get('reprojection_error', '?'):.4f}px)")
        else:
            # IMX477 default at 1332x990 crop mode
            # fx/fy ~ 0.8 * full-sensor focal length, cx/cy = half frame
            cam_mat = np.array(
                [[1400, 0, 666], [0, 1400, 495], [0, 0, 1]], dtype=np.float32
            )
            dist = np.zeros(5, dtype=np.float32)
            print(f"  [{self.camera_id}] WARNING: using estimated IMX477 matrix "
                  f"— calibrate for best accuracy")
        return cam_mat, dist

    def _grabber(self):
        """
        Thread A: capture frames from picamera2 as fast as possible.
        capture_array() returns the latest ISP-processed frame as a numpy
        array — no USB, no JPEG decode overhead, direct DMA from ISP.
        """
        while self._running:
            try:
                # capture_array blocks until the next frame is ready
                # At 120fps that's ~8.3ms — much faster than USB cams
                frame = self._picam.capture_array("main")
                # picamera2 delivers RGB — convert to BGR for OpenCV
                frame_bgr = frame[:, :, ::-1].copy()  # RGB→BGR channel swap
                with self._frame_lock:
                    self._latest_frame = frame_bgr
                self._new_frame.set()
            except Exception as e:
                print(f"  [{self.camera_id}] grab error: {e}")
                time.sleep(0.01)

    def _processor(self):
        """
        Thread B: ArUco detection on latest frame, publish results.
        Identical pipeline to CameraWorker._processor.
        """
        _fps_count = 0
        _fps_time  = time.perf_counter()
        _fps_val   = 0.0

        while self._running:
            if not self._new_frame.wait(timeout=0.1):
                continue
            self._new_frame.clear()

            if not self._running:
                break

            with self._frame_lock:
                if self._latest_frame is None:
                    continue
                frame = self._latest_frame.copy()

            # ── Downscale for fast detection ──────────────────────────────────
            gray_full  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray_full,
                                    (self._detect_w, self._detect_h),
                                    interpolation=cv2.INTER_LINEAR)

            # ── ArUco detection ───────────────────────────────────────────────
            corners_small, ids, _ = self.detector.detectMarkers(gray_small)

            rvecs_out = {}
            tvecs_out = {}

            if ids is not None:
                flat_ids     = ids.flatten()
                inv          = 1.0 / self.DETECT_SCALE
                corners_full = [c * inv for c in corners_small]

                aruco.drawDetectedMarkers(frame, corners_full, ids)

                for i, mid in enumerate(flat_ids):
                    if mid not in self.valid_ids:
                        continue
                    ok, rvec, tvec = cv2.solvePnP(
                        self.obj_pts,
                        corners_full[i][0],
                        self.cam_mat,
                        self.dist,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                    if ok:
                        rvecs_out[int(mid)] = rvec
                        tvecs_out[int(mid)] = tvec
                        if self._draw_axes:
                            cv2.drawFrameAxes(
                                frame, self.cam_mat, self.dist,
                                rvec, tvec, 0.02, 2)

            # ── FPS ───────────────────────────────────────────────────────────
            _fps_count += 1
            if _fps_count >= 30:
                now        = time.perf_counter()
                _fps_val   = 30 / (now - _fps_time)
                _fps_time  = now
                _fps_count = 0

            detected_str = str(sorted(rvecs_out.keys())) if rvecs_out else "none"
            # Draw label on a downscaled preview (1332x990 is large for display)
            preview = cv2.resize(frame, (640, 480))
            cv2.putText(preview,
                        f"{self.camera_id} [PRIMARY] | {_fps_val:.0f}fps | {detected_str}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # ── Publish ───────────────────────────────────────────────────────
            with self._result_lock:
                self._result['rvecs']           = rvecs_out
                self._result['tvecs']           = tvecs_out
                self._result['frame']           = preview   # 640x480 for display
                self._result['timestamp']       = time.perf_counter()
                self._result['fps']             = _fps_val
                self._result['alive']           = True
                self._result['priority_weight'] = self.PRIORITY_WEIGHT