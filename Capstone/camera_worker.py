"""
camera_worker.py
----------------
One CameraWorker per EMEET S600 USB webcam.

Two threads per camera:
  _grabber   : blocking grab()+retrieve() — one frame at a time, no drain loop
  _processor : ArUco detection + solvePnP, publishes results

Root cause of "cameras dead" was the drain loop:
    while self.cap.grab(): ...   ← returns False immediately if buffer empty
    grabbed stays False → retrieve() never called → no frames ever published

Fix: single blocking grab() which waits up to the frame interval, then retrieve().
This reliably delivers ~30fps per camera (S600 Linux firmware limit).
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import threading
import time
import json
import os


class CameraWorker:
    def __init__(self, cam_index: int, camera_id: str,
                 calibration_file: str = None):
        self.cam_index = cam_index
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

        # ── Force format at kernel level before OpenCV opens device ──────────
        dev = f"/dev/video{cam_index}"
        os.system(
            f"v4l2-ctl --device={dev} "
            f"--set-fmt-video=width=640,height=360,pixelformat=MJPG "
            f"--set-parm=60 2>/dev/null"
        )

        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.cap.set(cv2.CAP_PROP_FPS,          60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # minimal latency

        if not self.cap.isOpened():
            raise RuntimeError(
                f"[{camera_id}] Cannot open /dev/video{cam_index}"
            )

        actual_w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_cc  = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((actual_cc >> 8*i) & 0xFF) for i in range(4)])
        print(f"  [{camera_id}] /dev/video{cam_index}  "
              f"{actual_w}x{actual_h} @ {actual_fps:.0f}fps  {fourcc_str}")

        # Half-res detection (320×180) — ~4x faster, full-res solvePnP
        self.DETECT_SCALE = 0.5
        self._detect_w    = int(actual_w * self.DETECT_SCALE)
        self._detect_h    = int(actual_h * self.DETECT_SCALE)

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
            'rvecs':     {},
            'tvecs':     {},
            'frame':     None,
            'timestamp': time.perf_counter(),
            'fps':       0.0,
            'alive':     False,
        }

        self._running   = True
        self._draw_axes = True

        self._grab_thread = threading.Thread(
            target=self._grabber, daemon=True, name=f"grab-{camera_id}")
        self._proc_thread = threading.Thread(
            target=self._processor, daemon=True, name=f"proc-{camera_id}")
        self._grab_thread.start()
        self._proc_thread.start()

        # Wait up to 5 s for first frame
        if not self._new_frame.wait(timeout=5.0):
            print(f"  [{camera_id}] WARNING: no frames in 5 s — "
                  f"check cable and index")
        else:
            print(f"  [{camera_id}] ready")

    # ── Public API ────────────────────────────────────────────────────────────

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
        self.cap.release()
        print(f"  [{self.camera_id}] stopped.")

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_calibration(self, path):
        if path and os.path.exists(path):
            with open(path) as f:
                c = json.load(f)
            cam_mat = np.array(c['camera_matrix'],           dtype=np.float32)
            dist    = np.array(c['distortion_coefficients'], dtype=np.float32).flatten()
            print(f"  [{self.camera_id}] calibration loaded "
                  f"(err:{c.get('reprojection_error','?'):.4f}px)")
        else:
            cam_mat = np.array([[800,0,320],[0,800,180],[0,0,1]], dtype=np.float32)
            dist    = np.zeros(5, dtype=np.float32)
            print(f"  [{self.camera_id}] using default calibration")
        return cam_mat, dist

    def _grabber(self):
        """
        Thread A — blocking grab+retrieve loop.
        Single grab() blocks until camera delivers next frame (~33ms at 30fps).
        This is reliable on Pi where the drain-loop approach fails.
        """
        while self._running:
            if not self.cap.grab():
                time.sleep(0.005)
                continue
            ret, frame = self.cap.retrieve()
            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
                self._new_frame.set()

    def _processor(self):
        """Thread B — detect ArUco markers, run solvePnP, publish results."""
        _fps_count  = 0
        _fps_time   = time.perf_counter()
        _fps_val    = 0.0
        _last_frame = None

        while self._running:
            if not self._new_frame.wait(timeout=0.1):
                continue
            self._new_frame.clear()
            if not self._running:
                break

            with self._frame_lock:
                frame = self._latest_frame
            if frame is None or frame is _last_frame:
                continue
            _last_frame = frame
            frame = frame.copy()

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
                        self.obj_pts, corners_full[i][0],
                        self.cam_mat, self.dist,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                    if ok:
                        rvecs_out[int(mid)] = rvec
                        tvecs_out[int(mid)] = tvec
                        if self._draw_axes:
                            cv2.drawFrameAxes(frame, self.cam_mat, self.dist,
                                              rvec, tvec, 0.02, 2)

            # ── FPS ───────────────────────────────────────────────────────────
            _fps_count += 1
            if _fps_count >= 30:
                now        = time.perf_counter()
                _fps_val   = 30 / (now - _fps_time)
                _fps_time  = now
                _fps_count = 0

            detected_str = str(sorted(rvecs_out.keys())) if rvecs_out else "none"
            cv2.putText(frame,
                        f"{self.camera_id} | {_fps_val:.0f}fps | {detected_str}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            with self._result_lock:
                self._result['rvecs']     = rvecs_out
                self._result['tvecs']     = tvecs_out
                self._result['frame']     = frame
                self._result['timestamp'] = time.perf_counter()
                self._result['fps']       = _fps_val
                self._result['alive']     = True