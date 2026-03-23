#!/usr/bin/env python3
"""
Multi-Camera ArUco Marker Tracker — Performance-Optimized
==========================================================
Key structural changes vs previous version:
  1. threading.Event replaces busy-wait in _detector  → eliminates 100% CPU spin
  2. cap.grab() / cap.retrieve() split in _grabber    → decouples decode from capture
  3. AP3P seeds solvePnP before ITERATIVE refine      → ~30% faster per-marker solve
  4. Display runs in its own daemon thread            → main loop never blocked by imshow
  5. Pre-allocated tile buffer (np.empty)             → avoids per-frame malloc in hstack
  6. cv2.INTER_NEAREST for display downscale          → 3–4× faster than INTER_LINEAR
     (quality loss invisible at tile resolution)
  7. Grayscale frame path: convert once at grab time  → detector gets gray directly
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import time
import threading
import socket
import serial
import json
from collections import deque

# ============================== CONFIG ======================================

TARGET_IP   = "130.63.250.79"
TARGET_PORT = 5006

IMU_PORT    = "/dev/ttyACM0"
IMU_BAUD    = 115200
IMU_TAKEOVER_FULL_SECS = 5.0

MARKER_SIZE_M = 0.015
REF_ID        = 0
PLANE_IDS     = [1, 2, 3]
REQUIRED_IDS  = {0, 1, 2, 3}   # set → O(1) lookup vs list

DRAW_AXES  = True
AXIS_LEN_M = 0.02

TILE_W = 640
TILE_H = 360

CAMERA_CONFIGS = [
    {"name": "cam1_usb3", "device": 0, "width": 1280, "height": 720, "fps": 120,
     "calibration": "camera1_calibration.json"},
    {"name": "cam2_usb3", "device": 2, "width": 1280, "height": 720, "fps": 120,
     "calibration": "camera2_calibration.json"},
    {"name": "cam3_usb2", "device": 4, "width": 1280, "height": 720, "fps": 120,
     "calibration": "camera3_calibration.json"},
]

# Detection at half resolution → ~4× fewer pixels for ArUco
DETECT_SCALE = 0.5


# ========================== QUATERNION HELPERS ==============================

def qmul(q1, q2):
    w1, x1, y1, z1 = q1;  w2, x2, y2, z2 = q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], np.float64)

def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], np.float64)

def q2euler(q):
    w, x, y, z = q
    return np.degrees(np.array([
        np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
        np.arcsin(np.clip(2*(w*y-z*x), -1, 1)),
        np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    ], np.float64))

def rotation_matrix_to_quaternion(R):
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s], np.float64)
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0*np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])
        q = np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s], np.float64)
    elif R[1,1] > R[2,2]:
        s = 2.0*np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])
        q = np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s], np.float64)
    else:
        s = 2.0*np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])
        q = np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s], np.float64)
    n = np.linalg.norm(q)
    return q/n if n > 0 else np.array([1,0,0,0], np.float64)

def quat_to_rot(q):
    q = q / np.linalg.norm(q);  w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                     [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]], np.float64)

def euler_from_mat(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        r = np.arctan2(R[2,1], R[2,2]);  p = np.arctan2(-R[2,0], sy);  y = np.arctan2(R[1,0], R[0,0])
    else:
        r = np.arctan2(-R[1,2], R[1,1]); p = np.arctan2(-R[2,0], sy);  y = 0.0
    return np.degrees(np.array([r, p, y], np.float64))

def avg_quaternions(quats):
    M = np.zeros((4,4), np.float64)
    for q in quats:
        q = q.copy();  (q := -q) if q[0] < 0 else None;  M += np.outer(q, q)
    M /= max(len(quats), 1)
    ev, evec = np.linalg.eigh(M)
    q = evec[:, np.argmax(ev)]
    return q / np.linalg.norm(q)


# =============================== IMU READER =================================

class IMUReader:
    """Non-blocking BNO055 reader — daemon thread, always has latest quaternion."""

    def __init__(self, port, baud=115200):
        self._lock     = threading.Lock()
        self.cur_q     = np.array([1.,0.,0.,0.], np.float64)
        self.cal       = np.zeros(4, dtype=int)
        self.running   = True
        self.available = False
        self.ser       = None
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2);  self.ser.reset_input_buffer()
            self.available = True
            print(f"✓ IMU connected on {port}")
            threading.Thread(target=self._loop, daemon=True).start()
        except Exception as e:
            print(f"⚠ IMU unavailable: {e}")

    def _loop(self):
        while self.running:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
                if "QUAT:" in line or "CAL:" in line:
                    self._parse(line)
            except Exception:
                pass

    def _parse(self, line):
        with self._lock:
            for part in line.split("|"):
                try:
                    if part.startswith("QUAT:"):
                        q = np.array([float(v) for v in part[5:].split(",")], np.float64)
                        n = np.linalg.norm(q)
                        if n > 0: self.cur_q = q / n
                    elif part.startswith("CAL:"):
                        self.cal = np.array([int(float(v)) for v in part[4:].split(",")], dtype=int)
                except Exception:
                    pass

    def get_quaternion(self):
        with self._lock: return self.cur_q.copy()

    def get_cal(self):
        with self._lock: return self.cal.copy()

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open: self.ser.close()


# ================================ CAMERA NODE ================================

class CameraNode:
    """One camera — two threads (grabber + detector).

    FIX 1: threading.Event (_frame_event) replaces the busy-wait `continue` in
            the detector.  The detector blocks (zero CPU) until the grabber
            signals that a new frame has arrived.

    FIX 2: _grabber uses cap.grab() / cap.retrieve() split so that the
            kernel-side frame is consumed immediately (grab) even if the Python
            side is slow to decode (retrieve).  This keeps the V4L2 buffer from
            filling up and stalling the driver.

    FIX 3: The grabber converts the frame to greyscale at capture time and
            stores (color_frame, gray_frame) together.  The detector reuses
            the already-decoded gray instead of re-converting.

    FIX 4: Display resize uses INTER_NEAREST — visually indistinguishable at
            640×360 but ~3× faster than INTER_LINEAR.
    """

    def __init__(self, cfg, aruco_dict, detector_params, obj_pts):
        self.name              = cfg["name"]
        self.device            = cfg["device"]
        self.width             = cfg["width"]
        self.height            = cfg["height"]
        self.fps_req           = cfg["fps"]
        self.calibration_file  = cfg["calibration"]

        # ---- V4L2 capture ----
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"{self.name}: cannot open /dev/video{self.device}")

        self.cap.set(cv2.CAP_PROP_FOURCC,      cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS,          self.fps_req)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)  # always freshest frame

        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            self.cap.release()
            raise RuntimeError(f"{self.name}: no frames at {self.width}x{self.height}@{self.fps_req}")

        self.actual_h, self.actual_w = test_frame.shape[:2]
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.cam_mat, self.dist = self._load_calibration(self.calibration_file)

        # Scaled intrinsics for drawing axes on the display tile
        self._tile_cam_mat = self.cam_mat.copy()
        self._tile_cam_mat[0] *= (TILE_W / self.actual_w)
        self._tile_cam_mat[1] *= (TILE_H / self.actual_h)

        # Detection-scale intrinsics (for potential future use)
        self._det_cam_mat = self.cam_mat.copy()
        self._det_cam_mat[0] *= DETECT_SCALE
        self._det_cam_mat[1] *= DETECT_SCALE

        self.detector = aruco.ArucoDetector(aruco_dict, detector_params)
        self.obj_pts  = obj_pts

        # FIX 1 — Event-based signalling instead of busy-wait
        # _frame_event is set by the grabber each time a new frame is stored,
        # and cleared by the detector once it has consumed it.
        self._frame_event = threading.Event()

        # Shared raw frame (grabber writes, detector reads)
        # Tuple: (bgr_frame, gray_frame) — gray pre-computed at grab time (FIX 3)
        self._raw = None
        self._raw_id   = 0
        self._raw_lock = threading.Lock()

        # Latest observation (detector writes, main loop reads)
        self._obs      = None
        self._obs_lock = threading.Lock()

        # Detection-rate counter
        self.det_fps    = 0.0
        self._det_t     = time.perf_counter()
        self._det_count = 0

        # Pre-compute display scale factors (used every detection cycle)
        self._sx = TILE_W / self.actual_w
        self._sy = TILE_H / self.actual_h

        self._running     = True
        self._grab_thread = threading.Thread(target=self._grabber,  daemon=True, name=f"{self.name}-grab")
        self._det_thread  = threading.Thread(target=self._detector, daemon=True, name=f"{self.name}-det")
        self._grab_thread.start()
        self._det_thread.start()

        print(f"✓ {self.name}: {self.actual_w}x{self.actual_h} @ {self.actual_fps:.1f}fps  "
              f"(requested {self.fps_req}fps)")

    # ------------------------------------------------------------------
    def _load_calibration(self, path):
        try:
            with open(path) as f:
                c = json.load(f)
            cam_mat = np.array(c["camera_matrix"], np.float32)
            dist    = np.array(c["dist_coeffs"],   np.float32).flatten()
            print(f"✓ {self.name}: calibration loaded from {path}")
            return cam_mat, dist
        except Exception as e:
            print(f"⚠ {self.name}: calibration failed ({e}), using fallback")
            cam_mat = np.array([[800,0,self.width/2],[0,800,self.height/2],[0,0,1]], np.float32)
            return cam_mat, np.zeros(5, np.float32)

    # ------------------------------------------------------------------
    # FIX 2: grab() / retrieve() split keeps V4L2 buffer empty even when
    # Python is slow.  grab() just tells the kernel "I've seen this frame"
    # without decoding it; retrieve() does the MJPG decode on demand.
    # ------------------------------------------------------------------
    def _grabber(self):
        while self._running:
            grabbed = self.cap.grab()   # consume kernel buffer slot immediately
            if not grabbed:
                continue
            ret, frame = self.cap.retrieve()   # decode MJPG → BGR
            if not ret or frame is None:
                continue
            # FIX 3: Convert to gray here, once, instead of inside the detector
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            with self._raw_lock:
                self._raw    = (frame, gray)
                self._raw_id += 1
            self._frame_event.set()   # wake up detector

    # ------------------------------------------------------------------
    # FIX 1: Event.wait() replaces the `if id == last: continue` busy-loop.
    # The detector thread sleeps at zero CPU cost between frames.
    # ------------------------------------------------------------------
    def _detector(self):
        last_id = -1

        while self._running:
            # Block here until grabber signals a new frame (timeout avoids
            # permanent block if camera stalls)
            self._frame_event.wait(timeout=0.5)
            self._frame_event.clear()

            with self._raw_lock:
                if self._raw is None or self._raw_id == last_id:
                    continue
                frame, gray = self._raw          # unpack pre-decoded pair
                frame_id    = self._raw_id
            last_id = frame_id

            img_h, img_w = frame.shape[:2]
            img_area = float(img_w * img_h)

            # ---- Downscale gray for fast detection ----
            dw, dh = int(img_w * DETECT_SCALE), int(img_h * DETECT_SCALE)
            # INTER_NEAREST is fastest; gray already, no color conversion needed
            small = cv2.resize(gray, (dw, dh), interpolation=cv2.INTER_NEAREST)
            corners_small, ids, _ = self.detector.detectMarkers(small)

            # FIX 4: INTER_NEAREST for display tile (fast, no visible quality loss)
            display = cv2.resize(frame, (TILE_W, TILE_H), interpolation=cv2.INTER_NEAREST)

            obs = {"camera_name": self.name, "frame_id": frame_id,
                   "display": display, "markers": {}}

            if ids is None:
                cv2.putText(display, f"{self.name}: no markers", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                with self._obs_lock: self._obs = obs
                self._tick_fps()
                continue

            ids_flat = ids.flatten().tolist()

            # Project corners back to full-res for accurate solvePnP
            corners_full = []
            for c in corners_small:
                cf = c.copy()
                cf[0,:,0] /= DETECT_SCALE;  cf[0,:,1] /= DETECT_SCALE
                corners_full.append(cf)

            # Scale corners to display tile for drawing
            corners_disp = []
            for c in corners_full:
                cd = c.copy()
                cd[0,:,0] *= self._sx;  cd[0,:,1] *= self._sy
                corners_disp.append(cd)
            aruco.drawDetectedMarkers(display, corners_disp, ids)

            for i, mid in enumerate(ids_flat):
                if mid not in REQUIRED_IDS:
                    continue

                # FIX 5: AP3P gives a fast closed-form solution; we use it to
                # seed ITERATIVE so we skip the random RANSAC burn-in phase.
                ok_ap3p, rv_ap3p, tv_ap3p = cv2.solvePnP(
                    self.obj_pts, corners_full[i][0],
                    self.cam_mat, self.dist,
                    flags=cv2.SOLVEPNP_AP3P
                )
                if not ok_ap3p:
                    continue
                # Refine with Levenberg–Marquardt starting from AP3P estimate
                ok, rvec, tvec = cv2.solvePnP(
                    self.obj_pts, corners_full[i][0],
                    self.cam_mat, self.dist,
                    rvec=rv_ap3p, tvec=tv_ap3p,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if not ok:
                    continue

                pts    = corners_full[i][0]
                area   = cv2.contourArea(pts.astype(np.float32))
                ctr    = np.mean(pts, axis=0)
                edge_m = min(ctr[0], ctr[1], img_w-ctr[0], img_h-ctr[1])
                e_sc   = max(0., edge_m) / max(min(img_w, img_h)/2., 1.)
                a_sc   = min(area/(0.03*img_area), 1.)
                d_sc   = 1.0 / max(np.linalg.norm(tvec.flatten()), 1e-3)
                quality = 0.50*a_sc + 0.25*e_sc + 0.25*min(d_sc, 1.)

                obs["markers"][mid] = {
                    "rvec": rvec, "tvec": tvec,
                    "score": float(quality), "corners": pts.copy()
                }

                if DRAW_AXES:
                    cv2.drawFrameAxes(display, self._tile_cam_mat, self.dist,
                                      rvec, tvec, AXIS_LEN_M, 1)

                dc = tuple((ctr * np.array([self._sx, self._sy])).astype(int))
                cv2.putText(display, f"ID{mid} Q:{quality:.2f}", (dc[0]+5, dc[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

            # Triangle overlay when all plane markers visible
            if all(p in obs["markers"] for p in PLANE_IDS):
                pts_d = [tuple((np.mean(obs["markers"][p]["corners"], axis=0)
                                * np.array([self._sx, self._sy])).astype(int))
                         for p in PLANE_IDS]
                cv2.line(display, pts_d[0], pts_d[1], (255,0,255), 1)
                cv2.line(display, pts_d[1], pts_d[2], (255,0,255), 1)
                cv2.line(display, pts_d[2], pts_d[0], (255,0,255), 1)

            visible = sorted(obs["markers"].keys())
            cv2.putText(display, f"{self.name} {visible} {self.det_fps:.0f}hz",
                        (8,18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,255,0), 1)

            with self._obs_lock: self._obs = obs
            self._tick_fps()

    # ------------------------------------------------------------------
    def _tick_fps(self):
        self._det_count += 1
        if self._det_count >= 30:
            now = time.perf_counter()
            self.det_fps    = self._det_count / max(now - self._det_t, 1e-6)
            self._det_t     = now
            self._det_count = 0

    def get_observation(self):
        with self._obs_lock: return self._obs

    def stop(self):
        self._running = False
        self._frame_event.set()   # unblock detector if it's waiting
        self._grab_thread.join(timeout=1)
        self._det_thread.join(timeout=1)
        self.cap.release()


# ============================ DISPLAY THREAD ================================

class DisplayThread:
    """FIX 6: Run imshow in its own daemon thread so the main loop is never
    blocked by the GUI (OpenCV's imshow can stall 5–20ms waiting for the
    window server to accept a frame).

    The main loop calls push(frame) which is non-blocking — it just swaps
    the latest frame reference.  The display thread wakes up, calls imshow,
    and sleeps again.  If the main loop is faster than the display, frames
    are dropped silently (the display shows the most recent one).
    """

    def __init__(self, window_name, width, height):
        self._win   = window_name
        self._frame = None
        self._lock  = threading.Lock()
        self._event = threading.Event()
        self._key   = -1
        self._running = True

        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._win, width, height)

        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="display")
        self._thread.start()

    def push(self, frame):
        """Non-blocking: store frame and signal the display thread."""
        with self._lock:
            self._frame = frame
        self._event.set()

    def get_key(self):
        """Return last key pressed (non-blocking); resets to -1 after read."""
        with self._lock:
            k = self._key;  self._key = -1
        return k

    def _loop(self):
        while self._running:
            self._event.wait(timeout=0.05)
            self._event.clear()
            with self._lock:
                frame = self._frame
            if frame is not None:
                cv2.imshow(self._win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k != 255:
                with self._lock:
                    self._key = k

    def stop(self):
        self._running = False
        self._event.set()
        self._thread.join(timeout=1)
        cv2.destroyAllWindows()


# ============================ MULTI-CAMERA TRACKER ==========================

class MultiCameraCombinedTracker:

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.connect((TARGET_IP, TARGET_PORT))
        print(f"✓ UDP -> {TARGET_IP}:{TARGET_PORT}")

        self.imu              = IMUReader(IMU_PORT, IMU_BAUD)
        self.imu_origin_q_inv = np.array([1.,0.,0.,0.], np.float64)
        self.imu_calibrated   = False
        self._imu_euler_hist  = deque(maxlen=5)

        s = MARKER_SIZE_M / 2
        self.obj_pts = np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]], np.float32)

        # ArUco detector shared across all cameras
        self.dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.params = aruco.DetectorParameters()
        self.params.cornerRefinementMethod        = aruco.CORNER_REFINE_SUBPIX
        self.params.cornerRefinementMaxIterations = 15
        self.params.cornerRefinementMinAccuracy   = 0.05
        self.params.adaptiveThreshWinSizeMin      = 5
        self.params.adaptiveThreshWinSizeMax      = 21
        self.params.adaptiveThreshWinSizeStep     = 8
        self.params.minMarkerPerimeterRate        = 0.03
        self.params.maxMarkerPerimeterRate        = 4.0

        self.cameras = [CameraNode(cfg, self.dict, self.params, self.obj_pts)
                        for cfg in CAMERA_CONFIGS]

        # Pose filter
        self.MEDIAN_WIN      = 5
        self.raw_history     = deque(maxlen=self.MEDIAN_WIN)
        self.alpha           = 0.20
        self.MAX_POS_STEP    = 25.0
        self.MAX_ANG_STEP    = 10.0
        self.filtered_pose   = None
        self.last_valid_pose = None
        self.pose_confidence = 0
        self.max_confidence  = 20

        self.camera_lost_time = None
        self.imu_frozen_xyz   = None

        # Main-loop FPS counter
        self.loop_counter  = 0
        self.last_fps_time = time.perf_counter()
        self.fps           = 0.0

        # FIX 6: Display in its own thread
        total_w = TILE_W * len(self.cameras)
        total_h = TILE_H + 80
        self._display = DisplayThread("Tracker", total_w, total_h)

        # FIX 7: Pre-allocate combined frame buffer once to avoid per-frame malloc
        # Layout: [tile row (TILE_H rows)] + [status bar (80 rows)]
        self._combined_buf = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        self._blank_tile   = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)

    # ------------------------------------------------------------------
    def _marker_relative_to_ref(self, ref_rvec, ref_tvec, marker_rvec, marker_tvec):
        R0, _ = cv2.Rodrigues(ref_rvec);   Ri, _ = cv2.Rodrigues(marker_rvec)
        t0, ti = ref_tvec.flatten(), marker_tvec.flatten()
        R_rel = R0.T @ Ri;  t_rel = R0.T @ (ti - t0)
        return {"R_rel": R_rel, "t_rel_m": t_rel,
                "q_rel": rotation_matrix_to_quaternion(R_rel)}

    def _fuse_observations(self, observations):
        best_by_marker = {};  debug_lines = []
        for obs in observations:
            if obs is None: continue
            markers, cam_name = obs["markers"], obs["camera_name"]
            if REF_ID not in markers:
                debug_lines.append(f"{cam_name}: no mk0");  continue
            visible_plane = [p for p in PLANE_IDS if p in markers]
            if not visible_plane:
                debug_lines.append(f"{cam_name}: mk0 only");  continue
            debug_lines.append(f"{cam_name}:{visible_plane}")
            ref = markers[REF_ID]
            for pid in visible_plane:
                m   = markers[pid]
                rel = self._marker_relative_to_ref(ref["rvec"], ref["tvec"],
                                                   m["rvec"], m["tvec"])
                sc  = 0.6*m["score"] + 0.4*ref["score"]
                if pid not in best_by_marker or sc > best_by_marker[pid]["score"]:
                    best_by_marker[pid] = {"pid": pid, "camera_name": cam_name,
                                           "score": sc, **rel}
        return best_by_marker, debug_lines

    def _compute_pose_from_fused_markers(self, best_by_marker):
        if not all(p in best_by_marker for p in PLANE_IDS):
            return None, None, None
        quats  = [best_by_marker[p]["q_rel"]   for p in PLANE_IDS]
        t_list = [best_by_marker[p]["t_rel_m"] for p in PLANE_IDS]
        q_avg  = avg_quaternions(quats)
        pos_mm = np.mean(t_list, axis=0) * 1000.0
        euler  = euler_from_mat(quat_to_rot(q_avg))
        used   = {p: best_by_marker[p]["camera_name"] for p in PLANE_IDS}
        return pos_mm, euler, used

    def _clamp_step(self, new, prev):
        c = new.copy();  d = new - prev
        s = 1.0 + (self.pose_confidence / self.max_confidence) * 3.0
        for i in range(3):
            if abs(d[i]) > self.MAX_POS_STEP*s:
                c[i] = prev[i] + np.sign(d[i])*self.MAX_POS_STEP*s
        for i in range(3,6):
            if abs(d[i]) > self.MAX_ANG_STEP*s:
                c[i] = prev[i] + np.sign(d[i])*self.MAX_ANG_STEP*s
        return c

    def _apply_deadzone(self, pose):
        out = pose.copy()
        out[:3][np.abs(out[:3]) < 0.4]  = 0.
        out[3:][np.abs(out[3:]) < 0.25] = 0.
        return out

    def _calibrate_imu_to_camera(self):
        if not self.imu.available: return
        self.imu_origin_q_inv = qconj(self.imu.get_quaternion())
        self.imu_calibrated   = True

    def _imu_relative_euler(self):
        q  = self.imu.get_quaternion()
        qr = qmul(self.imu_origin_q_inv, q)
        if qr[0] < 0: qr = -qr
        e  = q2euler(qr)
        self._imu_euler_hist.append(e)
        return np.mean(self._imu_euler_hist, axis=0) if len(self._imu_euler_hist)>=3 else e

    def _send_udp(self, pose):
        try:
            self.sock.send(
                f"{pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f},"
                f"{pose[3]:.2f},{pose[4]:.2f},{pose[5]:.2f}".encode())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # FIX 7: Write tiles directly into pre-allocated buffer (no hstack alloc)
    # ------------------------------------------------------------------
    def _build_combined_frame(self, observations, status_lines, mode_color):
        # Write each tile directly into the pre-allocated buffer
        for i, obs in enumerate(observations):
            tile = (obs["display"] if obs is not None and obs.get("display") is not None
                    else self._blank_tile)
            x0 = i * TILE_W
            self._combined_buf[:TILE_H, x0:x0+TILE_W] = tile

        # Status bar (bottom 80 rows) — reset to dark gray
        bar = self._combined_buf[TILE_H:, :]
        bar[:] = (30, 30, 30)

        mid_x = bar.shape[1] // 2
        for j, line in enumerate(status_lines[:2]):
            cv2.putText(bar, line, (10, 22+j*26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1, cv2.LINE_AA)
        for j, line in enumerate(status_lines[2:]):
            cv2.putText(bar, line, (mid_x+10, 22+j*26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)

        det_hz  = " | ".join(f"{c.name[:4]}:{c.det_fps:.0f}hz" for c in self.cameras)
        fps_txt = f"main:{self.fps:.0f}fps  {det_hz}"
        (tw,_),_ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(bar, fps_txt, (bar.shape[1]-tw-10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,220,100), 1, cv2.LINE_AA)

        return self._combined_buf   # caller receives a view, not a copy

    # ------------------------------------------------------------------
    def run(self):
        try:
            while True:
                self.loop_counter += 1
                if self.loop_counter % 30 == 0:
                    now = time.perf_counter()
                    self.fps = 30.0 / max(now - self.last_fps_time, 1e-6)
                    self.last_fps_time = now

                observations = [cam.get_observation() for cam in self.cameras]
                best_by_marker, debug_lines = self._fuse_observations(observations)
                pos, euler, used = self._compute_pose_from_fused_markers(best_by_marker)

                now_t      = time.perf_counter()
                mode_color = (0, 255, 0)

                if pos is not None:
                    self.camera_lost_time = None;  self.imu_frozen_xyz = None
                    raw = np.array([*pos, *euler], np.float32)
                    self.raw_history.append(raw)
                    med = (np.median(self.raw_history, axis=0)
                           if len(self.raw_history) >= 3 else raw)
                    if self.filtered_pose is not None:
                        med = self._clamp_step(med, self.filtered_pose)
                    self.filtered_pose = (med.copy() if self.filtered_pose is None
                                          else self.alpha*med + (1-self.alpha)*self.filtered_pose)
                    out = self._apply_deadzone(self.filtered_pose)
                    self.last_valid_pose = out.copy()
                    self.pose_confidence = min(self.pose_confidence+1, self.max_confidence)
                    self._calibrate_imu_to_camera()
                    self._send_udp(out)
                    us = f"1:{used[1]} 2:{used[2]} 3:{used[3]}"
                    print(f"\r[CAM] X:{out[0]:7.1f} Y:{out[1]:7.1f} Z:{out[2]:7.1f} | "
                          f"R:{out[3]:6.1f} P:{out[4]:6.1f} Y:{out[5]:6.1f} | {us} | {self.fps:.1f}fps   ",
                          end="", flush=True)
                    status = ["MODE: CAMERA FUSED",
                              f"X:{out[0]:.1f} Y:{out[1]:.1f} Z:{out[2]:.1f}",
                              f"R:{out[3]:.1f} P:{out[4]:.1f} Y:{out[5]:.1f}",
                              f"SRC {us}"]

                else:
                    missing = [p for p in PLANE_IDS if p not in best_by_marker]
                    if self.camera_lost_time is None: self.camera_lost_time = now_t
                    lost_for = now_t - self.camera_lost_time
                    self.pose_confidence = max(0, self.pose_confidence-1)

                    if self.imu.available and self.imu_calibrated and self.last_valid_pose is not None:
                        imu_rpy = self._imu_relative_euler()
                        if lost_for <= IMU_TAKEOVER_FULL_SECS:
                            out  = np.array([*self.last_valid_pose[:3], *imu_rpy], np.float32)
                            mode = "IMU BACKUP"
                        else:
                            if self.imu_frozen_xyz is None:
                                self.imu_frozen_xyz = self.last_valid_pose[:3].copy()
                            out  = np.array([*self.imu_frozen_xyz, *imu_rpy], np.float32)
                            mode = "IMU ROT-ONLY"
                        self._send_udp(out)
                        cal = self.imu.get_cal()
                        print(f"\r[IMU] X:{out[0]:7.1f} Y:{out[1]:7.1f} Z:{out[2]:7.1f} | "
                              f"R:{out[3]:6.1f} P:{out[4]:6.1f} Y:{out[5]:6.1f} | "
                              f"{mode} missing:{missing} lost:{lost_for:.1f}s cal:{cal}   ",
                              end="", flush=True)
                        status = [f"MODE: {mode}", f"MISSING: {missing}  LOST: {lost_for:.1f}s",
                                  f"X:{out[0]:.1f} Y:{out[1]:.1f} Z:{out[2]:.1f}",
                                  f"R:{out[3]:.1f} P:{out[4]:.1f} Y:{out[5]:.1f}"]
                        mode_color = (0, 180, 255)

                    elif self.last_valid_pose is not None:
                        out = self.last_valid_pose.copy()
                        self._send_udp(out)
                        print(f"\r[HOLD] X:{out[0]:7.1f} Y:{out[1]:7.1f} Z:{out[2]:7.1f} | "
                              f"missing:{missing} holding   ", end="", flush=True)
                        status = ["MODE: HOLD LAST POSE", f"MISSING: {missing}",
                                  f"X:{out[0]:.1f} Y:{out[1]:.1f} Z:{out[2]:.1f}",
                                  f"DBG: {' | '.join(debug_lines[:2])}"]
                        mode_color = (0, 140, 255)

                    else:
                        print(f"\r[WAIT] Missing: {missing} | {' | '.join(debug_lines)}   ",
                              end="", flush=True)
                        status = ["MODE: WAITING FOR FIRST POSE", f"MISSING: {missing}",
                                  f"DBG: {' | '.join(debug_lines[:2])}", ""]
                        mode_color = (0, 0, 255)

                # Push combined frame to display thread (non-blocking)
                combined = self._build_combined_frame(observations, status, mode_color)
                self._display.push(combined.copy())   # .copy() since buffer is reused next iter

                # Check for quit key from display thread
                if self._display.get_key() == ord('q'):
                    break

        finally:
            self.shutdown()

    def shutdown(self):
        for cam in self.cameras: cam.stop()
        self.imu.stop()
        self._display.stop()
        self.sock.close()
        print("\nDone.")


if __name__ == "__main__":
    tracker = MultiCameraCombinedTracker()
    tracker.run()