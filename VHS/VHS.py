#!/usr/bin/env python3
"""
Multi-Camera ArUco Marker Tracker with IMU Fallback
====================================================
Tracks a rigid body defined by 4 ArUco markers (IDs 0-3) using up to 3 USB
cameras simultaneously.  Pose is fused across cameras, filtered, and streamed
over UDP.  When cameras lose sight of the markers the BNO055 IMU keeps
orientation alive.

Architecture (per camera):
  Thread 1 (_grabber) – reads frames from V4L2 as fast as possible.
  Thread 2 (_detector) – ArUco detection + solvePnP on the latest frame.
  Main loop             – collects observations, fuses them, filters, UDP-sends.
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

# UDP target – receives 6-DOF pose strings ("x,y,z,roll,pitch,yaw")
TARGET_IP   = "130.63.250.79"
TARGET_PORT = 5006

# IMU serial port (BNO055 via USB-ACM or similar)
IMU_PORT    = "/dev/ttyACM0"
IMU_BAUD    = 115200
# How long (s) after camera loss before position is frozen and only rotation
# continues from the IMU (prevents position drift during long occlusions)
IMU_TAKEOVER_FULL_SECS = 5.0

# Physical marker side length in metres (used by solvePnP)
MARKER_SIZE_M = 0.015

# Marker layout: ID 0 is the reference origin; IDs 1-3 define the tracked plane
REF_ID      = 0
PLANE_IDS   = [1, 2, 3]
REQUIRED_IDS = [0, 1, 2, 3]

# Visualisation – draw a small 3-D axis on each detected marker
DRAW_AXES   = True
AXIS_LEN_M  = 0.02   # axis arm length in metres (display only)

# Each camera's live view is shrunk to this tile before being stitched
# into the combined "Tracker" window
TILE_W = 640
TILE_H = 360

# Per-camera hardware configuration
CAMERA_CONFIGS = [
    {
        "name": "cam1_usb3",
        "device": 0,            # /dev/video0
        "width": 1280,
        "height": 720,
        "fps": 120,             # ↑ request max fps from driver
        "calibration": "camera1_calibration.json",
    },
    {
        "name": "cam2_usb3",
        "device": 2,            # /dev/video2
        "width": 1280,
        "height": 720,
        "fps": 120,
        "calibration": "camera2_calibration.json",
    },
    {
        "name": "cam3_usb2",
        "device": 4,            # /dev/video4
        "width": 1280,
        "height": 720,
        "fps": 120,
        "calibration": "camera3_calibration.json",
    },
]

# ---- Speed tuning knobs ---------------------------------------------------
# ArUco detection runs on a downscaled copy of each frame.
# 0.5 = half linear resolution → ¼ pixel count → ~3-4× faster detection.
# Corners are projected back to full-res coords before solvePnP so pose
# accuracy is not affected.
DETECT_SCALE = 0.5

# ========================== QUATERNION HELPERS ==============================

def qmul(q1, q2):
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def qconj(q):
    """Quaternion conjugate (= inverse for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def q2euler(q):
    """Convert unit quaternion [w,x,y,z] → Euler angles [roll, pitch, yaw] in degrees."""
    w, x, y, z = q
    roll  = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1, 1)))
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def rotation_matrix_to_quaternion(R):
    """Convert a 3×3 rotation matrix to a unit quaternion [w,x,y,z].
    Uses Shepperd's method for numerical stability near singularities."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = np.array([0.25 / s,
                      (R[2,1] - R[1,2]) * s,
                      (R[0,2] - R[2,0]) * s,
                      (R[1,0] - R[0,1]) * s], dtype=np.float64)
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        q = np.array([(R[2,1] - R[1,2]) / s, 0.25 * s,
                      (R[0,1] + R[1,0]) / s, (R[0,2] + R[2,0]) / s], dtype=np.float64)
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        q = np.array([(R[0,2] - R[2,0]) / s, (R[0,1] + R[1,0]) / s,
                      0.25 * s, (R[1,2] + R[2,1]) / s], dtype=np.float64)
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        q = np.array([(R[1,0] - R[0,1]) / s, (R[0,2] + R[2,0]) / s,
                      (R[1,2] + R[2,1]) / s, 0.25 * s], dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1, 0, 0, 0], dtype=np.float64)


def quat_to_rot(q):
    """Convert unit quaternion [w,x,y,z] → 3×3 rotation matrix."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=np.float64)


def euler_from_mat(R):
    """Extract Euler angles (degrees) from a 3×3 rotation matrix.
    Falls back to a degenerate-case formula when the matrix is near-singular."""
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:                          # generic (non-gimbal-lock) case
        roll  = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = np.arctan2(R[1,0], R[0,0])
    else:                                  # gimbal lock – yaw set to 0
        roll  = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = 0.0
    return np.degrees(np.array([roll, pitch, yaw], dtype=np.float64))


def avg_quaternions(quats):
    """Compute the weighted average of N unit quaternions via eigendecomposition
    of the 4×4 scatter matrix.  All quaternions are forced into the same
    hemisphere (w > 0) before accumulation to avoid cancellation."""
    M = np.zeros((4, 4), dtype=np.float64)
    for q in quats:
        q = q.copy()
        if q[0] < 0:      # flip to canonical hemisphere
            q = -q
        M += np.outer(q, q)
    M /= max(len(quats), 1)
    # Largest eigenvector of M is the Fréchet mean on SO(3)
    eigvals, eigvecs = np.linalg.eigh(M)
    q = eigvecs[:, np.argmax(eigvals)]
    return q / np.linalg.norm(q)


# =============================== IMU READER =================================

class IMUReader:
    """Non-blocking serial reader for a BNO055 IMU sending lines of the form:
        QUAT:w,x,y,z|CAL:sys,gyro,accel,mag
    Runs in a daemon thread; latest quaternion is always available via
    get_quaternion()."""

    def __init__(self, port, baud=115200):
        self._lock  = threading.Lock()
        self.cur_q  = np.array([1., 0., 0., 0.], dtype=np.float64)  # identity
        self.cal    = np.zeros(4, dtype=int)
        self.running   = True
        self.available = False
        self.ser = None

        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)                      # allow Arduino to reset
            self.ser.reset_input_buffer()
            self.available = True
            print(f"✓ IMU connected on {port}")
            threading.Thread(target=self._loop, daemon=True).start()
        except Exception as e:
            print(f"⚠ IMU unavailable: {e}")

    def _loop(self):
        """Serial read loop – runs continuously in its own daemon thread."""
        while self.running:
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
                if "QUAT:" in line or "CAL:" in line:
                    self._parse(line)
            except Exception:
                pass   # ignore transient decode/IO errors

    def _parse(self, line):
        """Parse a QUAT/CAL line and update shared state under the lock."""
        with self._lock:
            for part in line.split("|"):
                try:
                    if part.startswith("QUAT:"):
                        q = np.array([float(v) for v in part[5:].split(",")], dtype=np.float64)
                        n = np.linalg.norm(q)
                        if n > 0:
                            self.cur_q = q / n   # keep normalised
                    elif part.startswith("CAL:"):
                        self.cal = np.array([int(float(v)) for v in part[4:].split(",")], dtype=int)
                except Exception:
                    pass

    def get_quaternion(self):
        """Thread-safe snapshot of the latest quaternion."""
        with self._lock:
            return self.cur_q.copy()

    def get_cal(self):
        """Thread-safe snapshot of the latest calibration status [sys,gyro,accel,mag]."""
        with self._lock:
            return self.cal.copy()

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()


# ================================ CAMERA NODE ================================

class CameraNode:
    """Manages one USB camera.

    Two daemon threads run concurrently:
      _grabber   – V4L2 capture loop, stores the latest raw frame.
      _detector  – ArUco detection + solvePnP on the latest raw frame,
                   stores a fully processed 'observation' dict.

    The main loop calls get_observation() (lock-free after the first frame)
    to receive the latest result without ever blocking.
    """

    def __init__(self, cfg, aruco_dict, detector_params, obj_pts):
        self.name       = cfg["name"]
        self.device     = cfg["device"]
        self.width      = cfg["width"]
        self.height     = cfg["height"]
        self.fps_req    = cfg["fps"]
        self.calibration_file = cfg["calibration"]

        # ---- Open V4L2 capture ----
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"{self.name}: could not open /dev/video{self.device} with V4L2")

        # MJPG gives significantly higher USB bandwidth efficiency than raw YUV
        self.cap.set(cv2.CAP_PROP_FOURCC,      cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS,          self.fps_req)
        # Buffer size 1 → always grab the newest frame, discard stale ones
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        # ---- Verify camera actually delivers frames ----
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            self.cap.release()
            raise RuntimeError(
                f"{self.name}: opened device {self.device} but could not read frames "
                f"at requested mode {self.width}x{self.height}@{self.fps_req}"
            )

        self.actual_h, self.actual_w = test_frame.shape[:2]
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        # ---- Load calibration or fall back to approximate intrinsics ----
        self.cam_mat, self.dist = self._load_calibration(self.calibration_file)

        # Pre-compute the scaled camera matrix used for axis drawing on the
        # display tile (avoids recomputing it per frame)
        self._det_cam_mat = self.cam_mat.copy()
        self._det_cam_mat[0] *= DETECT_SCALE
        self._det_cam_mat[1] *= DETECT_SCALE

        # ---- ArUco detector (shared dict & params, one instance per camera) ----
        self.detector = aruco.ArucoDetector(aruco_dict, detector_params)
        self.obj_pts  = obj_pts      # marker corner 3-D object points

        # ---- Inter-thread shared state ----
        # Raw frame (grabber → detector)
        self._raw_frame = None
        self._raw_id    = 0
        self._raw_lock  = threading.Lock()

        # Processed observation (detector → main loop)
        self._obs      = None
        self._obs_lock = threading.Lock()

        # Per-camera detection throughput counter
        self.det_fps    = 0.0
        self._det_t     = time.perf_counter()
        self._det_count = 0

        # ---- Start background threads ----
        self._running     = True
        self._grab_thread = threading.Thread(target=self._grabber,  daemon=True)
        self._det_thread  = threading.Thread(target=self._detector, daemon=True)
        self._grab_thread.start()
        self._det_thread.start()

        print(f"✓ {self.name}: dev={self.device} {self.actual_w}x{self.actual_h} "
              f"@ {self.actual_fps:.1f}fps  (requested {self.fps_req}fps)")

    # ------------------------------------------------------------------
    def _load_calibration(self, path):
        """Load camera_matrix and dist_coeffs from a JSON calibration file.
        Falls back to a reasonable pinhole estimate if the file is missing."""
        try:
            with open(path, "r") as f:
                calib = json.load(f)
            cam_mat = np.array(calib["camera_matrix"], dtype=np.float32)
            dist    = np.array(calib["dist_coeffs"],   dtype=np.float32).flatten()
            print(f"✓ {self.name}: calibration loaded from {path}")
            return cam_mat, dist
        except Exception as e:
            print(f"⚠ {self.name}: calibration load failed ({e}), using fallback intrinsics")
            # Rough estimate: focal length ≈ image width, principal point at centre
            cam_mat = np.array([[800, 0, self.width  / 2],
                                [0, 800, self.height / 2],
                                [0, 0,  1             ]], dtype=np.float32)
            dist = np.zeros(5, dtype=np.float32)
            return cam_mat, dist

    # ------------------------------------------------------------------
    # Thread 1: grab frames as fast as the camera delivers them
    # ------------------------------------------------------------------
    def _grabber(self):
        """Tight capture loop.  Stores each new frame under _raw_lock and
        increments _raw_id so the detector can detect 'new frame available'
        without copying data unnecessarily."""
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._raw_lock:
                    self._raw_frame = frame   # overwrite; detector always sees latest
                    self._raw_id   += 1

    # ------------------------------------------------------------------
    # Thread 2: detect markers on the latest raw frame
    # ------------------------------------------------------------------
    def _detector(self):
        """Detection loop.

        Steps per iteration:
          1. Grab latest raw frame (skip if unchanged).
          2. Downscale to DETECT_SCALE for fast ArUco detection.
          3. Project corners back to full-res for accurate solvePnP.
          4. Compute per-marker quality scores.
          5. Build observation dict + annotated display tile.
          6. Publish observation under _obs_lock.
        """
        last_id = -1
        # Scale factors to map full-res corners → display tile corners
        sx_disp = TILE_W / self.actual_w
        sy_disp = TILE_H / self.actual_h

        while self._running:
            # ---- 1. Fetch latest frame (skip if same as last) ----
            with self._raw_lock:
                if self._raw_frame is None or self._raw_id == last_id:
                    continue                  # busy-wait until a new frame arrives
                frame    = self._raw_frame.copy()
                frame_id = self._raw_id
            last_id = frame_id

            img_h, img_w = frame.shape[:2]
            img_area = float(img_w * img_h)

            # ---- 2. Downscale → detect ----
            dw    = int(img_w * DETECT_SCALE)
            dh    = int(img_h * DETECT_SCALE)
            small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
            # ArUco works in greyscale; converting here avoids internal conversion
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            corners_small, ids, _ = self.detector.detectMarkers(gray_small)

            # Shrink frame to display tile (done once per detection cycle)
            display = cv2.resize(frame, (TILE_W, TILE_H), interpolation=cv2.INTER_LINEAR)

            # Base observation dict (populated below)
            obs = {
                "camera_name": self.name,
                "frame_id":    frame_id,
                "display":     display,
                "markers":     {},
            }

            if ids is None:
                # No markers detected – annotate tile and publish empty obs
                cv2.putText(display, f"{self.name}: no markers", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                with self._obs_lock:
                    self._obs = obs
                self._tick_fps()
                continue

            ids_flat = ids.flatten().tolist()

            # ---- 3. Project corners back to full-res ----
            # corners_small coordinates are in [0, dw/dh]; divide by scale to
            # recover the original [0, img_w/img_h] coordinates used by solvePnP.
            corners_full = []
            for c in corners_small:
                cf = c.copy()
                cf[0, :, 0] /= DETECT_SCALE   # x
                cf[0, :, 1] /= DETECT_SCALE   # y
                corners_full.append(cf)

            # Scale corners to display tile for drawing
            corners_disp = []
            for c in corners_full:
                cd = c.copy()
                cd[0, :, 0] *= sx_disp
                cd[0, :, 1] *= sy_disp
                corners_disp.append(cd)
            aruco.drawDetectedMarkers(display, corners_disp, ids)

            # ---- 4. solvePnP + quality scoring per detected marker ----
            for i, mid in enumerate(ids_flat):
                if mid not in REQUIRED_IDS:
                    continue                  # ignore markers not part of the rig

                # Estimate pose using full-res corners + calibration
                ok, rvec, tvec = cv2.solvePnP(
                    self.obj_pts,
                    corners_full[i][0],
                    self.cam_mat,
                    self.dist,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if not ok:
                    continue

                pts  = corners_full[i][0]
                area = cv2.contourArea(pts.astype(np.float32))
                ctr  = np.mean(pts, axis=0)

                # Quality = weighted combination of three sub-scores:
                #   area_score  – larger → closer → more accurate corners
                #   edge_score  – further from frame border → less distortion
                #   dist_score  – closer to camera → smaller reprojection error
                edge_margin = min(ctr[0], ctr[1], img_w - ctr[0], img_h - ctr[1])
                edge_score  = max(0.0, edge_margin) / max(min(img_w, img_h) / 2.0, 1.0)
                area_score  = min(area / (0.03 * img_area), 1.0)
                tnorm       = np.linalg.norm(tvec.flatten())
                dist_score  = 1.0 / max(tnorm, 1e-3)
                quality     = 0.50 * area_score + 0.25 * edge_score + 0.25 * min(dist_score, 1.0)

                obs["markers"][mid] = {
                    "rvec":    rvec,
                    "tvec":    tvec,
                    "score":   float(quality),
                    "corners": pts.copy(),
                }

                # ---- 5. Draw axis on display tile ----
                if DRAW_AXES:
                    # Scale camera matrix to tile resolution for correct axis projection
                    scaled_cam_mat = self.cam_mat.copy()
                    scaled_cam_mat[0, 0] *= sx_disp;  scaled_cam_mat[1, 1] *= sy_disp
                    scaled_cam_mat[0, 2] *= sx_disp;  scaled_cam_mat[1, 2] *= sy_disp
                    cv2.drawFrameAxes(display, scaled_cam_mat, self.dist,
                                      rvec, tvec, AXIS_LEN_M, 1)

                # Label each marker with its ID and quality score
                dc = tuple((np.mean(pts, axis=0) * np.array([sx_disp, sy_disp])).astype(int))
                cv2.putText(display, f"ID{mid} Q:{quality:.2f}", (dc[0] + 5, dc[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # ---- Triangle overlay connecting the 3 plane markers ----
            # Drawn when all three plane markers are visible – shows rig outline
            if all(pid in obs["markers"] for pid in PLANE_IDS):
                pts_disp = []
                for pid in PLANE_IDS:
                    c = np.mean(obs["markers"][pid]["corners"], axis=0) * \
                        np.array([sx_disp, sy_disp])
                    pts_disp.append(tuple(c.astype(int)))
                cv2.line(display, pts_disp[0], pts_disp[1], (255, 0, 255), 1)
                cv2.line(display, pts_disp[1], pts_disp[2], (255, 0, 255), 1)
                cv2.line(display, pts_disp[2], pts_disp[0], (255, 0, 255), 1)

            # Status label: camera name, visible marker IDs, detection rate
            visible = sorted(obs["markers"].keys())
            cv2.putText(display, f"{self.name} {visible} {self.det_fps:.0f}hz",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

            # ---- 6. Publish ----
            with self._obs_lock:
                self._obs = obs

            self._tick_fps()

    # ------------------------------------------------------------------
    def _tick_fps(self):
        """Update the rolling detection-rate estimate every 30 detections."""
        self._det_count += 1
        if self._det_count >= 30:
            now = time.perf_counter()
            self.det_fps    = self._det_count / max(now - self._det_t, 1e-6)
            self._det_t     = now
            self._det_count = 0

    def get_observation(self):
        """Return the latest processed observation dict (non-blocking).
        Called from the main loop; the dict is read-only so no copy is needed."""
        with self._obs_lock:
            return self._obs

    def stop(self):
        """Signal threads to stop and release the capture device."""
        self._running = False
        self._grab_thread.join(timeout=1)
        self._det_thread.join(timeout=1)
        self.cap.release()


# ============================ MULTI-CAMERA TRACKER ==========================

class MultiCameraCombinedTracker:
    """Top-level tracker.

    Responsibilities:
      • Initialise all CameraNode instances and the IMUReader.
      • Per main-loop iteration: collect observations, fuse them, filter,
        apply deadzone, stream over UDP, and display a combined tile window.
      • Fall back to IMU rotation (and optionally frozen position) when all
        cameras lose sight of the marker rig.
    """

    def __init__(self):
        # ---- UDP socket (non-blocking send) ----
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.connect((TARGET_IP, TARGET_PORT))
        print(f"✓ UDP -> {TARGET_IP}:{TARGET_PORT}")

        # ---- IMU ----
        self.imu               = IMUReader(IMU_PORT, IMU_BAUD)
        # Inverse of the quaternion at the moment of IMU calibration.
        # Multiplying future IMU readings by this gives *relative* rotation.
        self.imu_origin_q_inv  = np.array([1., 0., 0., 0.], dtype=np.float64)
        self.imu_calibrated    = False
        self._imu_euler_hist   = deque(maxlen=5)   # short window for IMU smoothing

        # ---- Marker geometry ----
        # 3-D corners of a square marker centred at origin, in marker-local coords
        s = MARKER_SIZE_M / 2
        self.obj_pts = np.array([
            [-s,  s, 0],
            [ s,  s, 0],
            [ s, -s, 0],
            [-s, -s, 0]
        ], dtype=np.float32)

        # ---- ArUco setup ----
        self.dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.params = aruco.DetectorParameters()
        # Sub-pixel corner refinement improves solvePnP accuracy at low cost
        self.params.cornerRefinementMethod        = aruco.CORNER_REFINE_SUBPIX
        self.params.cornerRefinementMaxIterations = 15
        self.params.cornerRefinementMinAccuracy   = 0.05
        # Adaptive threshold window sweeps a small range for speed vs robustness trade-off
        self.params.adaptiveThreshWinSizeMin  = 5
        self.params.adaptiveThreshWinSizeMax  = 21
        self.params.adaptiveThreshWinSizeStep = 8
        # Marker size filter – rejects noise while allowing markers across a range of distances
        self.params.minMarkerPerimeterRate = 0.03
        self.params.maxMarkerPerimeterRate = 4.0

        # ---- Camera nodes (each starts its own threads) ----
        self.cameras = [CameraNode(cfg, self.dict, self.params, self.obj_pts)
                        for cfg in CAMERA_CONFIGS]

        # ---- Pose filter state ----
        self.MEDIAN_WIN  = 5       # rolling median window (frames) before EMA
        self.raw_history = deque(maxlen=self.MEDIAN_WIN)
        self.alpha       = 0.20    # EMA smoothing coefficient (lower = smoother/laggier)
        # Per-axis jump clamping thresholds (mm and degrees)
        self.MAX_POS_STEP = 25.0
        self.MAX_ANG_STEP = 10.0

        self.filtered_pose   = None
        self.last_valid_pose = None
        # Confidence ramp: builds up while cameras are tracking, decays when lost
        self.pose_confidence = 0
        self.max_confidence  = 20

        # IMU fallback state
        self.camera_lost_time = None   # timestamp when cameras first lost markers
        self.imu_frozen_xyz   = None   # position frozen after IMU_TAKEOVER_FULL_SECS

        # ---- Performance counters ----
        self.loop_counter  = 0
        self.last_fps_time = time.perf_counter()
        self.fps           = 0.0

        # ---- Display window ----
        cv2.namedWindow("Tracker", cv2.WINDOW_NORMAL)
        total_w = TILE_W * len(self.cameras)
        total_h = TILE_H + 80           # extra bar at bottom for status text
        cv2.resizeWindow("Tracker", total_w, total_h)

        # Pre-allocate blank tile used when a camera has no observation yet
        self._blank_tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _marker_relative_to_ref(self, ref_rvec, ref_tvec, marker_rvec, marker_tvec):
        """Compute the pose of marker_i expressed *relative to* the reference
        marker (ID 0).  Returns rotation (matrix + quaternion) and translation."""
        R0, _ = cv2.Rodrigues(ref_rvec)        # reference rotation in camera frame
        Ri, _ = cv2.Rodrigues(marker_rvec)     # marker-i rotation in camera frame
        t0    = ref_tvec.flatten()
        ti    = marker_tvec.flatten()
        # Express Ri and ti in the reference marker's local frame
        R_rel = R0.T @ Ri
        t_rel = R0.T @ (ti - t0)
        q_rel = rotation_matrix_to_quaternion(R_rel)
        return {"R_rel": R_rel, "t_rel_m": t_rel, "q_rel": q_rel}

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _fuse_observations(self, observations):
        """For each plane marker (IDs 1-3), pick the best measurement across
        all cameras (highest combined quality score).

        Returns:
          best_by_marker – dict {marker_id: best_entry}
          debug_lines    – list of strings for display/logging
        """
        best_by_marker = {}
        debug_lines    = []

        for obs in observations:
            if obs is None:
                continue
            markers  = obs["markers"]
            cam_name = obs["camera_name"]

            # Reference marker must be visible to establish the local frame
            if REF_ID not in markers:
                debug_lines.append(f"{cam_name}: no mk0")
                continue

            visible_plane = [pid for pid in PLANE_IDS if pid in markers]
            if not visible_plane:
                debug_lines.append(f"{cam_name}: mk0 only")
                continue

            debug_lines.append(f"{cam_name}:{visible_plane}")
            ref = markers[REF_ID]

            for pid in visible_plane:
                m   = markers[pid]
                rel = self._marker_relative_to_ref(
                    ref["rvec"], ref["tvec"], m["rvec"], m["tvec"])
                # Blended score: plane marker weighted 60%, reference 40%
                combined_score = 0.6 * m["score"] + 0.4 * ref["score"]
                entry = {
                    "pid":         pid,
                    "camera_name": cam_name,
                    "score":       combined_score,
                    "q_rel":       rel["q_rel"],
                    "R_rel":       rel["R_rel"],
                    "t_rel_m":     rel["t_rel_m"],
                }
                # Keep only the best-scoring observation for each marker ID
                if pid not in best_by_marker or combined_score > best_by_marker[pid]["score"]:
                    best_by_marker[pid] = entry

        return best_by_marker, debug_lines

    def _compute_pose_from_fused_markers(self, best_by_marker):
        """Average orientations (quaternion mean) and positions across the 3
        plane markers to produce a single 6-DOF pose.

        Returns (pos_mm, euler_deg, used_cameras) or (None, None, None)."""
        if not all(pid in best_by_marker for pid in PLANE_IDS):
            return None, None, None   # incomplete rig – cannot compute pose

        # Orientation: quaternion average of all 3 plane-marker readings
        quats = [best_by_marker[pid]["q_rel"] for pid in PLANE_IDS]
        q_avg = avg_quaternions(quats)
        R_avg = quat_to_rot(q_avg)

        # Position: arithmetic mean of all 3 plane-marker translations
        t_avg  = np.mean([best_by_marker[pid]["t_rel_m"] for pid in PLANE_IDS], axis=0)
        pos_mm = t_avg * 1000.0            # convert metres → millimetres
        euler  = euler_from_mat(R_avg)
        used   = {pid: best_by_marker[pid]["camera_name"] for pid in PLANE_IDS}
        return pos_mm, euler, used

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _clamp_step(self, new_pose, prev_pose):
        """Reject implausibly large frame-to-frame jumps.
        The allowed step grows with pose_confidence so the filter 'opens up'
        progressively after the rig comes into view."""
        clamped = new_pose.copy()
        delta   = new_pose - prev_pose
        # Loosen limits as confidence builds (avoids tracking lag at start)
        scale     = 1.0 + (self.pose_confidence / self.max_confidence) * 3.0
        pos_limit = self.MAX_POS_STEP * scale
        ang_limit = self.MAX_ANG_STEP * scale
        # Clamp position components (indices 0-2)
        for i in range(3):
            if abs(delta[i]) > pos_limit:
                clamped[i] = prev_pose[i] + np.sign(delta[i]) * pos_limit
        # Clamp rotation components (indices 3-5)
        for i in range(3, 6):
            if abs(delta[i]) > ang_limit:
                clamped[i] = prev_pose[i] + np.sign(delta[i]) * ang_limit
        return clamped

    def _apply_deadzone(self, pose):
        """Zero out sub-threshold motion to suppress jitter at rest.
        Thresholds: 0.4 mm for position, 0.25° for rotation."""
        out = pose.copy()
        out[:3][np.abs(out[:3]) < 0.4]  = 0.0   # mm deadzone
        out[3:][np.abs(out[3:]) < 0.25] = 0.0   # degree deadzone
        return out

    # ------------------------------------------------------------------
    # IMU helpers
    # ------------------------------------------------------------------

    def _calibrate_imu_to_camera(self):
        """Store the inverse of the current IMU quaternion so that subsequent
        IMU readings report rotation *relative to this calibration pose*.
        Called once each time the camera successfully delivers a fused pose."""
        if not self.imu.available:
            return
        imu_q                  = self.imu.get_quaternion()
        self.imu_origin_q_inv  = qconj(imu_q)
        self.imu_calibrated    = True

    def _imu_relative_euler(self):
        """Return Euler angles of the IMU relative to the calibration pose,
        smoothed by a short rolling average to suppress sensor noise."""
        q  = self.imu.get_quaternion()
        qr = qmul(self.imu_origin_q_inv, q)   # relative quaternion
        if qr[0] < 0:
            qr = -qr                           # canonical hemisphere
        e  = q2euler(qr)
        self._imu_euler_hist.append(e)
        if len(self._imu_euler_hist) >= 3:
            return np.mean(self._imu_euler_hist, axis=0)
        return e

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _send_udp(self, pose):
        """Encode a 6-element pose as a comma-separated string and fire it off
        as a UDP datagram.  Non-blocking; dropped packets are silently ignored."""
        try:
            msg = (
                f"{float(pose[0]):.2f},"
                f"{float(pose[1]):.2f},"
                f"{float(pose[2]):.2f},"
                f"{float(pose[3]):.2f},"
                f"{float(pose[4]):.2f},"
                f"{float(pose[5]):.2f}"
            )
            self.sock.send(msg.encode())
        except Exception:
            pass

    def _build_combined_frame(self, observations, status_lines, mode_color):
        """Stitch per-camera display tiles side-by-side and append a status bar.

        Layout:
          [cam1 tile] [cam2 tile] [cam3 tile]   ← TILE_H rows
          [======== status bar ================]  ← 80 rows
        """
        # Collect tile from each observation (or blank if not yet available)
        tiles = []
        for obs in observations:
            if obs is not None and obs.get("display") is not None:
                tiles.append(obs["display"])
            else:
                tiles.append(self._blank_tile.copy())

        row = np.hstack(tiles)   # horizontally concatenate all tiles

        # Dark status bar below the tiles
        bar       = np.zeros((80, row.shape[1], 3), dtype=np.uint8)
        bar[:]    = (30, 30, 30)
        mid_x     = bar.shape[1] // 2

        # Left column: mode and position (colour-coded by mode)
        left_lines  = status_lines[:2]
        right_lines = status_lines[2:]
        for j, line in enumerate(left_lines):
            cv2.putText(bar, line, (10, 22 + j * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1, cv2.LINE_AA)
        # Right column: rotation and source camera info
        for j, line in enumerate(right_lines):
            cv2.putText(bar, line, (mid_x + 10, 22 + j * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # Top-right corner: main loop FPS + per-camera detection rate
        det_hz  = " | ".join(f"{cam.name[:4]}:{cam.det_fps:.0f}hz" for cam in self.cameras)
        fps_txt = f"main:{self.fps:.0f}fps  {det_hz}"
        (tw, _), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(bar, fps_txt, (bar.shape[1] - tw - 10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 220, 100), 1, cv2.LINE_AA)

        return np.vstack([row, bar])

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Main processing loop.  Runs on the calling thread; never blocks on
        camera I/O because all capture and detection happens in background threads."""
        try:
            while True:
                self.loop_counter += 1
                # Compute main-loop FPS every 30 iterations
                if self.loop_counter % 30 == 0:
                    now      = time.perf_counter()
                    self.fps = 30.0 / max(now - self.last_fps_time, 1e-6)
                    self.last_fps_time = now

                # ---- Collect latest pre-processed observations (non-blocking) ----
                observations = [cam.get_observation() for cam in self.cameras]

                # ---- Fuse across cameras → one best reading per plane marker ----
                best_by_marker, debug_lines = self._fuse_observations(observations)

                # ---- Compute combined 6-DOF pose ----
                pos, euler, used = self._compute_pose_from_fused_markers(best_by_marker)

                now_t      = time.perf_counter()
                mode_color = (0, 255, 0)

                if pos is not None:
                    # ======================================================
                    # CAMERA FUSED MODE – all three plane markers visible
                    # ======================================================
                    self.camera_lost_time = None    # reset occlusion timer
                    self.imu_frozen_xyz   = None

                    # Stack into a 6-vector [x, y, z, roll, pitch, yaw]
                    raw = np.array([pos[0], pos[1], pos[2],
                                    euler[0], euler[1], euler[2]], dtype=np.float32)
                    self.raw_history.append(raw)

                    # Rolling median over the last N frames removes outliers
                    median_pose = (np.median(self.raw_history, axis=0)
                                   if len(self.raw_history) >= 3 else raw)

                    # Clamp frame-to-frame jumps
                    if self.filtered_pose is not None:
                        median_pose = self._clamp_step(median_pose, self.filtered_pose)

                    # Exponential moving average for smooth output
                    if self.filtered_pose is None:
                        self.filtered_pose = median_pose.copy()
                    else:
                        self.filtered_pose = (self.alpha * median_pose
                                              + (1.0 - self.alpha) * self.filtered_pose)

                    output_pose = self._apply_deadzone(self.filtered_pose)
                    self.last_valid_pose = output_pose.copy()
                    self.pose_confidence = min(self.pose_confidence + 1, self.max_confidence)
                    self._calibrate_imu_to_camera()   # keep IMU origin aligned with cameras
                    self._send_udp(output_pose)

                    used_str = f"1:{used[1]} 2:{used[2]} 3:{used[3]}"
                    print(
                        f"\r[CAM-FUSED] X:{output_pose[0]:7.1f} Y:{output_pose[1]:7.1f} "
                        f"Z:{output_pose[2]:7.1f} | "
                        f"R:{output_pose[3]:6.1f} P:{output_pose[4]:6.1f} Y:{output_pose[5]:6.1f} | "
                        f"{used_str} | {self.fps:.1f}fps   ",
                        end="", flush=True
                    )
                    status = [
                        "MODE: CAMERA FUSED",
                        f"X:{output_pose[0]:.1f} Y:{output_pose[1]:.1f} Z:{output_pose[2]:.1f}",
                        f"R:{output_pose[3]:.1f} P:{output_pose[4]:.1f} Y:{output_pose[5]:.1f}",
                        f"SRC {used_str}",
                    ]
                    mode_color = (0, 255, 0)   # green

                else:
                    # ======================================================
                    # FALLBACK MODES – camera tracking lost
                    # ======================================================
                    missing = [pid for pid in PLANE_IDS if pid not in best_by_marker]

                    if self.camera_lost_time is None:
                        self.camera_lost_time = now_t   # record when we first lost tracking

                    lost_for = now_t - self.camera_lost_time
                    self.pose_confidence = max(0, self.pose_confidence - 1)

                    if (self.imu.available and self.imu_calibrated
                            and self.last_valid_pose is not None):
                        # ---- IMU fallback (rotation from IMU, position held) ----
                        imu_rpy = self._imu_relative_euler()

                        if lost_for <= IMU_TAKEOVER_FULL_SECS:
                            # Phase 1: hold last camera XYZ, use live IMU angles
                            output_pose = np.array([
                                self.last_valid_pose[0], self.last_valid_pose[1],
                                self.last_valid_pose[2],
                                imu_rpy[0], imu_rpy[1], imu_rpy[2],
                            ], dtype=np.float32)
                            mode = "IMU BACKUP"
                        else:
                            # Phase 2: freeze XYZ at the moment of phase transition
                            if self.imu_frozen_xyz is None:
                                self.imu_frozen_xyz = self.last_valid_pose[:3].copy()
                            output_pose = np.array([
                                self.imu_frozen_xyz[0], self.imu_frozen_xyz[1],
                                self.imu_frozen_xyz[2],
                                imu_rpy[0], imu_rpy[1], imu_rpy[2],
                            ], dtype=np.float32)
                            mode = "IMU ROT-ONLY"

                        self._send_udp(output_pose)
                        cal = self.imu.get_cal()
                        print(
                            f"\r[IMU] X:{output_pose[0]:7.1f} Y:{output_pose[1]:7.1f} "
                            f"Z:{output_pose[2]:7.1f} | "
                            f"R:{output_pose[3]:6.1f} P:{output_pose[4]:6.1f} Y:{output_pose[5]:6.1f} | "
                            f"{mode} missing:{missing} lost:{lost_for:.1f}s cal:{cal}   ",
                            end="", flush=True
                        )
                        status = [
                            f"MODE: {mode}",
                            f"MISSING: {missing}  LOST: {lost_for:.1f}s",
                            f"X:{output_pose[0]:.1f} Y:{output_pose[1]:.1f} Z:{output_pose[2]:.1f}",
                            f"R:{output_pose[3]:.1f} P:{output_pose[4]:.1f} Y:{output_pose[5]:.1f}",
                        ]
                        mode_color = (0, 180, 255)   # orange

                    elif self.last_valid_pose is not None:
                        # ---- Hold last pose (no IMU available) ----
                        output_pose = self.last_valid_pose.copy()
                        self._send_udp(output_pose)
                        print(
                            f"\r[HOLD] X:{output_pose[0]:7.1f} Y:{output_pose[1]:7.1f} "
                            f"Z:{output_pose[2]:7.1f} | "
                            f"missing:{missing} holding last pose   ",
                            end="", flush=True
                        )
                        status = [
                            "MODE: HOLD LAST POSE",
                            f"MISSING: {missing}",
                            f"X:{output_pose[0]:.1f} Y:{output_pose[1]:.1f} Z:{output_pose[2]:.1f}",
                            f"DBG: {' | '.join(debug_lines[:2])}",
                        ]
                        mode_color = (0, 140, 255)   # amber

                    else:
                        # ---- Waiting – no pose ever established ----
                        output_pose = None
                        print(
                            f"\r[WAITING] No valid fused pose. Missing: {missing} | "
                            f"{' | '.join(debug_lines)}   ",
                            end="", flush=True
                        )
                        status = [
                            "MODE: WAITING FOR FIRST POSE",
                            f"MISSING: {missing}",
                            f"DBG: {' | '.join(debug_lines[:2])}",
                            "",
                        ]
                        mode_color = (0, 0, 255)   # red

                # ---- Render combined display ----
                combined = self._build_combined_frame(observations, status, mode_color)
                cv2.imshow("Tracker", combined)

                # 1 ms wait keeps the GUI responsive; 'q' exits cleanly
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        finally:
            self.shutdown()

    def shutdown(self):
        """Graceful teardown: stop camera threads, close IMU, close sockets."""
        for cam in self.cameras:
            cam.stop()
        self.imu.stop()
        self.sock.close()
        cv2.destroyAllWindows()
        print("\nDone.")


# ===========================================================================
if __name__ == "__main__":
    tracker = MultiCameraCombinedTracker()
    tracker.run()