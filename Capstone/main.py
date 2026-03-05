"""
main.py
-------
3-camera 6-DOF ArUco tracker + BNO055 IMU backup.

Cameras:
  cam1 → /dev/video0  (USB 3.0 port 1)
  cam2 → /dev/video2  (USB 3.0 port 2)
  cam3 → /dev/video4  (USB 2.0 port 1)

IMU:
  BNO055 → /dev/ttyUSB0 or /dev/ttyACM0 (USB 2.0 port 2, auto-detected)
  Press Z to zero IMU at marker-0 position before tracking.

Fusion:
  Markers visible   → cameras drive all 6 values (X/Y/Z + Roll/Pitch/Yaw)
  Markers lost <5s  → position FROZEN at last camera value
                       rotation taken over by IMU (live, no drift for ~5s)
  Markers lost >5s  → last known pose continues to be reported via UDP
                       (never stops sending — receiver always gets something)
  IMU not zeroed    → system still works, IMU rotation relative to power-on
  IMU not connected → camera rotation used throughout, held on marker loss

UDP:  "X,Y,Z,Roll,Pitch,Yaw"  sent to TARGET_IP:TARGET_PORT
      Also sent over WiFi if the Pi has a WiFi interface — same packet.

Keyboard shortcuts:
  Q  quit
  A  toggle 3-D axis overlays on camera previews
  H  print marker affinity table
  Z  zero IMU to current orientation (place IMU at marker-0 first)
  R  reset IMU zero (back to power-on reference)
"""

import cv2
import numpy as np
import time
import sys
import os

from camera_worker import CameraWorker
from pose_fusion   import PoseFusion
from smoother      import PoseSmoother
from udp_sender    import UDPSender

try:
    from imu_worker import IMUWorker
    _IMU_AVAILABLE = True
except ImportError:
    _IMU_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# UDP targets — sends to BOTH (Ethernet + WiFi if available)
UDP_ETH_IP    = "10.0.0.2"        # laptop on direct Ethernet
UDP_WIFI_IP   = "10.0.0.2"        # change to WiFi IP if different subnet
UDP_PORT      = 5005

# Cameras — names 1/2/3, indices 0/2/4
CAMERAS = [
    {"index": 0, "id": "cam1", "calibration": None},   # USB 3.0
    {"index": 2, "id": "cam2", "calibration": None},   # USB 3.0
    {"index": 4, "id": "cam3", "calibration": None},   # USB 2.0
]
SHARED_CALIBRATION = "camera_calibration.json"

# IMU
IMU_PORT     = None      # None = auto-detect /dev/ttyUSB* or /dev/ttyACM*
IMU_BAUDRATE = 115200
USE_IMU      = True

# How long to freeze position before giving up and just holding last value
FREEZE_TIME_SEC = 5.0

# Camera confidence threshold to trigger IMU yaw correction (0-20)
YAW_CORRECT_CONFIDENCE = 15

SHOW_PREVIEW    = True
AFFINITY_WINDOW = 300

# ═══════════════════════════════════════════════════════════════════════════════


class AffinityTracker:
    def __init__(self, n_cams, window=AFFINITY_WINDOW):
        self._win    = window
        self._counts = [{} for _ in range(n_cams)]
        self._frame  = 0

    def update(self, results):
        self._frame += 1
        decay = (self._frame % self._win == 0)
        for i, res in enumerate(results):
            if decay:
                self._counts[i] = {}
            for mid in res.get('rvecs', {}).keys():
                self._counts[i][mid] = self._counts[i].get(mid, 0) + 1

    def coverage_strings(self, cam_ids):
        denom = max(self._frame % self._win or self._win, 1)
        out = []
        for i, cid in enumerate(cam_ids):
            parts = [f"m{mid}({int(100*c/denom)}%)"
                     for mid, c in sorted(self._counts[i].items())]
            out.append(f"{cid}: " + (" ".join(parts) if parts else "—"))
        return out

    def print_affinity(self, cam_ids):
        print("\n── Marker Affinity ──")
        for line in self.coverage_strings(cam_ids):
            print(" ", line)
        print()


def resolve_calibration(cfg):
    per = cfg.get("calibration")
    if per and os.path.exists(per):
        return per
    if SHARED_CALIBRATION and os.path.exists(SHARED_CALIBRATION):
        return SHARED_CALIBRATION
    return None


def build_hud(cam_results, cam_ids, fused_pose, rot_src,
              imu_result, smoother, fps_val, affinity,
              markers_visible, frozen, imu_zeroed):

    COL_OK   = (0, 255,   0)
    COL_WARN = (0, 210, 210)
    COL_DEAD = (0,   0, 255)
    COL_GREY = (80,  80,  80)
    COL_CYAN = (255, 200,  0)
    COL_ORG  = (0,  165, 255)   # orange — frozen position

    # ── Camera panels (side by side, 426×240 each) ────────────────────────────
    panels = []
    for res in cam_results:
        f = res.get('frame')
        if f is not None:
            f = cv2.resize(f, (426, 240))
        else:
            f = np.zeros((240, 426, 3), dtype=np.uint8)
            cv2.putText(f, "DEAD", (150, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COL_DEAD, 2)
        panels.append(f)
    cam_strip = np.hstack(panels)          # 1278 × 240

    # ── Status panel ─────────────────────────────────────────────────────────
    pw, ph = 480, 240
    panel  = np.zeros((ph, pw, 3), dtype=np.uint8)

    def put(txt, y, col=(200,200,200), sc=0.46):
        cv2.putText(panel, txt, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)

    # Title
    cc = COL_OK if smoother.confidence > 10 else COL_WARN
    put(f"6-DOF | {fps_val:.0f}fps | conf:{smoother.confidence}/20", 20, cc, 0.50)

    # Source banner
    if markers_visible:
        put("LIVE — cameras tracking", 38, COL_OK, 0.42)
    elif frozen and imu_result and imu_result.get('alive'):
        put("HOLD — IMU rotation, position frozen", 38, COL_CYAN, 0.42)
    elif frozen:
        put("HOLD — last known pose (no IMU)", 38, COL_ORG, 0.42)
    else:
        put("STALE — reporting last known pose", 38, COL_WARN, 0.42)

    cv2.line(panel, (0, 46), (pw, 46), (60,60,60), 1)

    # Pose values
    y = 64
    if fused_pose is not None:
        fp      = fused_pose
        pos_col = COL_OK if markers_visible else COL_ORG
        rot_col = COL_OK if rot_src == 'camera' else COL_CYAN
        for lbl, v, col in [
            ("X   mm", fp[0], pos_col), ("Y   mm", fp[1], pos_col),
            ("Z   mm", fp[2], pos_col), ("Roll ° ", fp[3], rot_col),
            ("Pitch°", fp[4], rot_col), ("Yaw  °", fp[5], rot_col),
        ]:
            put(f"{lbl} : {v:+9.2f}", y, col)
            y += 22
    else:
        put("NO POSE — waiting for markers", y, COL_DEAD, 0.50)
        y += 26
        put("Need marker 0 + any plane marker", y, COL_GREY, 0.38)
        y += 18
        put("in at least one camera view", y, COL_GREY, 0.38)
        y += 18

    cv2.line(panel, (0, y+4), (pw, y+4), (60,60,60), 1)
    y += 14

    # IMU row
    if imu_result and imu_result.get('alive'):
        zstr = "zeroed" if imu_zeroed else "NOT zeroed — press Z"
        ir   = imu_result
        put(f"IMU {ir['fps']:.0f}Hz {zstr} | "
            f"R:{ir['roll']:+6.1f} P:{ir['pitch']:+6.1f} Y:{ir['corrected_yaw']:+6.1f}",
            y, COL_CYAN, 0.38)
        y += 16
        cal = ir.get('cal', (0,0,0,0))
        put(f"  Cal sys:{cal[0]} gyro:{cal[1]} accel:{cal[2]} mag:{cal[3]}",
            y, COL_GREY, 0.36)
    else:
        put("IMU: not connected", y, COL_GREY, 0.40)
        y += 16
        put("  rotation from cameras only", y, COL_GREY, 0.36)

    cv2.line(panel, (0, ph-54), (pw, ph-54), (60,60,60), 1)

    # Camera status
    y = ph - 42
    for res, cid in zip(cam_results, cam_ids):
        ids_seen = sorted(res.get('rvecs', {}).keys())
        cfps     = res.get('fps', 0.0)
        ts       = res.get('timestamp', 0.0)
        stale    = (time.perf_counter() - ts) > 3.0 if ts else True
        if stale:
            txt, col = f"{cid} DEAD", COL_DEAD
        elif ids_seen:
            txt, col = f"{cid} {ids_seen} {cfps:.0f}fps", COL_OK
        else:
            txt, col = f"{cid} (none) {cfps:.0f}fps", COL_WARN
        put(txt, y, col, 0.36)
        y += 14

    # Controls
    cv2.line(panel, (0, ph-6), (pw, ph-6), (60,60,60), 1)
    put("Q=quit A=axes H=affinity Z=zero-IMU R=reset-IMU", ph-1,
        (100,100,100), 0.32)

    return np.hstack([cam_strip, panel])


def main():
    print("=" * 65)
    print("  6-DOF Tracker  |  cam1+cam2 (USB3)  cam3+IMU (USB2)")
    print("=" * 65)
    print()
    print("  cam1 → /dev/video0  (USB 3.0)")
    print("  cam2 → /dev/video2  (USB 3.0)")
    print("  cam3 → /dev/video4  (USB 2.0)")
    print("  IMU  → auto-detect  (USB 2.0)")
    print()

    # ── Cameras ───────────────────────────────────────────────────────────────
    workers = []
    cam_ids = []
    for cfg in CAMERAS:
        try:
            w = CameraWorker(
                cam_index        = cfg["index"],
                camera_id        = cfg["id"],
                calibration_file = resolve_calibration(cfg),
            )
            workers.append(w)
            cam_ids.append(cfg["id"])
        except RuntimeError as e:
            print(f"  WARNING: {e} — skipping")

    if not workers:
        print("ERROR: No cameras opened. Check USB connections.")
        sys.exit(1)

    # ── IMU ───────────────────────────────────────────────────────────────────
    imu = None
    if USE_IMU and _IMU_AVAILABLE:
        try:
            imu = IMUWorker(port=IMU_PORT, baudrate=IMU_BAUDRATE)
            print("  IMU ready — press Z after placing IMU at marker-0 to zero it")
        except Exception as e:
            print(f"  WARNING: IMU failed: {e}")
            print("  Continuing camera-only.")
    elif USE_IMU and not _IMU_AVAILABLE:
        print("  WARNING: pyserial missing.")
        print("  Run: pip3 install pyserial --break-system-packages")

    # ── Modules ───────────────────────────────────────────────────────────────
    fusion   = PoseFusion()
    smoother = PoseSmoother()
    udp      = UDPSender(UDP_ETH_IP, UDP_PORT)

    fusion   = PoseFusion()
    smoother = PoseSmoother()
    affinity = AffinityTracker(n_cams=len(workers))

    # ── Tracking state ────────────────────────────────────────────────────────
    fps_count = 0
    fps_time  = time.perf_counter()
    fps_val   = 0.0

    last_valid_pose  = None    # full 6-element array, last camera-good pose
    last_cam_time    = 0.0     # time of last successful camera pose
    output_pose      = None
    rot_src          = 'camera'
    markers_visible  = False
    is_frozen        = False

    print(f"\n  Cameras active: {len(workers)}")
    print(f"  IMU: {'connected' if imu else 'not connected'}")
    print(f"  UDP → {UDP_ETH_IP}:{UDP_PORT}")
    print(f"  SHOW_PREVIEW = {SHOW_PREVIEW}")
    print("  Running — Q to quit\n")

    while True:

        # ── Camera detections ─────────────────────────────────────────────────
        cam_results = [w.get_result() for w in workers]
        affinity.update(cam_results)
        raw_pose, n_seen = fusion.fuse(cam_results)

        # ── IMU ───────────────────────────────────────────────────────────────
        imu_result = imu.get_result() if imu else None
        imu_alive  = bool(imu_result and imu_result.get('alive'))
        imu_zeroed = imu.is_zeroed() if imu else False

        # ── Fusion ───────────────────────────────────────────────────────────
        if raw_pose is not None:
            # Cameras are live — cameras own everything
            markers_visible = True
            is_frozen       = False
            smoothed        = smoother.update(raw_pose)
            last_valid_pose = smoothed.copy()
            last_cam_time   = time.perf_counter()

            # Correct IMU yaw drift while we have confident camera view
            if imu_alive and smoother.confidence >= YAW_CORRECT_CONFIDENCE:
                imu.correct_yaw_drift(smoothed[5])

            output_pose = smoothed.copy()
            rot_src     = 'camera'

        else:
            # Cameras lost markers
            markers_visible = False
            smoother.decay()
            age = time.perf_counter() - last_cam_time

            if last_valid_pose is not None:
                is_frozen = True

                if imu_alive and age < FREEZE_TIME_SEC:
                    # IMU takes rotation, cameras froze position
                    output_pose = np.array([
                        last_valid_pose[0],
                        last_valid_pose[1],
                        last_valid_pose[2],
                        imu_result['roll'],
                        imu_result['pitch'],
                        imu_result['corrected_yaw'],
                    ], dtype=np.float64)
                    rot_src = 'imu'
                else:
                    # No IMU or past freeze window — hold entire last pose
                    output_pose = last_valid_pose.copy()
                    rot_src     = 'camera'
            else:
                # Never had a valid pose yet
                is_frozen   = False
                output_pose = None

        # ── UDP send — always send when we have any pose ──────────────────────
        if output_pose is not None:
            udp.send(output_pose)

        # ── FPS ───────────────────────────────────────────────────────────────
        fps_count += 1
        if fps_count >= 60:
            now      = time.perf_counter()
            fps_val  = 60 / (now - fps_time)
            fps_time = now
            fps_count = 0

        # ── Console line ──────────────────────────────────────────────────────
        if output_pose is not None:
            fp  = output_pose
            tag = ("LIVE   " if markers_visible else
                   "IMU-ROT" if (rot_src == 'imu') else
                   "FROZEN ")
            print(
                f"\r  [{tag}] "
                f"X:{fp[0]:+7.1f} Y:{fp[1]:+7.1f} Z:{fp[2]:+7.1f}  "
                f"R:{fp[3]:+6.1f} P:{fp[4]:+6.1f} Yw:{fp[5]:+6.1f}  "
                f"{fps_val:.0f}fps   ",
                end='', flush=True
            )
        else:
            print(
                f"\r  [WAIT ] no pose yet — "
                f"markers:{n_seen}  imu:{'ok' if imu_alive else 'no'}   ",
                end='', flush=True
            )

        # ── Preview window ────────────────────────────────────────────────────
        if SHOW_PREVIEW:
            vis = build_hud(
                cam_results, cam_ids,
                output_pose, rot_src,
                imu_result, smoother, fps_val,
                affinity, markers_visible, is_frozen, imu_zeroed
            )
            cv2.imshow("6-DOF Tracker", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('a'):
                for w in workers:
                    w.set_draw_axes(not w.get_draw_axes())
            elif key == ord('h'):
                affinity.print_affinity(cam_ids)
            elif key == ord('z') and imu:
                imu.set_zero()
            elif key == ord('r') and imu:
                imu.clear_zero()
                print("\n  [R] IMU zero cleared")
        else:
            time.sleep(0.001)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("\nShutting down...")
    for w in workers:
        w.stop()
    if imu:
        imu.stop()
    udp.close()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()
    affinity.print_affinity(cam_ids)
    print("Done.")


if __name__ == "__main__":
    main()