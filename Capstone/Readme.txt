# Multi-Camera 6-DOF Tracker — Setup Guide

## File structure
```
tracker/
├── main.py            ← run this
├── camera_worker.py   ← one instance per camera
├── pose_fusion.py     ← merges detections across cameras
├── smoother.py        ← all temporal filtering
├── udp_sender.py      ← sends pose over UDP
├── calib_cam0.json    ← (optional) calibration files, one per camera
├── calib_cam1.json
└── calib_cam2.json
```

---

## 1  Hardware — where to plug the cameras

The Raspberry Pi 5 has:
- 2 × USB 3.0 (blue) ports  
- 2 × USB 2.0 (black) ports

```
USB 3.0  ──►  EMEET S600 #0  (fastest, use for the most important view)
USB 3.0  ──►  EMEET S600 #1
USB 2.0  ──►  EMEET S600 #2

Ethernet ──►  your laptop (direct cable, no switch needed)
```

If you get USB bandwidth warnings (rare with 3 UVC cams at 640×360), plug
cam2 into a powered USB 3.0 hub on the USB 2.0 port, or drop its resolution
to 320×240 in camera_worker.py.

---

## 2  Static IP configuration (Pi ↔ Laptop direct ethernet)

### On the Pi (`/etc/dhcpcd.conf`, add at the bottom):
```
interface eth0
static ip_address=10.0.0.1/24
static routers=10.0.0.2
```
Then: `sudo systemctl restart dhcpcd`

### On the laptop (Windows):
Network Settings → Ethernet adapter → IPv4 → Manual  
IP: `10.0.0.2`   Subnet: `255.255.255.0`   Gateway: `10.0.0.1`

### On the laptop (Ubuntu/Debian):
```bash
sudo ip addr add 10.0.0.2/24 dev eth0
sudo ip link set eth0 up
```

Verify connectivity: `ping 10.0.0.1` from the laptop.

---

## 3  Install dependencies on the Pi

```bash
sudo apt update
sudo apt install -y python3-opencv python3-pip
pip3 install numpy --break-system-packages
# opencv-contrib is needed for ArUco
pip3 install opencv-contrib-python --break-system-packages
```

---

## 4  Find the correct /dev/video* indices

```bash
v4l2-ctl --list-devices
```

Each UVC webcam on Linux usually creates **two** device nodes (e.g. video0 + video1).
Only the even-numbered one (video0, video2, video4) is the actual video stream.

Edit `CAMERAS` list in `main.py` to match your actual indices.

---

## 5  Camera calibration (recommended, not required)

Run a standard OpenCV calibration script with a chessboard for each camera.
Save results as `calib_cam0.json`, `calib_cam1.json`, `calib_cam2.json` with:
```json
{
  "camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]],
  "distortion_coefficients": [k1,k2,p1,p2,k3],
  "reprojection_error": 0.3
}
```
If any file is missing the system falls back to a reasonable default.

---

## 6  Run

```bash
cd tracker
python3 main.py
```

To run headless over SSH (no display):
```python
# in main.py, set:
SHOW_PREVIEW = False
```

---

## 7  Fusion logic explained

| Scenario | Result |
|---|---|
| Cam0 sees {0,1,2,3} | Full pose from cam0 only |
| Cam0 sees {0,1}, Cam1 sees {0,2,3} | Two independent estimates → weighted average (cam1 gets 2× weight) |
| Cam0 sees {0}, Cam1 sees {1,2,3} | Cam0 has ref but no plane markers, Cam1 has plane but no ref → **cannot fuse**, hold last pose |
| Cam0 sees {0,1}, Cam1 sees {1,2,3} | Cam0 gives partial estimate (ref + 1 plane marker, weight=1), valid partial pose |
| No camera sees marker 0 | Hold last valid pose up to HOLD_TIME_SEC |

The key rule: **at least one camera must see marker 0 (the reference) AND at
least one plane marker (1, 2, or 3) simultaneously** for a new pose to be
computed. Plan your camera placement accordingly — ensure at least one camera
reliably sees marker 0.

---

## 8  Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot open camera index N` | Wrong index — run `v4l2-ctl --list-devices` |
| All cameras show 0 fps | USB bandwidth — drop resolution or use powered hub |
| Jittery pose | Increase `MEDIAN_WIN` or reduce `alpha` in smoother.py |
| Pose lags real motion | Increase `alpha` in smoother.py (closer to 1.0) |
| UDP packets not arriving | Check firewall on laptop, verify IPs with `ping` |