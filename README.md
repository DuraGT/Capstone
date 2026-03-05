# 6-DOF ArUco Tracker — User Manual

**System:** Raspberry Pi 5 · 3× EMEET S600 USB cameras · BNO055 IMU · Direct Ethernet to laptop

This manual takes you from unpacked hardware to a running 6-DOF (six degrees of freedom) tracking system. No prior experience is assumed. Read each section in order.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Hardware You Need](#2-hardware-you-need)
3. [How It Works — Overview](#3-how-it-works--overview)
4. [Printing the ArUco Markers](#4-printing-the-aruco-markers)
5. [Hardware Setup](#5-hardware-setup)
   - 5.1 [USB Port Layout](#51-usb-port-layout)
   - 5.2 [Wiring the BNO055 IMU](#52-wiring-the-bno055-imu)
   - 5.3 [Ethernet Connection](#53-ethernet-connection)
6. [Raspberry Pi Software Setup](#6-raspberry-pi-software-setup)
7. [Laptop Software Setup](#7-laptop-software-setup)
8. [Network Configuration](#8-network-configuration)
9. [Configuring the Tracker](#9-configuring-the-tracker)
10. [Running the Tracker](#10-running-the-tracker)
11. [Zeroing the IMU](#11-zeroing-the-imu)
12. [Understanding the Display](#12-understanding-the-display)
13. [Keyboard Controls](#13-keyboard-controls)
14. [Understanding the UDP Output](#14-understanding-the-udp-output)
15. [Auto-Start on Boot](#15-auto-start-on-boot)
16. [Troubleshooting](#16-troubleshooting)
17. [System Parameters Reference](#17-system-parameters-reference)
18. [File Reference](#18-file-reference)

---

## 1. What This System Does

This tracker outputs the **position and orientation** of an object in real time — six numbers updated ~30 times per second:

| Value | Unit | Meaning |
|-------|------|---------|
| X | millimetres | Left/right |
| Y | millimetres | Up/down |
| Z | millimetres | Forward/back |
| Roll | degrees | Tilt left/right |
| Pitch | degrees | Tilt forward/back |
| Yaw | degrees | Rotate left/right |

All values are measured **relative to ArUco marker 0**, which acts as the fixed origin of your coordinate system.

The data is sent over UDP as a plain text string: `X,Y,Z,Roll,Pitch,Yaw` — receivable by any application on your laptop.

**What happens when cameras lose sight of the markers:**
The last known position is held (frozen). The IMU takes over orientation so rotation continues updating live for up to 5 seconds. After that, the last known full pose continues to be transmitted indefinitely until markers are seen again.

---

## 2. Hardware You Need

| Item | Quantity | Notes |
|------|----------|-------|
| Raspberry Pi 5 | 1 | Any RAM size (min 4 GB recommended) |
| EMEET SmartCam S600 | 3 | USB webcams |
| BNO055 IMU module | 1 | 9-axis, with USB-UART adapter (CP2102 or CH340) |
| USB-UART adapter | 1 | CP2102 or CH340 — needed to connect BNO055 to Pi |
| MicroSD card | 1 | 16 GB minimum, Class 10 |
| USB-C power supply | 1 | 5V 5A for Pi 5 |
| Ethernet cable | 1 | Direct Pi ↔ Laptop |
| Laptop | 1 | Windows, Mac, or Linux |
| A4 paper or card | 4 sheets | For printing markers |

**Optional but recommended:**
- Small monitor + HDMI cable (for first-time setup on Pi)
- USB keyboard + mouse (for Pi setup)

---

## 3. How It Works — Overview

```
┌─────────────────────────────────────────────────────┐
│                  Raspberry Pi 5                      │
│                                                      │
│  cam1 ──► detect markers ──►┐                       │
│  cam2 ──► detect markers ──►├──► fuse poses ──►     │
│  cam3 ──► detect markers ──►┘    smoother   ──► UDP │
│                                      ▲              │
│  IMU ────────────────────────────────┘              │
│         (backup rotation when markers lost)         │
└─────────────────────────────────────────────────────┘
                                          │
                                     Ethernet cable
                                          │
                               ┌──────────────────┐
                               │     Laptop        │
                               │  receives UDP     │
                               │  on port 5005     │
                               └──────────────────┘
```

**ArUco markers** are printed black-and-white square patterns. The cameras detect them and calculate their exact position in 3D space:
- **Marker 0** is the fixed reference point (origin). It stays stationary.
- **Markers 1, 2, 3** are attached to the object being tracked.

No single camera needs to see all markers. If cam1 sees markers 0 and 1 while cam2 sees markers 0 and 2, the system combines both results into one accurate pose.

---

## 4. Printing the ArUco Markers

The system uses the **4×4 ArUco dictionary**, markers with IDs **0, 1, 2, and 3**.

**Step 1 — Generate and print markers**

Run this on any computer with Python and OpenCV:

```python
import cv2
import cv2.aruco as aruco

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

for marker_id in range(4):
    img = aruco.generateImageMarker(dictionary, marker_id, 300)
    cv2.imwrite(f"marker_{marker_id}.png", img)
    print(f"Saved marker_{marker_id}.png")
```

**Step 2 — Print at the correct size**

Print each marker so the **black square border measures exactly 15 mm on each side**.

- Use "actual size" or "100%" in your print dialog — do not scale to fit page
- Print on plain white paper or card
- After printing, measure with a ruler. If the black square is not 15 mm, adjust the print size accordingly and reprint

> **Important:** If the printed size does not match 15 mm exactly, edit `marker_size = 0.015` in `camera_worker.py` to match your actual printed size in metres. For example, 20 mm would be `marker_size = 0.020`.

**Step 3 — Mount the markers**

- **Marker 0:** Stick flat to a rigid surface in a fixed location. This is your coordinate origin — it must not move during tracking.
- **Markers 1, 2, 3:** Attach to the object you are tracking, spread apart so multiple cameras can see at least one. They do not need to be parallel or at the same angle.

---

## 5. Hardware Setup

### 5.1 USB Port Layout

The Raspberry Pi 5 has **4 USB ports**:
- **2 × USB 3.0** (blue inside) — faster, higher bandwidth
- **2 × USB 2.0** (black inside) — standard speed

Connect in this exact order for best performance:

```
┌─────────────────────────────────────────┐
│         Raspberry Pi 5 (rear)            │
│                                          │
│  [USB 3.0] ──► cam1 (EMEET S600 #1)    │
│  [USB 3.0] ──► cam2 (EMEET S600 #2)    │
│  [USB 2.0] ──► cam3 (EMEET S600 #3)    │
│  [USB 2.0] ──► IMU USB-UART adapter     │
│  [Ethernet]──► Laptop                   │
│  [USB-C]   ──► Power supply (5V 5A)    │
└─────────────────────────────────────────┘
```

> **Why this layout?** cam1 and cam2 on USB 3.0 get the most bandwidth. cam3 on USB 2.0 is fine because there is one camera per USB 2.0 controller. The IMU sends only tiny serial data so it is perfectly suited to USB 2.0.

### 5.2 Wiring the BNO055 IMU

The BNO055 connects to the Pi via a **USB-UART adapter** (CP2102 or CH340 chip). The adapter plugs into USB; the BNO055 connects to the adapter's pins.

**Wiring (BNO055 → USB-UART adapter):**

| BNO055 Pin | → | Adapter Pin | Notes |
|------------|---|-------------|-------|
| VIN / 3V3 | → | 3.3V | Power — use 3.3V not 5V |
| GND | → | GND | Ground |
| TX | → | RX | BNO055 transmits to adapter receive |
| RX | → | TX | BNO055 receives from adapter transmit |
| PS1 | → | 3.3V | Selects UART mode |
| PS0 | → | GND | Selects RVC protocol |

> **Double-check:** TX goes to RX and RX goes to TX — this is a cross-connection. If you connect TX→TX the IMU will not send any data.

**After wiring, plug the USB-UART adapter into the Pi's USB 2.0 port.**

Verify the Pi detects it:
```bash
ls /dev/ttyUSB* /dev/ttyACM*
# Should show something like: /dev/ttyUSB0
```

**IMU placement:**
Place the IMU at the same position as marker 0 and aligned to the same orientation before zeroing. Once zeroed, all angles are reported relative to this starting position. The IMU can then be moved with the tracked object.

### 5.3 Ethernet Connection

Connect an Ethernet cable directly between the Pi's Ethernet port and your laptop's Ethernet port. No router or switch is needed — this is a direct peer-to-peer connection.

---

## 6. Raspberry Pi Software Setup

### Step 1 — Flash the OS

1. Download **Raspberry Pi Imager** from https://www.raspberrypi.com/software/
2. Insert MicroSD card into your computer
3. Open Imager → Choose OS → **Raspberry Pi OS (64-bit)** (Bookworm or Trixie)
4. Choose your MicroSD card
5. Click the settings gear icon and configure:
   - Hostname: `raspberrypi` (or any name you prefer)
   - Enable SSH
   - Set username: `pi` and a password
   - Configure WiFi if needed for initial downloads
6. Click Write and wait for it to finish
7. Insert MicroSD into Pi and power on

### Step 2 — Update the system

Connect a monitor and keyboard to the Pi, or SSH in:
```bash
ssh pi@raspberrypi.local
```

Then run:
```bash
sudo apt update && sudo apt upgrade -y
```

### Step 3 — Install Python dependencies

```bash
# OpenCV with ArUco support
pip3 install opencv-contrib-python numpy --break-system-packages

# Serial port library for IMU
pip3 install pyserial --break-system-packages
```

> **Note:** If you get a conflict with system opencv, remove it first:
> ```bash
> sudo apt remove python3-opencv -y
> ```
> Then re-run the pip install command above.

### Step 4 — Verify camera detection

```bash
v4l2-ctl --list-devices
```

You should see three entries like:
```
EMEET SmartCam S600: EMEET Smar (usb-xhci-hcd.0-1):
    /dev/video0
    /dev/video1

EMEET SmartCam S600: EMEET Smar (usb-xhci-hcd.1-1):
    /dev/video2
    /dev/video3

EMEET SmartCam S600: EMEET Smar (usb-xhci-hcd.1-2):
    /dev/video4
    /dev/video5
```

The tracker uses `/dev/video0`, `/dev/video2`, and `/dev/video4` (the first node of each camera). If your numbers are different, update the `index` values in `main.py` — see Section 9.

### Step 5 — Verify IMU detection

With the IMU plugged in:
```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

Should show `/dev/ttyUSB0` or `/dev/ttyACM0`. The tracker auto-detects whichever appears.

### Step 6 — Copy tracker files to Pi

From your laptop, copy the tracker folder to the Pi:
```bash
scp -r tracker/ pi@10.0.0.1:/home/pi/tracker
```

Or if copying from within the Pi, place all files in `/home/pi/tracker/`.

The folder should contain:
```
tracker/
├── main.py
├── camera_worker.py
├── imu_worker.py
├── pose_fusion.py
├── smoother.py
├── udp_sender.py
├── tracker.service
└── camera_calibration.json   ← (optional, see Section 9)
```

---

## 7. Laptop Software Setup

### Receiving UDP data

The tracker sends data to your laptop on **UDP port 5005**. Any language can receive it.

**Python example (receive and print):**
```python
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5005))
sock.settimeout(1.0)

print("Listening for tracker data on port 5005...")

while True:
    try:
        data, addr = sock.recvfrom(1024)
        values = data.decode().split(",")
        x, y, z       = float(values[0]), float(values[1]), float(values[2])
        roll, pitch, yaw = float(values[3]), float(values[4]), float(values[5])
        print(f"X:{x:+8.2f}mm  Y:{y:+8.2f}mm  Z:{z:+8.2f}mm  "
              f"Roll:{roll:+7.2f}°  Pitch:{pitch:+7.2f}°  Yaw:{yaw:+7.2f}°")
    except socket.timeout:
        print("(waiting...)")
```

Save as `receive.py` and run it before starting the tracker on the Pi.

---

## 8. Network Configuration

The Pi and laptop communicate over a **direct Ethernet cable** using static IP addresses. No internet or router is needed.

### On the Raspberry Pi

Edit the DHCP configuration:
```bash
sudo nano /etc/dhcpcd.conf
```

Add these lines at the bottom:
```
interface eth0
static ip_address=10.0.0.1/24
static routers=10.0.0.2
```

Save with `Ctrl+O`, `Enter`, `Ctrl+X`, then reboot:
```bash
sudo reboot
```

### On the Laptop (Windows)

1. Open **Control Panel → Network and Sharing Centre → Change adapter settings**
2. Right-click your **Ethernet adapter** → Properties
3. Select **Internet Protocol Version 4 (TCP/IPv4)** → Properties
4. Select **Use the following IP address:**
   - IP address: `10.0.0.2`
   - Subnet mask: `255.255.255.0`
   - Default gateway: `10.0.0.1`
5. Click OK

### On the Laptop (Mac)

1. System Settings → Network → Ethernet
2. Configure IPv4: **Manually**
3. IP address: `10.0.0.2`
4. Subnet Mask: `255.255.255.0`
5. Click Apply

### Verify the connection

After rebooting the Pi, test from your laptop:
```bash
ping 10.0.0.1
```

You should see replies. If not, check the cable and repeat the steps above.

SSH into the Pi:
```bash
ssh pi@10.0.0.1
```

---

## 9. Configuring the Tracker

All configuration is at the top of `main.py`. Open it with:
```bash
nano /home/pi/tracker/main.py
```

### Essential settings

```python
# ── Network ──────────────────────────────────────────────────────────────────
UDP_ETH_IP    = "10.0.0.2"    # Your laptop's IP — change if different
UDP_PORT      = 5005           # Port your laptop listens on

# ── Cameras ──────────────────────────────────────────────────────────────────
CAMERAS = [
    {"index": 0, "id": "cam1", "calibration": None},   # USB 3.0
    {"index": 2, "id": "cam2", "calibration": None},   # USB 3.0
    {"index": 4, "id": "cam3", "calibration": None},   # USB 2.0
]
# Change index numbers if v4l2-ctl --list-devices shows different values

# ── Shared calibration file ──────────────────────────────────────────────────
SHARED_CALIBRATION = "camera_calibration.json"
# Set to None if you have no calibration file (will use estimated values)

# ── IMU ──────────────────────────────────────────────────────────────────────
IMU_PORT     = None        # None = auto-detect. Or set e.g. "/dev/ttyUSB0"
IMU_BAUDRATE = 115200
USE_IMU      = True        # Set False to disable IMU

# ── Display ──────────────────────────────────────────────────────────────────
SHOW_PREVIEW = True        # Set False for headless / SSH-only operation

# ── Marker size ──────────────────────────────────────────────────────────────
# In camera_worker.py — change if your printed markers are not 15 mm:
self.marker_size = 0.015   # metres (15 mm)
```

### Camera calibration (optional but improves accuracy)

Without calibration, the tracker uses estimated camera parameters — accuracy is approximately ±2–5 mm. With a proper calibration file, accuracy improves to ±0.1–0.5 mm.

If you have a `camera_calibration.json` file, place it in `/home/pi/tracker/`. The format is:
```json
{
    "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "distortion_coefficients": [k1, k2, p1, p2, k3],
    "reprojection_error": 0.047
}
```

---

## 10. Running the Tracker

### On the Pi

```bash
cd /home/pi/tracker
python3 main.py
```

### Startup sequence

When you run the tracker you will see:

```
=================================================================
  6-DOF Tracker  |  cam1+cam2 (USB3)  cam3+IMU (USB2)
=================================================================

  cam1 → /dev/video0  (USB 3.0)
  cam2 → /dev/video2  (USB 3.0)
  cam3 → /dev/video4  (USB 2.0)
  IMU  → auto-detect  (USB 2.0)

  [cam1] using default calibration
  [cam1] /dev/video0  640x360 @ 60fps  MJPG
  [cam1] ready
  [cam2] /dev/video2  640x360 @ 60fps  MJPG
  [cam2] ready
  [cam3] /dev/video4  640x360 @ 60fps  MJPG
  [cam3] ready
  [IMU] opening /dev/ttyUSB0 @ 115200 baud
  [IMU] first packet OK  (port:/dev/ttyUSB0)
  IMU ready — press Z after placing IMU at marker-0 to zero it

  Cameras active: 3
  IMU: connected
  UDP → 10.0.0.2:5005
  Running — Q to quit
```

Once running, the terminal prints a live one-line status:
```
  [LIVE   ] X: +123.4 Y:  -45.6 Z: +789.0  R: +1.2 P: -0.5 Yw:+90.0  30fps
```

### On the laptop

Start your UDP receiver before or after starting the tracker — it does not matter which order.

---

## 11. Zeroing the IMU

The IMU must be zeroed before its angles are meaningful. This sets the "zero position" that all future angles are measured relative to.

**Procedure:**

1. Start the tracker (`python3 main.py`)
2. Place the IMU **physically at marker-0** — same position, same orientation
3. Hold still for 2–3 seconds so the IMU settles
4. Press **Z** on the keyboard (with the preview window focused)
5. The terminal prints: `[IMU] zeroed — origin set to current orientation`
6. The HUD shows `zeroed` next to the IMU status

From this point, all Roll/Pitch/Yaw values from the IMU are measured relative to the orientation at marker-0.

**To reset the zero** (go back to power-on reference): press **R**.

**Calibration status** is shown in the HUD as `Cal sys:3 gyro:3 accel:3 mag:3`. Values of 3 mean fully calibrated. If any value is 0 or 1, rotate the IMU gently in all directions for 10–20 seconds to let it calibrate before zeroing.

---

## 12. Understanding the Display

When `SHOW_PREVIEW = True`, a window opens showing:

```
┌──────────────────────────────────────────────────────────────────────┐
│  cam1 preview (426×240)  │  cam2 preview (426×240)  │  cam3 preview  │
│  ArUco markers highlighted with coloured borders and 3-D axes        │
│  cam1 | 30fps | markers:[0, 1]    cam2 | 29fps | markers:[2]        │
├──────────────────────────────────────────────────────────────────────┤
│ Status panel:                                                         │
│  6-DOF | 30fps | conf:18/20                                          │
│  LIVE — cameras tracking                              ← green        │
│                                                                      │
│  X   mm :  +123.45                                   ← green=live   │
│  Y   mm :   -45.67                                                   │
│  Z   mm :  +789.01                                                   │
│  Roll ° :    +1.23                                   ← green=camera  │
│  Pitch°  :   -0.56                                   ← cyan=IMU      │
│  Yaw  °  :  +90.12                                                   │
│                                                                      │
│  IMU 65Hz zeroed | R:+1.2 P:-0.5 Y:+90.1                           │
│    Cal sys:3 gyro:3 accel:3 mag:3                                    │
│                                                                      │
│  cam1  [0, 1]  30fps                                                 │
│  cam2  [2]     29fps                                                 │
│  cam3  (none)  30fps                                                 │
└──────────────────────────────────────────────────────────────────────┘
```

**Colour meanings:**

| Colour | Meaning |
|--------|---------|
| Green | Live camera data — accurate and current |
| Cyan | IMU data — cameras have lost markers |
| Orange | Frozen — holding last known camera position |
| Yellow/warm | Low confidence (fewer markers visible) |
| Red | Dead / no signal |

**Source banner meanings:**

| Banner | Meaning |
|--------|---------|
| `LIVE — cameras tracking` | All 6 values from cameras |
| `HOLD — IMU rotation, position frozen` | Position held, IMU driving rotation |
| `HOLD — last known pose (no IMU)` | Entire last pose held, no IMU available |
| `STALE — reporting last known pose` | 5-second freeze expired, sending stale data |

**Confidence score (0–20):**
Increases by 1 each frame markers are visible, decreases by 1 each frame they are not. Above 15 is good. Below 10 the display turns yellow and IMU yaw correction is paused.

---

## 13. Keyboard Controls

These keys work when the preview window is open and focused (click on it first):

| Key | Action |
|-----|--------|
| `Q` | Quit the tracker cleanly |
| `A` | Toggle 3-D axis arrows on detected markers (on/off) |
| `H` | Print marker affinity table to terminal |
| `Z` | Zero the IMU to current orientation |
| `R` | Reset IMU zero (return to power-on reference) |

**Marker affinity table** (press H) shows which camera most reliably detects each marker over the last 300 frames — useful for optimising camera placement:

```
── Marker Affinity ──
  cam1: m0(97%) m1(82%)
  cam2: m0(45%) m2(91%) m3(78%)
  cam3: m3(34%)
```

This tells you cam1 reliably sees markers 0 and 1, cam2 reliably sees markers 2 and 3.

---

## 14. Understanding the UDP Output

Every frame (approximately 30 times per second) the Pi sends one UDP packet to port 5005:

```
X,Y,Z,Roll,Pitch,Yaw
```

**Example packet:**
```
+123.45,-45.67,+789.01,+1.23,-0.56,+90.12
```

**Units:**
- X, Y, Z: millimetres, relative to marker 0
- Roll, Pitch, Yaw: degrees

**Coordinate system:**
- X: positive = right of marker 0
- Y: positive = above marker 0
- Z: positive = in front of marker 0
- Roll: positive = right side down
- Pitch: positive = nose up
- Yaw: positive = turned right

**When cameras have no markers:** the last valid pose continues to be sent. The packet format never changes — your receiver always gets 6 comma-separated values.

**Python parsing:**
```python
data, _ = sock.recvfrom(1024)
x, y, z, roll, pitch, yaw = [float(v) for v in data.decode().split(",")]
```

---

## 15. Auto-Start on Boot

To have the tracker start automatically every time the Pi powers on:

```bash
# Copy the service file
sudo cp /home/pi/tracker/tracker.service /etc/systemd/system/

# Enable it
sudo systemctl daemon-reload
sudo systemctl enable tracker
sudo systemctl start tracker

# Check status
sudo systemctl status tracker
```

**View live logs:**
```bash
journalctl -u tracker -f
```

**Stop auto-start:**
```bash
sudo systemctl disable tracker
sudo systemctl stop tracker
```

> **Note:** When running as a service, `SHOW_PREVIEW` in `main.py` must be set to `False` — the service runs headless with no display.

---

## 16. Troubleshooting

### Cameras show as DEAD

**Check device indices:**
```bash
v4l2-ctl --list-devices
```
The first `/dev/video` node listed under each EMEET camera is the correct one (e.g. `/dev/video0`, not `/dev/video1`). Update the `index` values in `CAMERAS` in `main.py` if they differ from 0, 2, 4.

**Check camera is detected by the OS:**
```bash
python3 -c "
import cv2
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
print('Opened:', cap.isOpened())
ret, frame = cap.read()
print('Read:', ret)
cap.release()
"
```

**Try a different USB port** — unplug and replug each camera one at a time.

---

### IMU shows as not connected

**Check the serial port:**
```bash
ls /dev/ttyUSB* /dev/ttyACM*
```
If nothing appears, the USB-UART adapter is not detected. Try a different USB port or check the adapter's USB connection.

**Check permissions:**
```bash
sudo usermod -a -G dialout pi
# Then log out and back in
```

**Test IMU data manually:**
```bash
python3 -c "
import serial, time
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)
for _ in range(10):
    line = ser.readline().decode(errors='ignore').strip()
    if line: print(line)
ser.close()
"
```
You should see lines containing `QUAT:` and `CAL:`. If not, check TX/RX wiring — try swapping TX and RX connections.

---

### No pose — markers seen: 0

The cameras are running but detecting no markers.

- Check that markers are **well lit** — avoid shadows or glare on the marker surface
- Check that markers are **printed at the correct size** (15 mm black square)
- Make sure you are using the **correct dictionary** — the system uses `DICT_4X4_50`
- Ensure at least one camera can see **marker 0 and at least one of markers 1, 2, or 3** at the same time — a camera that only sees marker 0 alone cannot produce a pose
- Try holding a marker directly in front of each camera and watching the preview to confirm detection works

---

### Pose jumps or is unstable

- Improve lighting — bright, even, diffuse light gives the most stable detections
- Check marker flatness — wrinkled or curved markers produce errors
- Increase the smoothing by reducing `alpha` in `smoother.py` (default 0.08; try 0.05)
- Ensure markers are rigidly mounted and not vibrating

---

### No UDP packets received on laptop

**Check the IP address** in `main.py`:
```python
UDP_ETH_IP = "10.0.0.2"   # must match your laptop's Ethernet IP
```

**Verify the Ethernet link:**
```bash
# From Pi:
ping 10.0.0.2

# From laptop:
ping 10.0.0.1
```

**Check firewall** — on Windows, allow UDP port 5005 through Windows Defender Firewall, or temporarily disable the firewall for testing.

---

### Cameras running at 30fps instead of 60fps

The EMEET S600 camera firmware on Linux caps at 30fps regardless of the 60fps setting — this is a known limitation of the camera's Linux driver. The system is designed to work correctly at 30fps. With 3 cameras running in parallel, the fused pose updates approximately 60–90 times per second because the three 30fps streams are staggered in time.

---

## 17. System Parameters Reference

Parameters you may want to adjust are collected here with their locations and effects:

| Parameter | File | Default | Effect |
|-----------|------|---------|--------|
| `UDP_ETH_IP` | main.py | `"10.0.0.2"` | Laptop IP address |
| `UDP_PORT` | main.py | `5005` | UDP port number |
| `FREEZE_TIME_SEC` | main.py | `5.0` | Seconds to hold position after marker loss before switching to "stale" mode |
| `YAW_CORRECT_CONFIDENCE` | main.py | `15` | Confidence threshold (0–20) to trigger IMU yaw drift correction |
| `SHOW_PREVIEW` | main.py | `True` | Show camera preview window |
| `USE_IMU` | main.py | `True` | Enable/disable IMU |
| `IMU_PORT` | main.py | `None` | Serial port (None = auto) |
| `AFFINITY_WINDOW` | main.py | `300` | Frames over which marker affinity is measured |
| `marker_size` | camera_worker.py | `0.015` | Physical marker size in metres |
| `alpha` | smoother.py | `0.08` | EMA smoothing factor (lower = smoother, more lag) |
| `MAX_POS_STEP` | smoother.py | `8.0 mm` | Maximum position change per frame |
| `MAX_ANGLE_STEP` | smoother.py | `3.0°` | Maximum angle change per frame |

---

## 18. File Reference

| File | Purpose |
|------|---------|
| `main.py` | Main program — starts cameras and IMU, runs fusion loop, sends UDP |
| `camera_worker.py` | Per-camera thread — captures frames, detects ArUco markers, runs solvePnP |
| `imu_worker.py` | IMU thread — reads BNO055 serial data, computes relative euler angles |
| `pose_fusion.py` | Combines detections from multiple cameras into one 6-DOF pose |
| `smoother.py` | Filters raw pose — median filter, step clamp, EMA, deadzone |
| `udp_sender.py` | Sends pose as UDP string to laptop |
| `tracker.service` | Systemd service file for auto-start on boot |
| `camera_calibration.json` | Camera calibration data — optional, improves accuracy |
| `README.md` | This document |

---

*End of manual*
