# GP VR Guardian System

GP VR Guardian System is a Unity + Python rehabilitation/guardian prototype.  
The Unity Android app streams AR camera frames to a Python computer-vision server. The Python server detects hand landmarks and gestures, then sends control data back to Unity so the scene can render a hand skeleton, ray/aim point, boundary visuals, warnings, and gesture-based ball interaction.

## Main Features

- AR floor detection and safe-space setup.
- Guardian boundary rendering.
- Corner/boundary proximity warning.
- Python MediaPipe hand tracking.
- Unity hand skeleton rendering.
- Gesture-based ball interaction:
  - Pinch = grab/carry the ball.
  - Open Palm = release the ball.
  - Point = aiming / marker interaction.
- TCP control channel for structured messages.
- UDP frame streaming from Unity to Python.

## Project Structure

### Python server files

| File | Purpose |
| --- | --- |
| `server.py` | Main Python entry point. Starts UDP, TCP, and CV worker. |
| `config.py` | Shared Python configuration: ports, gesture thresholds, smoothing values, warning distances. |
| `udp_handler.py` | Receives camera JPEG frames from Unity over UDP port `9001`. |
| `tcp_handler.py` | Handles TCP messages on port `9000` and sends hand/guardian results back to Unity. |
| `cv_worker.py` | Reads camera frames and runs the hand detector. |
| `hand_detector.py` | MediaPipe-based hand landmark and gesture processing. |
| `geometry.py` | Floor projection and guardian polygon geometry helpers. |
| `smoother.py` | Landmark smoothing, gesture vote buffering, and cooldown logic. |
| `session.py` | Creates/reset per-client runtime session state. |
| `gesture_recognizer.task` | MediaPipe gesture recognition model file. |
| `requirements.txt` | Python dependencies. |

### Unity C# files

| File | Purpose |
| --- | --- |
| `UDPFrameSender.cs` | Sends AR camera frames from Unity to Python over UDP. |
| `TCPControlChannel.cs` | Connects Unity to Python over TCP and dispatches incoming messages. |
| `FloorDetectionController.cs` | Detects and confirms AR floor planes. |
| `PointMarkerReceiver.cs` | Receives and displays boundary points. |
| `SafeSpaceSceneSpawner.cs` | Spawns the safe-space scene root and objects. |
| `BoundaryBoxRenderer.cs` | Renders boundary/box visuals. |
| `GuardianRenderer.cs` | Renders guardian polygon/walls. |
| `CornerProximityWarner.cs` | Warns when the user is close to safe-space corners/boundaries. |
| `HandSkeletonReceiver.cs` | Renders the hand skeleton from Python landmark data. |
| `HandCursorReceiver.cs` | Displays hand cursor / floor projected point. |
| `GestureBallInteractor.cs` | Handles pinch grab, ball movement, and open-palm release. |
| `VirtualRehabSceneController.cs` | Controls the virtual rehab scene. |
| `ImmersiveModeTransitionController.cs` | Switches between AR calibration and immersive rehab mode. |
| `StateUIController.cs` | Controls UI panels based on app state. |
| `ElegantXRUIStyler.cs` | Applies visual UI styling. |
| `WorldSpaceBillboard.cs` | Keeps world-space UI readable from the camera. |
| `link.xml` | Prevents IL2CPP stripping of required runtime types. |

## Requirements

### PC / Python side

- Windows PC connected to the same Wi-Fi network as the Android phone.
- Python 3.10+ recommended.
- Python virtual environment.
- Python packages installed from `requirements.txt`.
- Firewall must allow:
  - TCP `9000`
  - UDP `9001`

### Unity / Android side

- Unity 2022.3 LTS recommended.
- Android Build Support installed in Unity Hub.
- AR Foundation / ARCore-capable Android device.
- Android phone and PC must be on the same local network.
- Camera permission must be granted on the phone.

## Network Ports

| Direction | Protocol | Port | Purpose |
| --- | --- | --- | --- |
| Unity -> Python | UDP | `9001` | Camera frame streaming |
| Unity <-> Python | TCP | `9000` | Control messages and hand/guardian data |

## Python Setup

Open PowerShell in the project folder:

~~~powershell
cd "F:\OneDrive\Desktop\Grad-Env\try-5-yousef\MVP-Guardian-Unity\GP-VR-Guardian-System"
~~~

Create and activate a virtual environment:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
~~~

Install dependencies:

~~~powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

Run the Python server:

~~~powershell
python -u server.py
~~~

Expected output:

~~~text
[*] CV worker thread started
[*] UDP listening on :9001
[*] TCP listening on :9000
[HAND] detector initialized with model=...
[CV] worker ready
~~~

## Windows Firewall Setup

Run PowerShell as Administrator:

~~~powershell
New-NetFirewallRule -DisplayName "Guardian TCP 9000" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 9000

New-NetFirewallRule -DisplayName "Guardian UDP 9001" -Direction Inbound -Action Allow -Protocol UDP -LocalPort 9001
~~~

## Find the PC IP Address

On the Python PC:

~~~powershell
ipconfig
~~~

Use the active Wi-Fi adapter IPv4 address, for example:

~~~text
192.168.1.10
~~~

Then set this IP in Unity on both components:

- `TCPControlChannel.serverIP`
- `UDPFrameSender.serverIP`

## Unity Setup

1. Open the Unity project.
2. Open the main scene.
3. Confirm the scene has the required managers:
   - AR Session
   - AR Session Origin / XR Origin
   - AR Camera
   - Network Manager or objects containing:
     - `UDPFrameSender`
     - `TCPControlChannel`
   - Hand skeleton receiver object.
   - Gesture ball interactor object.
   - Floor detection controller.
   - Safe-space scene spawner.
4. Set `UDPFrameSender`:
   - Server IP = PC IPv4 address
   - Server Port = `9001`
   - Target FPS = `15`
   - JPEG Quality = `45`
   - Output Size = `320 x 240`
5. Set `TCPControlChannel`:
   - Server IP = PC IPv4 address
   - Server Port = `9000`
6. Build and run on Android.

## Runtime Flow

1. Start the Python server.
2. Build/run the Unity app on the Android phone.
3. Unity connects to Python over TCP.
4. Unity sends AR camera frames to Python over UDP.
5. Python detects hand landmarks/gestures.
6. Python sends `HAND_DATA` back to Unity.
7. Unity renders the hand skeleton and ray.
8. User confirms floor / safe space.
9. User uses gestures:
   - Pinch to grab/move the ball.
   - Open Palm to release the ball.
   - Point for marker/aim interactions.

## Useful Logcat Filters

In Unity Android Logcat, enable Regex and use:

~~~regex
\[(TCP|UDP|HAND|HAND SKELETON|BALL)\]|Pinch|Open_Palm|Closed_Fist|POINT|grabbed|released|Socket|Connect timeout|Network is unreachable
~~~

Network-only filter:

~~~regex
\[(TCP|UDP)\]|Connected|disconnected|Streaming OK|Socket|Connect timeout|Network is unreachable|Invalid arguments
~~~

Floor/guardian filter:

~~~regex
\[(FLOOR|GUARDIAN)\]|Plane accepted|Plane rejected|GUARDIAN_READY|safe space|boundary
~~~

## Expected Healthy Logs

Unity Logcat:

~~~text
[TCP] Connected to Python server.
[UDP] Streaming OK. sentFrames=...
[HAND SKELETON] Registered with TCPControlChannel.HandDataReceived.
[BALL] Registered with TCPControlChannel.HandDataReceived.
[BALL] Pinch grabbed...
[BALL] Open_Palm released...
~~~

Python terminal:

~~~text
[TCP] client connected: ...
[UDP] receiving OK packets=...
[CV] frame #...
[HAND] gesture=Pinch confirmed=Pinch ...
[HAND] gesture=Open_Palm confirmed=Open_Palm ...
[CV] produced response type=HAND_DATA ...
~~~

## Troubleshooting

### TCP timeout in Unity

If Unity shows:

~~~text
[TCP] Connection loop recovered from socket failure: Connect timeout
~~~

Check:

1. Python server is running.
2. `server.py` prints `TCP listening on :9000`.
3. Unity IP matches the PC IPv4 address.
4. Windows Firewall allows TCP `9000`.
5. Phone and PC are on the same Wi-Fi network.

### UDP works but no hand skeleton

If Unity shows:

~~~text
[UDP] Streaming OK
~~~

but no skeleton appears, check Python terminal for:

~~~text
[HAND] gesture=...
[CV] produced response type=HAND_DATA
~~~

If those lines are missing, Python is receiving frames but not producing usable hand data.

### Floor keeps getting rejected

If Unity shows:

~~~text
[FLOOR] Plane rejected for rehab...
~~~

Move the phone slowly over a larger visible floor area. The detected plane is smaller than the required safe-space size.

### Ball does not move

Check that Python logs show confirmed gestures:

~~~text
gesture=Pinch confirmed=Pinch
gesture=Open_Palm confirmed=Open_Palm
~~~

Then check Unity logs:

~~~text
[BALL] Pinch grabbed...
[BALL] Open_Palm released...
~~~

## Git Notes

Do not commit generated folders or environments:

- `.venv/`
- `__pycache__/`
- `Library/`
- `Temp/`
- `Obj/`
- `Build/`
- `Logs/`
- `.vscode/`

Use `.gitignore` to keep the repository clean.

## Maintainers

Graduation project team — GP VR Guardian System.
