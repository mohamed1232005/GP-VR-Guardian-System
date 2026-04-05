// ============================================================================
// GuardianSystem.cs  —  v17  FLOOR-PROJECTED HUD RECTANGLE
// ============================================================================
// CHANGES FROM v16:
//  FIX 1 — HUD rect is HIDDEN on startup; only shown after real floor detected
//  FIX 2 — CreateAutoBoundary() returns early if no real floor hit yet
//  FIX 3 — Removed cam.y-1.5m fallback that caused fake boundary on empty screen
//  FIX 4 — UpdateHUDFromWorldPoints() projects 4 world corners → screen every frame
//  FIX 5 — Update() calls UpdateHUDFromWorldPoints() every frame when areaComplete
//  FIX 6 — PollFloorDetection() only calls CreateAutoBoundary on real raycast hit
// ============================================================================

using UnityEngine;
using UnityEngine.UI;
using System;
using System.Net.Sockets;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using Unity.Collections;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

#if UNITY_ANDROID
using UnityEngine.Android;
#endif

public class GuardianSystem : MonoBehaviour
{
    [Header("══ NETWORK ══")]
    public string serverIP   = "192.168.1.10";
    public int    serverPort = 9999;

    [Header("Performance")]
    public int  targetServerFPS  = 8;
    public int  processingWidth  = 320;
    public int  processingHeight = 240;
    [Range(10, 60)]
    public int  jpegQuality = 32;

    [Header("AR References")]
    public ARCameraManager   arCameraManager;
    public ARRaycastManager  arRaycastManager;
    public ARPlaneManager    arPlaneManager;
    public Canvas            arOverlayCanvas;
    public GameObject        connectingPanel;

    [Header("Hand Skeleton")]
    public bool  showHandSkeleton = true;
    public float jointSize        = 14f;
    public Color handColor        = Color.cyan;
    public Color pointingColor    = Color.yellow;
    public float boneThickness    = 6f;

    [Header("Fingertip Dot")]
    public bool  showFingertipDot  = true;
    public float fingertipDotSize  = 24f;
    public Color fingertipDotColor = Color.red;

    [Header("Boundary")]
    public Color boundaryColor     = Color.green;
    public float minPointDistanceM = 0.08f;
    public int   maxBoundaryPoints = 10;
    public float pointingHoldTime  = 0.5f;
    public float lineWidthM        = 0.02f;
    public float lineHeightOffset  = 0.015f;

    [Header("Auto Boundary")]
    [Tooltip("Shrink factor: 1.0 = full plane size, 0.8 = 80% of detected plane")]
    public float boundaryFitScale = 0.85f;  // scales the detected plane extents

    [Header("Status UI")]
    public Text statusText;
    public Text fpsText;
    public Text serverFpsText;

    [HideInInspector] public ServerResult latestData;

    // ── Camera ──────────────────────────────────────────────────────────────
    private WebCamTexture webcam;
    private Texture2D     captBuf;
    private bool          useWebCam    = false;
    private Matrix4x4?    displayMatrix = null;

    // ── Network ──────────────────────────────────────────────────────────────
    private TcpClient     tcp;
    private NetworkStream net;
    private Thread        netThread;
    private readonly Queue<byte[]> sendQ = new Queue<byte[]>();
    private readonly Queue<string> recvQ = new Queue<string>();
    private readonly object sendLk = new object();
    private readonly object recvLk = new object();
    private volatile bool connected = false;
    private volatile bool running   = false;
    private volatile bool needRecon = false;
    private const int SEND_Q_MAX    = 1;

    private float lastSend   = 0f;
    private float sendIval   = 0f;
    private int   sentFrames = 0;

    // ── Hand skeleton UI ────────────────────────────────────────────────────
    private readonly List<Image> jointDots = new List<Image>();
    private readonly List<Image> boneLines = new List<Image>();
    private GameObject fingertipDot;

    // ── Boundary world points ────────────────────────────────────────────────
    private readonly List<Vector3> bWorldPts = new List<Vector3>();
    private Vector3 bLastWorldPt;
    private bool areaComplete = false;
    private LineRenderer boundaryLine;      // kept for ClearBoundary compat (disabled)
    private readonly List<GameObject> bMarks = new List<GameObject>();

    // ── 2D HUD boundary (v17) ────────────────────────────────────────────────
    private GameObject hudBoundaryGO;
    private Image      hudFill;
    private Image      hudTop, hudBot, hudLeft, hudRight;

    // ── Manual drawing state ─────────────────────────────────────────────────
    private float pointHoldTimer    = 0f;
    private float lostPointingTimer = 0f;
    private bool  drawingUnlocked   = false;
    private readonly List<ARRaycastHit> rayHits = new List<ARRaycastHit>();

    // ── Misc UI ──────────────────────────────────────────────────────────────
    private GameObject bannerGO;
    private GameObject raycastCursor;
    private Image      holdRing;
    private GameObject holdRingGO;
    private Text       instrText;
    private Text       planeStatusText;
    private Text       arStateText;

    // ── FPS counters ─────────────────────────────────────────────────────────
    private float mFps    = 0f;
    private int   mFpsCnt = 0;
    private float mFpsTime = 0f;

    // ── AR state ─────────────────────────────────────────────────────────────
    private bool  cameraReady      = false;
    private bool  planeDetected    = false;
    private float detectedFloorY   = 0f;
    private float floorSearchTimer = 0f;
    private bool  floorRaycastDone = false;

    // ── Hand bone pairs ──────────────────────────────────────────────────────
    private static readonly (int a, int b)[] HAND_BONES = new (int, int)[]
    {
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
    };

    // =========================================================================
    // LIFECYCLE
    // =========================================================================

    void Start()
    {
        Debug.Log("[GS] Start v17 — Floor-projected HUD boundary");
        sendIval = 1f / Mathf.Max(1, targetServerFPS);
        mFpsTime = Time.time;

#if UNITY_ANDROID
        if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            Permission.RequestUserPermission(Permission.Camera);
            StartCoroutine(WaitPerm());
            return;
        }
#endif
        Boot();
    }

    IEnumerator WaitPerm()
    {
        while (!Permission.HasUserAuthorizedPermission(Permission.Camera))
            yield return new WaitForSeconds(0.3f);
        Boot();
    }

    void Boot()
    {
        BuildMinimalUI();
        BuildHandSkeletonUI();
        BuildExtraUI();
        BuildBoundaryLine();
        BuildRaycastCursor();
        InitCamera();
        RegisterAREvents();
    }

    void Update()
    {
        UpdateMFPS();
        UpdateARStateHUD();

        if (needRecon)
        {
            needRecon = false;
            CancelInvoke(nameof(ConnectSrv));
            Invoke(nameof(ConnectSrv), 2f);
            if (statusText) statusText.text = "Reconnecting in 2s…";
        }

        if (!cameraReady) return;

        if (connected && Time.time - lastSend >= sendIval)
        {
            CaptureAndSend();
            lastSend = Time.time;
        }

        DrainRecv();
        UpdateHandSkeleton();
        UpdateFingertipDot();

        // Poll floor detection every 2s until boundary placed
        if (!areaComplete)
            PollFloorDetection();

        HandleBoundary();

        // FIX 5: Re-project world corners every frame so rect tracks phone movement
        if (areaComplete)
            UpdateHUDFromWorldPoints();

        RefreshHUD();
    }

    void OnDestroy()
    {
        running = false;
        netThread?.Join(400);
        try { net?.Close(); }  catch { }
        try { tcp?.Close(); }  catch { }
        webcam?.Stop();

        if (arCameraManager != null)
            arCameraManager.frameReceived -= OnARFrameReceived;
        if (arPlaneManager != null)
            arPlaneManager.planesChanged -= OnPlanesChanged;
    }

    // =========================================================================
    // CAMERA INIT
    // =========================================================================

    void InitCamera()
    {
#if UNITY_EDITOR
        useWebCam = true;
        StartCamWebCam();
#else
        useWebCam = false;
        if (arCameraManager == null)
            arCameraManager = FindObjectOfType<ARCameraManager>();

        if (arCameraManager != null)
        {
            arCameraManager.frameReceived += OnARFrameReceived;
            cameraReady = true;
            Debug.Log("[CAM] AR mode active.");
            if (statusText) statusText.text = "AR Camera ✓\nDetecting floor…";
            Invoke(nameof(ConnectSrv), 1.0f);
        }
        else
        {
            Debug.LogError("[CAM] ARCameraManager not found!");
            if (statusText) statusText.text = "ERROR: Missing ARCameraManager!";
        }
#endif
    }

    void StartCamWebCam()
    {
        var devs = WebCamTexture.devices;
        if (devs.Length == 0) return;

        webcam = new WebCamTexture(devs[0].name, processingWidth, processingHeight, 30);
        webcam.Play();
        StartCoroutine(WebCamReady());
    }

    IEnumerator WebCamReady()
    {
        float t = 0f;
        while (!webcam.isPlaying && t < 8f)
        { yield return new WaitForSeconds(0.1f); t += 0.1f; }

        if (!webcam.isPlaying) yield break;

        captBuf = new Texture2D(webcam.width, webcam.height, TextureFormat.RGB24, false);
        cameraReady   = true;
        displayMatrix = Matrix4x4.identity;
        Debug.Log("[CAM] WebCam ready.");
        if (statusText) statusText.text = "WebCam ✓\nConnecting…";
        Invoke(nameof(ConnectSrv), 0.5f);
    }

    void OnARFrameReceived(ARCameraFrameEventArgs args)
    {
        if (!connected || Time.time - lastSend < sendIval) return;

        if (args.displayMatrix.HasValue)
            displayMatrix = args.displayMatrix.Value;

        XRCpuImage image;
        if (!arCameraManager.TryAcquireLatestCpuImage(out image)) return;

        var conversionParams = new XRCpuImage.ConversionParams
        {
            inputRect        = new RectInt(0, 0, image.width, image.height),
            outputDimensions = new Vector2Int(processingWidth, processingHeight),
            outputFormat     = TextureFormat.RGB24,
            transformation   = XRCpuImage.Transformation.None
        };

        int size = image.GetConvertedDataSize(conversionParams);
        var buf  = new NativeArray<byte>(size, Allocator.Temp);

        image.Convert(conversionParams, buf);
        image.Dispose();

        var tex = new Texture2D(processingWidth, processingHeight, TextureFormat.RGB24, false);
        tex.LoadRawTextureData(buf);
        tex.Apply();
        buf.Dispose();

        var jpg = tex.EncodeToJPG(jpegQuality);
        Destroy(tex);

        lock (sendLk) { sendQ.Clear(); sendQ.Enqueue(jpg); sentFrames++; }
        lastSend = Time.time;
    }

    void CaptureAndSend()
    {
        if (useWebCam && webcam != null && webcam.isPlaying)
        {
            try
            {
                captBuf.SetPixels(webcam.GetPixels());
                captBuf.Apply();
                var jpg = captBuf.EncodeToJPG(jpegQuality);
                lock (sendLk) { sendQ.Clear(); sendQ.Enqueue(jpg); sentFrames++; }
            }
            catch { }
        }
    }

    // =========================================================================
    // AR EVENTS
    // =========================================================================

    void RegisterAREvents()
    {
        if (arPlaneManager == null)
            arPlaneManager = FindObjectOfType<ARPlaneManager>();
        if (arPlaneManager != null)
            arPlaneManager.planesChanged += OnPlanesChanged;
    }

    // Best floor plane found so far — we use its real extents
    private ARPlane bestFloorPlane = null;

    void OnPlanesChanged(ARPlanesChangedEventArgs args)
    {
        var allChanges = new List<ARPlane>();
        allChanges.AddRange(args.added);
        allChanges.AddRange(args.updated);

        foreach (var plane in allChanges)
        {
            if (plane.alignment != PlaneAlignment.HorizontalUp) continue;

            float planeY = plane.transform.position.y;

            // Pick the largest floor plane (most likely the actual floor)
            bool isBetter = !planeDetected ||
                            planeY < detectedFloorY ||
                            (bestFloorPlane != null &&
                             plane.size.x * plane.size.y > bestFloorPlane.size.x * bestFloorPlane.size.y);

            if (isBetter)
            {
                detectedFloorY = planeY;
                planeDetected  = true;
                bestFloorPlane = plane;
                Debug.Log($"[AR] Best floor plane → Y={detectedFloorY:F3}m size={plane.size}");
                if (planeStatusText) planeStatusText.text = $"Floor ✓ Y={detectedFloorY:F2}m size={plane.size.x:F1}x{plane.size.y:F1}m";

                if (!areaComplete)
                    CreateAutoBoundary();
            }
        }
    }

    void UpdateARStateHUD()
    {
        if (arStateText == null) return;
#if !UNITY_EDITOR
        var state = ARSession.state;
        arStateText.text = state switch
        {
            ARSessionState.SessionTracking     => "AR:Tracking ✓",
            ARSessionState.SessionInitializing => "AR:Init…",
            _                                  => $"AR:{state}"
        };
#else
        arStateText.text = "AR:Editor";
#endif
    }

    // =========================================================================
    // FLOOR DETECTION — polls every 2s until boundary is placed
    // =========================================================================

    void PollFloorDetection()
    {
        if (areaComplete) return;

        floorSearchTimer += Time.deltaTime;
        if (floorSearchTimer < 2f) return;
        floorSearchTimer = 0f;

        if (arRaycastManager == null) return;

        var cam = Camera.main;
        if (cam == null) return;

        // Probe multiple screen positions — bottom area most likely to hit floor
        Vector2[] probePoints = new Vector2[]
        {
            new Vector2(Screen.width * 0.5f, Screen.height * 0.15f),
            new Vector2(Screen.width * 0.3f, Screen.height * 0.15f),
            new Vector2(Screen.width * 0.7f, Screen.height * 0.15f),
            new Vector2(Screen.width * 0.5f, Screen.height * 0.30f),
            new Vector2(Screen.width * 0.5f, Screen.height * 0.50f),
        };

        bool gotHit = false;
        foreach (var screenPt in probePoints)
        {
            if (arRaycastManager.Raycast(screenPt, rayHits, TrackableType.Planes))
            {
                float hitY = rayHits[0].pose.position.y;
                // Only accept if hit is clearly below the camera (real floor)
                if (hitY < cam.transform.position.y - 0.3f)
                {
                    if (!floorRaycastDone || hitY < detectedFloorY)
                    {
                        detectedFloorY   = hitY;
                        floorRaycastDone = true;
                        planeDetected    = true;
                        Debug.Log($"[FLOOR] Raycast hit Y={hitY:F3} screen={screenPt}");
                        if (planeStatusText) planeStatusText.text = $"Floor ✓ Y={hitY:F2}m";
                    }
                    gotHit = true;
                    break;
                }
            }
        }

        // FIX 6: Only create boundary on a REAL floor hit — no fallback drawing
        if (gotHit && !areaComplete)
        {
            CreateAutoBoundary();
        }
        else if (!gotHit && !floorRaycastDone)
        {
            // Just update the instruction text — do NOT draw anything
            if (instrText) instrText.text = "Point phone at the floor";
            Debug.Log("[FLOOR] No floor hit yet — waiting…");
        }
    }

    // =========================================================================
    // AUTO BOUNDARY — builds 4 world-space corners on the detected floor
    // =========================================================================

    void CreateAutoBoundary()
    {
        if (!planeDetected && !floorRaycastDone)
        {
            Debug.Log("[BOUNDARY] Skipped — no confirmed floor yet.");
            if (instrText) instrText.text = "Point phone at the floor…";
            return;
        }

        Debug.Log("[BOUNDARY] Creating AUTO boundary — fitting to detected plane…");

        float floorY = detectedFloorY + lineHeightOffset;
        Vector3 center;
        float hw, hd;
        Quaternion planeRot;

        if (bestFloorPlane != null)
        {
            // ── USE REAL PLANE DATA ───────────────────────────────────────────
            // ARCore gives us the plane centre, size, and orientation directly.
            // We just scale it down slightly so the rect fits inside the plane.
            center   = bestFloorPlane.transform.position;
            center.y = floorY;

            hw = (bestFloorPlane.size.x * 0.5f) * boundaryFitScale;
            hd = (bestFloorPlane.size.y * 0.5f) * boundaryFitScale;

            // Use the plane's own rotation for correct alignment
            planeRot = bestFloorPlane.transform.rotation;
        }
        else
        {
            // ── FALLBACK: raycast hit but no plane object ─────────────────────
            // Use camera forward to guess orientation
            var cam2 = Camera.main;
            if (cam2 == null) return;

            Vector3 fwd2 = cam2.transform.forward; fwd2.y = 0f;
            if (fwd2.sqrMagnitude < 0.01f) fwd2 = Vector3.forward;
            fwd2.Normalize();

            center   = new Vector3(cam2.transform.position.x + fwd2.x * 1.5f, floorY,
                                   cam2.transform.position.z + fwd2.z * 1.5f);
            hw       = 0.8f;
            hd       = 1.2f;
            planeRot = Quaternion.LookRotation(fwd2, Vector3.up);
        }

        // Build 4 corners using the plane's local right/forward axes
        Vector3 right   = planeRot * Vector3.right;
        Vector3 forward = planeRot * Vector3.forward;

        // Flatten both axes onto XZ — ensures corners stay on floorY
        right.y   = 0f; right.Normalize();
        forward.y = 0f; forward.Normalize();

        bWorldPts.Clear();
        bWorldPts.Add(new Vector3(center.x + right.x*hw + forward.x*hd, floorY, center.z + right.z*hw + forward.z*hd));
        bWorldPts.Add(new Vector3(center.x - right.x*hw + forward.x*hd, floorY, center.z - right.z*hw + forward.z*hd));
        bWorldPts.Add(new Vector3(center.x - right.x*hw - forward.x*hd, floorY, center.z - right.z*hw - forward.z*hd));
        bWorldPts.Add(new Vector3(center.x + right.x*hw - forward.x*hd, floorY, center.z + right.z*hw - forward.z*hd));

        UpdateBoundaryLine();
        MarkAreaComplete();
        UpdateHUDFromWorldPoints();

        Debug.Log($"[BOUNDARY] Fitted rect floorY={floorY:F3} center={center} hw={hw:F2} hd={hd:F2} usedPlane={bestFloorPlane != null}");
    }

    // =========================================================================
    // HUD PROJECTION — re-projects 4 world corners to screen every frame
    // This is the KEY FIX: the rect is NOT a static screen overlay.
    // It is computed from where the floor corners actually appear on screen.
    // =========================================================================

    void UpdateHUDFromWorldPoints()
    {
        if (bWorldPts.Count < 4 || hudBoundaryGO == null) return;

        var cam = Camera.main;
        if (cam == null) return;

        var canvasRect = arOverlayCanvas.GetComponent<RectTransform>();
        float canvasW  = canvasRect.rect.width;
        float canvasH  = canvasRect.rect.height;

        float minX = float.MaxValue, maxX = float.MinValue;
        float minY = float.MaxValue, maxY = float.MinValue;

        bool anyBehind = false;
        foreach (var wp in bWorldPts)
        {
            Vector3 vp = cam.WorldToViewportPoint(wp);

            // If any corner is behind the camera, hide the rect
            if (vp.z < 0f) { anyBehind = true; break; }

            // Viewport 0..1 → canvas space (origin = centre of canvas)
            float cx = (vp.x - 0.5f) * canvasW;
            float cy = (vp.y - 0.5f) * canvasH;

            if (cx < minX) minX = cx;
            if (cx > maxX) maxX = cx;
            if (cy < minY) minY = cy;
            if (cy > maxY) maxY = cy;
        }

        if (anyBehind)
        {
            hudBoundaryGO.SetActive(false);
            return;
        }

        float width  = maxX - minX;
        float height = maxY - minY;

        // Sanity check — if projection is degenerate, hide
        if (width < 10f || height < 10f)
        {
            hudBoundaryGO.SetActive(false);
            return;
        }

        // Anchor to canvas centre; position = centre of projected rect
        var rt = hudBoundaryGO.GetComponent<RectTransform>();
        rt.anchorMin = new Vector2(0.5f, 0.5f);
        rt.anchorMax = new Vector2(0.5f, 0.5f);
        rt.pivot     = new Vector2(0.5f, 0.5f);

        rt.anchoredPosition = new Vector2((minX + maxX) * 0.5f, (minY + maxY) * 0.5f);
        rt.sizeDelta        = new Vector2(width, height);

        hudBoundaryGO.SetActive(true);

        Debug.Log($"[HUD] canvasPos=({rt.anchoredPosition.x:F0},{rt.anchoredPosition.y:F0}) size=({width:F0}x{height:F0})");
    }

    // =========================================================================
    // NETWORK
    // =========================================================================

    void ConnectSrv()
    {
        if (connected) return;
        Debug.Log($"[NET] Connecting → {serverIP}:{serverPort}");

        try
        {
            tcp = new TcpClient();
            var ar = tcp.BeginConnect(serverIP, serverPort, null, null);
            if (!ar.AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(5)))
            { tcp.Close(); throw new TimeoutException(); }
            tcp.EndConnect(ar);

            net = tcp.GetStream();
            connected = true;
            running   = true;

            Debug.Log("[NET] Connected.");
            HideConnPanel();

            netThread = new Thread(NetLoop) { IsBackground = true };
            netThread.Start();
        }
        catch
        {
            connected = false;
            Invoke(nameof(ConnectSrv), 2f);
        }
    }

    void NetLoop()
    {
        var buf = new byte[4 * 1024 * 1024];
        try
        {
            while (running && connected)
            {
                byte[] frame = null;
                lock (sendLk)
                {
                    while (sendQ.Count > SEND_Q_MAX) sendQ.Dequeue();
                    if (sendQ.Count > 0) frame = sendQ.Dequeue();
                }

                if (frame != null)
                {
                    try
                    {
                        var sz = BitConverter.GetBytes((uint)frame.Length);
                        if (BitConverter.IsLittleEndian) Array.Reverse(sz);
                        net.Write(sz, 0, 4);
                        net.Write(frame, 0, frame.Length);
                        net.Flush();
                    }
                    catch { Disc(); break; }
                }

                if (net != null && net.DataAvailable)
                {
                    try
                    {
                        int hdr = 0;
                        while (hdr < 4)
                        {
                            int r = net.Read(buf, hdr, 4 - hdr);
                            if (r == 0) { Disc(); break; }
                            hdr += r;
                        }
                        if (!connected) break;

                        int jLen = BitConverter.ToInt32(buf, 0);
                        if (jLen <= 0 || jLen >= buf.Length) continue;

                        int got = 0;
                        while (got < jLen)
                        {
                            int r = net.Read(buf, got, jLen - got);
                            if (r == 0) break;
                            got += r;
                        }
                        if (got == jLen)
                        {
                            var json = Encoding.UTF8.GetString(buf, 0, jLen);
                            lock (recvLk) { recvQ.Enqueue(json); }
                        }
                    }
                    catch { Disc(); break; }
                }

                Thread.Sleep(5);
            }
        }
        catch { }
        finally { Disc(); }
    }

    void Disc()
    {
        connected = false;
        running   = false;
        needRecon = true;
        try { net?.Close(); } catch { }
        try { tcp?.Close(); } catch { }
    }

    void DrainRecv()
    {
        string json = null;
        lock (recvLk) { if (recvQ.Count > 0) json = recvQ.Dequeue(); }
        if (json == null) return;
        try
        {
            latestData = JsonUtility.FromJson<ServerResult>(json);
            if (latestData?.fps > 0 && serverFpsText)
                serverFpsText.text = $"Server: {latestData.fps:F0} FPS";
        }
        catch { latestData = null; }
    }

    // =========================================================================
    // HAND SKELETON UI
    // =========================================================================

    void BuildHandSkeletonUI()
    {
        if (!showHandSkeleton || arOverlayCanvas == null) return;

        for (int i = 0; i < 21; i++)
        {
            var go = new GameObject($"_HandJoint_{i}");
            go.transform.SetParent(arOverlayCanvas.transform, false);
            var rt = go.AddComponent<RectTransform>();
            rt.sizeDelta = new Vector2(jointSize, jointSize);
            var img = go.AddComponent<Image>();
            img.color = handColor;
            try { img.sprite = Resources.GetBuiltinResource<Sprite>("UI/Skin/Knob.psd"); } catch { }
            go.SetActive(false);
            jointDots.Add(img);
        }

        for (int i = 0; i < HAND_BONES.Length; i++)
        {
            var go = new GameObject($"_HandBone_{i}");
            go.transform.SetParent(arOverlayCanvas.transform, false);
            var img = go.AddComponent<Image>();
            img.color = new Color(handColor.r, handColor.g, handColor.b, 0.6f);
            go.SetActive(false);
            boneLines.Add(img);
        }
    }

    void UpdateHandSkeleton()
    {
        if (!showHandSkeleton)
        {
            foreach (var j in jointDots) if (j) j.gameObject.SetActive(false);
            foreach (var b in boneLines) if (b) b.gameObject.SetActive(false);
            return;
        }

        foreach (var j in jointDots) if (j) j.gameObject.SetActive(false);
        foreach (var b in boneLines) if (b) b.gameObject.SetActive(false);

        if (latestData?.hands == null || !latestData.hands.detected ||
            latestData.hands.hands == null || latestData.hands.hands.Count == 0)
            return;

        var hand = latestData.hands.hands[0];
        if (hand?.landmarks == null || hand.landmarks.Count < 21) return;

        Color col = hand.is_pointing ? pointingColor : handColor;

        Vector2[] pts = new Vector2[21];
        for (int i = 0; i < 21; i++)
        {
            var lm = hand.landmarks[i];
            pts[i] = TransformLandmarkToCanvas(lm.x, lm.y);

            if (i < jointDots.Count && jointDots[i] != null)
            {
                var rt = (RectTransform)jointDots[i].transform;
                rt.anchoredPosition = pts[i];
                jointDots[i].color  = col;
                jointDots[i].gameObject.SetActive(true);
            }
        }

        for (int i = 0; i < HAND_BONES.Length && i < boneLines.Count; i++)
        {
            var (a, b) = HAND_BONES[i];
            if (a >= 21 || b >= 21) continue;

            DrawUILine(boneLines[i], pts[a], pts[b], boneThickness);
            boneLines[i].color = new Color(col.r, col.g, col.b, 0.6f);
            boneLines[i].gameObject.SetActive(true);
        }
    }

    static void DrawUILine(Image img, Vector2 a, Vector2 b, float thickness)
    {
        var rt = (RectTransform)img.transform;
        Vector2 d   = (b - a);
        float   len = d.magnitude;

        rt.sizeDelta        = new Vector2(len, thickness);
        rt.anchoredPosition = (a + b) * 0.5f;

        float ang = Mathf.Atan2(d.y, d.x) * Mathf.Rad2Deg;
        rt.localRotation = Quaternion.Euler(0, 0, ang);
    }

    void UpdateFingertipDot()
    {
        if (!showFingertipDot || fingertipDot == null)
        {
            if (fingertipDot) fingertipDot.SetActive(false);
            return;
        }

        if (latestData?.hands == null || !latestData.hands.detected ||
            latestData.hands.hands == null || latestData.hands.hands.Count == 0)
        {
            fingertipDot.SetActive(false);
            return;
        }

        var hand = latestData.hands.hands[0];
        if (hand?.index_tip == null)
        {
            fingertipDot.SetActive(false);
            return;
        }

        fingertipDot.SetActive(true);
        fingertipDot.GetComponent<RectTransform>().anchoredPosition =
            TransformLandmarkToCanvas(hand.index_tip.x, hand.index_tip.y);

        fingertipDot.GetComponent<Image>().color =
            hand.is_pointing ? Color.magenta : fingertipDotColor;
    }

    // =========================================================================
    // COORDINATE TRANSFORMS
    // =========================================================================

    Vector2 TransformLandmarkToCanvas(float nx, float ny)
    {
        Vector2 screenPt;

        if (displayMatrix.HasValue)
        {
            var mat = displayMatrix.Value;
            float vx = nx * 2f - 1f;
            float vy = ny * 2f - 1f;

            Vector3 transformed = mat.MultiplyPoint3x4(new Vector3(vx, vy, 0));

            float tx = (transformed.x + 1f) * 0.5f;
            float ty = (transformed.y + 1f) * 0.5f;

            screenPt = new Vector2(tx * Screen.width, ty * Screen.height);
        }
        else
        {
            screenPt = new Vector2(nx * Screen.width, (1f - ny) * Screen.height);
        }

        return ScreenToCanvas(screenPt);
    }

    Vector2 ScreenToCanvas(Vector2 screenPt)
    {
        var canvasRect = arOverlayCanvas.GetComponent<RectTransform>();
        float w = canvasRect.rect.width;
        float h = canvasRect.rect.height;

        float nx = screenPt.x / Screen.width;
        float ny = screenPt.y / Screen.height;

        return new Vector2(
            nx * w - w * 0.5f,
            ny * h - h * 0.5f
        );
    }

    Vector2 CanvasToScreen(Vector2 canvasPos)
    {
        var canvasRect = arOverlayCanvas.GetComponent<RectTransform>();
        float w = canvasRect.rect.width;
        float h = canvasRect.rect.height;

        float nx = (canvasPos.x + w * 0.5f) / w;
        float ny = (canvasPos.y + h * 0.5f) / h;

        return new Vector2(nx * Screen.width, ny * Screen.height);
    }

    // =========================================================================
    // RAYCAST CURSOR (debug sphere)
    // =========================================================================

    void BuildRaycastCursor()
    {
        raycastCursor = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        raycastCursor.name = "_RaycastCursor";
        raycastCursor.transform.localScale = Vector3.one * 0.05f;

        var renderer = raycastCursor.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.material       = new Material(Shader.Find("Standard"));
            renderer.material.color = Color.red;
        }

        raycastCursor.SetActive(false);
        Debug.Log("[BOUNDARY] Raycast cursor created.");
    }

    // =========================================================================
    // BOUNDARY — manual drawing (skipped while areaComplete)
    // =========================================================================

    void HandleBoundary()
    {
        if (areaComplete) return;

        bool anyPointing = false;

        if (latestData?.hands != null
            && latestData.hands.detected && latestData.hands.hands != null)
        {
            foreach (var hand in latestData.hands.hands)
            {
                if (hand.is_pointing) { anyPointing = true; break; }
            }
        }

        if (Time.frameCount % 30 == 0)
        {
            int handCount = latestData?.hands?.hands?.Count ?? 0;
            Debug.Log($"[BOUNDARY] Hands:{handCount} Pointing:{anyPointing} " +
                      $"Floor:{planeDetected} DrawUnlocked:{drawingUnlocked} " +
                      $"HoldTimer:{pointHoldTimer:F2}s Points:{bWorldPts.Count}");
        }

        if (anyPointing && (planeDetected || useWebCam))
        {
            lostPointingTimer = 0f;
            pointHoldTimer   += Time.deltaTime;

            if (holdRingGO != null && latestData.hands.hands != null)
            {
                foreach (var hand in latestData.hands.hands)
                {
                    if (!hand.is_pointing || hand.index_tip == null) continue;
                    holdRingGO.GetComponent<RectTransform>().anchoredPosition =
                        TransformLandmarkToCanvas(hand.index_tip.x, hand.index_tip.y);
                    break;
                }
            }

            float progress = Mathf.Clamp01(pointHoldTimer / pointingHoldTime);
            if (holdRingGO != null) holdRingGO.SetActive(true);
            if (holdRing   != null) holdRing.fillAmount = progress;

            if (pointHoldTimer >= pointingHoldTime)
            {
                if (!drawingUnlocked)
                    Debug.Log($"[BOUNDARY] ✓ DRAWING UNLOCKED after {pointHoldTimer:F2}s!");
                drawingUnlocked = true;
            }
        }
        else
        {
            lostPointingTimer += Time.deltaTime;
            if (lostPointingTimer > 0.3f)
            {
                if (drawingUnlocked && !anyPointing)
                    Debug.Log("[BOUNDARY] Drawing locked (pointing lost)");
                pointHoldTimer  = 0f;
                drawingUnlocked = false;
                if (holdRingGO != null) holdRingGO.SetActive(false);
            }
        }

        UpdateInstructionText(anyPointing);

        if (!drawingUnlocked) return;

        if (bWorldPts.Count >= maxBoundaryPoints)
        {
            MarkAreaComplete();
            return;
        }

        if (latestData?.hands?.hands == null) return;

        foreach (var hand in latestData.hands.hands)
        {
            if (!hand.is_pointing || hand.index_tip == null) continue;

            Vector2 canvasPos = TransformLandmarkToCanvas(hand.index_tip.x, hand.index_tip.y);
            Vector2 screenPt  = CanvasToScreen(canvasPos);

            if (Time.frameCount % 30 == 0)
                Debug.Log($"[BOUNDARY] Raycast attempt: screen=({screenPt.x:F1},{screenPt.y:F1})");

#if UNITY_EDITOR
            TryAddWorldPointEditor(screenPt);
#else
            TryAddWorldPointAR(screenPt);
#endif
            break;
        }
    }

    void TryAddWorldPointAR(Vector2 screenPt)
    {
        if (arRaycastManager == null)
        {
            Debug.LogError("[BOUNDARY] arRaycastManager is NULL!");
            return;
        }

        bool hit     = false;
        TrackableType hitType = TrackableType.None;

        if (arRaycastManager.Raycast(screenPt, rayHits, TrackableType.Planes))
        { hit = true; hitType = TrackableType.Planes; }
        else if (arRaycastManager.Raycast(screenPt, rayHits, TrackableType.PlaneWithinPolygon))
        { hit = true; hitType = TrackableType.PlaneWithinPolygon; }
        else if (arRaycastManager.Raycast(screenPt, rayHits, TrackableType.PlaneWithinBounds))
        { hit = true; hitType = TrackableType.PlaneWithinBounds; }

        if (hit && rayHits.Count > 0)
        {
            Vector3 worldPt = rayHits[0].pose.position;
            if (raycastCursor != null)
            {
                raycastCursor.SetActive(true);
                raycastCursor.transform.position = worldPt;
            }
            Debug.Log($"[BOUNDARY] ✓ Raycast HIT! type:{hitType} world:{worldPt}");
            MaybeAddWorldPoint(worldPt);
        }
        else
        {
            if (raycastCursor != null) raycastCursor.SetActive(false);
            if (Time.frameCount % 60 == 0)
                Debug.LogWarning($"[BOUNDARY] ✗ Raycast MISS at screen=({screenPt.x:F1},{screenPt.y:F1})");
        }
    }

    void TryAddWorldPointEditor(Vector2 screenPt)
    {
        var cam = Camera.main;
        if (cam == null) return;
        var ray = cam.ScreenPointToRay(new Vector3(screenPt.x, screenPt.y, 0));
        if (Mathf.Abs(ray.direction.y) < 1e-5f) return;
        float t = -ray.origin.y / ray.direction.y;
        if (t < 0) return;
        MaybeAddWorldPoint(ray.origin + ray.direction * t);
    }

    void MaybeAddWorldPoint(Vector3 worldPt)
    {
        if (bWorldPts.Count > 0)
        {
            float dist = Vector3.Distance(worldPt, bLastWorldPt);
            if (dist < minPointDistanceM)
            {
                if (Time.frameCount % 120 == 0)
                    Debug.Log($"[BOUNDARY] Point too close: {dist:F3}m < {minPointDistanceM}m");
                return;
            }
        }

        worldPt.y += lineHeightOffset;
        bWorldPts.Add(worldPt);
        bLastWorldPt = worldPt;

        Debug.Log($"[BOUNDARY] ✓✓ POINT ADDED! Total: {bWorldPts.Count}/{maxBoundaryPoints} at {worldPt}");
        UpdateBoundaryLine();
    }

    void UpdateBoundaryLine()
    {
        if (boundaryLine == null) return;
        int n = bWorldPts.Count;
        boundaryLine.positionCount = (n >= 3) ? n + 1 : n;
        for (int i = 0; i < n; i++)
            boundaryLine.SetPosition(i, bWorldPts[i]);
        if (n >= 3)
            boundaryLine.SetPosition(n, bWorldPts[0]);
    }

    void MarkAreaComplete()
    {
        areaComplete    = true;
        drawingUnlocked = false;
        if (holdRingGO) holdRingGO.SetActive(false);
        if (instrText)  instrText.text = "";
        Debug.Log("[BOUNDARY] Area complete — HUD rect is now floor-projected.");
    }

    public void ClearBoundary()
    {
        foreach (var m in bMarks) if (m) Destroy(m);
        bMarks.Clear();
        bWorldPts.Clear();
        areaComplete      = false;
        drawingUnlocked   = false;
        pointHoldTimer    = 0f;
        lostPointingTimer = 0f;
        floorRaycastDone  = false;
        planeDetected     = false;
        floorSearchTimer  = 0f;
        bestFloorPlane    = null;

        if (holdRingGO) holdRingGO.SetActive(false);

        // FIX 3: Hide HUD rect completely until floor is re-detected
        if (hudBoundaryGO) hudBoundaryGO.SetActive(false);

        if (instrText) instrText.text = "Point phone at the floor";
        if (planeStatusText) planeStatusText.text = "Floor: searching…";

        Debug.Log("[BOUNDARY] Cleared — waiting for new floor detection.");
    }

    // =========================================================================
    // INSTRUCTION TEXT
    // =========================================================================

    void UpdateInstructionText(bool anyPointing)
    {
        if (instrText == null) return;

        if (areaComplete)
            instrText.text = "";
        else if (!planeDetected && !floorRaycastDone)
            instrText.text = "Point phone at the floor";
        else if (drawingUnlocked)
            instrText.text = "DRAWING — keep pointing";
        else if (anyPointing)
            instrText.text = $"Hold… ({pointHoldTimer:F1}s/{pointingHoldTime:F1}s)";
        else
            instrText.text = "Point finger to draw";
    }

    // =========================================================================
    // UI BUILDERS
    // =========================================================================

    void BuildMinimalUI()
    {
        if (arOverlayCanvas == null)
        {
            var ovGO = new GameObject("_OverlayCanvas");
            arOverlayCanvas = ovGO.AddComponent<Canvas>();
            arOverlayCanvas.renderMode   = RenderMode.ScreenSpaceOverlay;
            arOverlayCanvas.sortingOrder = 10;
            Canvas.ForceUpdateCanvases();
            ovGO.AddComponent<CanvasScaler>();
            ovGO.AddComponent<GraphicRaycaster>();
        }

        HideConnPanel();

        if (statusText    == null) statusText    = MkText("_Status",  new Vector2(.01f,.80f), new Vector2(.55f,.19f), 20);
        if (fpsText       == null) fpsText       = MkText("_FPS",     new Vector2(.66f,.95f), new Vector2(.33f,.04f), 20);
        if (serverFpsText == null) serverFpsText = MkText("_SrvFPS",  new Vector2(.66f,.90f), new Vector2(.33f,.04f), 20);

        arStateText = MkText("_ARState", new Vector2(.66f,.85f), new Vector2(.33f,.04f), 18);
        arStateText.color = Color.cyan;

        instrText = MkText("_Instr", new Vector2(.1f,.91f), new Vector2(.8f,.06f), 24);
        instrText.alignment = TextAnchor.MiddleCenter;
        instrText.color     = new Color(1f, 1f, .3f, 1f);
        instrText.text      = "Point phone at the floor";

        planeStatusText = MkText("_Plane", new Vector2(.01f,.76f), new Vector2(.55f,.03f), 18);
        planeStatusText.color = Color.cyan;
        planeStatusText.text  = "Floor: searching…";

        if (showFingertipDot)
        {
            fingertipDot = new GameObject("_FingertipDot");
            fingertipDot.transform.SetParent(arOverlayCanvas.transform, false);
            var rt = fingertipDot.AddComponent<RectTransform>();
            rt.sizeDelta = new Vector2(fingertipDotSize, fingertipDotSize);
            var img = fingertipDot.AddComponent<Image>();
            try { img.sprite = Resources.GetBuiltinResource<Sprite>("UI/Skin/Knob.psd"); } catch { }
            img.color = fingertipDotColor;
            fingertipDot.SetActive(false);
        }
    }

    void BuildExtraUI()
    {
        if (!arOverlayCanvas) return;

        var ringGO = MkImg("_HoldRing", 80, 80, new Color(1f, 1f, 0, .7f));
        var rrt    = ringGO.GetComponent<RectTransform>();
        rrt.anchorMin = rrt.anchorMax = new Vector2(.5f, .5f);
        holdRing = ringGO.GetComponent<Image>();
        holdRing.type       = Image.Type.Filled;
        holdRing.fillMethod = Image.FillMethod.Radial360;
        holdRing.fillAmount = 0f;
        ringGO.SetActive(false);
        holdRingGO = ringGO;

        var btnGO = new GameObject("_ClearBtn");
        btnGO.transform.SetParent(arOverlayCanvas.transform, false);
        var brtB = btnGO.AddComponent<RectTransform>();
        brtB.anchorMin = new Vector2(.25f, .02f);
        brtB.anchorMax = new Vector2(.75f, .09f);
        brtB.offsetMin = brtB.offsetMax = Vector2.zero;
        var bg  = btnGO.AddComponent<Image>(); bg.color = new Color(.85f, .1f, .1f, .9f);
        var btn = btnGO.AddComponent<Button>(); btn.targetGraphic = bg;
        btn.onClick.AddListener(ClearBoundary);
        MkChildText(btnGO, "CLEAR BOUNDARY", 26, Color.white, TextAnchor.MiddleCenter);
    }

    void BuildBoundaryLine()
    {
        // Legacy 3D LineRenderer — disabled; we use 2D HUD rect instead
        var go = new GameObject("_BoundaryLine");
        boundaryLine = go.AddComponent<LineRenderer>();
        boundaryLine.positionCount = 0;
        boundaryLine.enabled       = false;

        Build2DBoundary();
    }

    void Build2DBoundary()
    {
        if (arOverlayCanvas == null) return;

        float border = 8f;

        Color neon = new Color(0.2f, 1f, 0.1f, 1f);
        Color fill = new Color(0.2f, 1f, 0.1f, 0.07f);

        hudBoundaryGO = new GameObject("_HUDBoundary");
        hudBoundaryGO.transform.SetParent(arOverlayCanvas.transform, false);

        // Use a plain RectTransform — size is set dynamically in UpdateHUDFromWorldPoints()
        var parentRT = hudBoundaryGO.AddComponent<RectTransform>();
        parentRT.anchorMin = Vector2.zero;
        parentRT.anchorMax = Vector2.zero;
        parentRT.pivot     = new Vector2(0.5f, 0.5f);
        parentRT.sizeDelta = Vector2.zero;

        // Semi-transparent fill
        var fillGO = new GameObject("_Fill");
        fillGO.transform.SetParent(hudBoundaryGO.transform, false);
        var fillRT = fillGO.AddComponent<RectTransform>();
        fillRT.anchorMin = Vector2.zero;
        fillRT.anchorMax = Vector2.one;
        fillRT.offsetMin = fillRT.offsetMax = Vector2.zero;
        hudFill = fillGO.AddComponent<Image>();
        hudFill.color = fill;

        // Neon border sides
        hudTop   = MkBorder("_Top",   hudBoundaryGO, new Vector2(0, 1), new Vector2(1, 1), border, true,  neon);
        hudBot   = MkBorder("_Bot",   hudBoundaryGO, new Vector2(0, 0), new Vector2(1, 0), border, true,  neon);
        hudLeft  = MkBorder("_Left",  hudBoundaryGO, new Vector2(0, 0), new Vector2(0, 1), border, false, neon);
        hudRight = MkBorder("_Right", hudBoundaryGO, new Vector2(1, 0), new Vector2(1, 1), border, false, neon);

        // FIX 1: HIDDEN on startup — only shown once floor is confirmed
        hudBoundaryGO.SetActive(false);

        Debug.Log("[BOUNDARY] ✓ 2D HUD neon rectangle built (hidden until floor detected)");
    }

    Image MkBorder(string name, GameObject parent, Vector2 anchorMin, Vector2 anchorMax,
                   float thickness, bool horizontal, Color col)
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent.transform, false);
        var rt = go.AddComponent<RectTransform>();

        if (horizontal)
        {
            rt.anchorMin = anchorMin;
            rt.anchorMax = anchorMax;
            float pivot  = (anchorMin.y > 0.5f) ? 1f : 0f;
            rt.pivot     = new Vector2(0.5f, pivot);
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;
            rt.sizeDelta = new Vector2(0, thickness);
        }
        else
        {
            rt.anchorMin = anchorMin;
            rt.anchorMax = anchorMax;
            float pivot  = (anchorMin.x > 0.5f) ? 1f : 0f;
            rt.pivot     = new Vector2(pivot, 0.5f);
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;
            rt.sizeDelta = new Vector2(thickness, 0);
        }

        var img = go.AddComponent<Image>();
        img.color = col;
        return img;
    }

    // =========================================================================
    // HUD / UTILS
    // =========================================================================

    void ShowConnPanel(string msg)
    {
        if (connectingPanel == null) return;
        connectingPanel.SetActive(true);
        var t = connectingPanel.GetComponentInChildren<Text>();
        if (t) t.text = msg;
    }

    void HideConnPanel()
    {
        if (connectingPanel != null) connectingPanel.SetActive(false);
    }

    void RefreshHUD()
    {
        if (!statusText) return;
        string s = $"{(useWebCam ? "WebCam" : "ARCam")}:{(cameraReady ? "✓" : "…")}  Net:{(connected ? "✓" : "✗")}\n";
        s += $"Sent:{sentFrames}  Floor:{(planeDetected ? "✓" : "…")}\n";
        if (latestData != null)
        {
            s += $"Frame:{latestData.frame}\n";
            if (latestData.hands != null && latestData.hands.detected)
                s += $"Hands:{latestData.hands.hands.Count}\n";
            s += $"Boundary:{bWorldPts.Count}/{maxBoundaryPoints}pt";
        }
        statusText.text = s;
    }

    void UpdateMFPS()
    {
        mFpsCnt++;
        if (Time.time - mFpsTime < 1f) return;
        mFps    = mFpsCnt / (Time.time - mFpsTime);
        mFpsCnt = 0;
        mFpsTime = Time.time;
        if (fpsText) fpsText.text = $"Mobile: {mFps:F0} FPS";
    }

    GameObject MkImg(string n, float w, float h, Color col)
    {
        var go = new GameObject(n);
        go.transform.SetParent(arOverlayCanvas.transform, false);
        var rt = go.AddComponent<RectTransform>();
        rt.sizeDelta = new Vector2(w, h);
        go.AddComponent<Image>().color = col;
        return go;
    }

    Text MkText(string n, Vector2 amin, Vector2 asize, int sz)
    {
        var go = new GameObject(n);
        go.transform.SetParent(arOverlayCanvas.transform, false);
        var rt = go.AddComponent<RectTransform>();
        rt.anchorMin = amin;
        rt.anchorMax = amin + asize;
        rt.offsetMin = rt.offsetMax = Vector2.zero;
        var t = go.AddComponent<Text>();
        rt.pivot   = new Vector2(0.5f, 0.5f);
        t.font     = Resources.GetBuiltinResource<Font>("Arial.ttf");
        t.fontSize = sz;
        t.color    = Color.white;
        t.horizontalOverflow = HorizontalWrapMode.Overflow;
        t.verticalOverflow   = VerticalWrapMode.Overflow;
        var sh = go.AddComponent<Shadow>();
        sh.effectColor    = new Color(0, 0, 0, .9f);
        sh.effectDistance = new Vector2(1, -1);
        return t;
    }

    Text MkChildText(GameObject parent, string txt, int sz, Color col, TextAnchor align)
    {
        var go = new GameObject("_Lbl");
        go.transform.SetParent(parent.transform, false);
        var rt = go.AddComponent<RectTransform>();
        rt.anchorMin = Vector2.zero;
        rt.anchorMax = Vector2.one;
        rt.offsetMin = rt.offsetMax = Vector2.zero;
        var t = go.AddComponent<Text>();
        t.font      = Resources.GetBuiltinResource<Font>("Arial.ttf");
        t.fontSize  = sz;
        t.color     = col;
        t.alignment = align;
        t.text      = txt;
        return t;
    }
}