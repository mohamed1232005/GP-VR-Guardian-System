using UnityEngine;
using System.Net.Sockets;
using System;
using System.IO;
using System.Text;
using System.Threading;
using System.Collections.Generic;

#if UNITY_ANDROID
using UnityEngine.Android;
#endif

// ============================================================================
// GUARDIAN SYSTEM v3.0 - COMPLETE VR GUARDIAN FOR MOBILE
// ============================================================================

public class GuardianSystem : MonoBehaviour
{
    // ========================================================================
    // SERIALIZABLE DATA CLASSES
    // ========================================================================
    
    [Serializable]
    public class GroundData
    {
        public bool detected;
        public float[] plane;
        public int confidence;
    }

    [Serializable]
    public class HandLandmark
    {
        public float x;
        public float y;
        public float z;
    }

    [Serializable]
    public class HandData
    {
        public string handedness;
        public HandLandmark index_finger;
        public bool is_pointing;
        public List<HandLandmark> all_landmarks;
    }

    [Serializable]
    public class HandsData
    {
        public bool detected;
        public List<HandData> hands;
    }

    [Serializable]
    public class PoseLandmark
    {
        public float x;
        public float y;
        public float z;
        public float visibility;
    }

    [Serializable]
    public class PoseData
    {
        public bool detected;
        public List<PoseLandmark> landmarks;
    }

    [Serializable]
    public class PlayAreaData
    {
        public List<float[]> points;
        public bool is_complete;
        public int num_points;
    }

    [Serializable]
    public class GuardianResult
    {
        public int frame;
        public GroundData ground;
        public HandsData hands;
        public PoseData pose;
        public PlayAreaData play_area;
    }

    // ========================================================================
    // INSPECTOR FIELDS
    // ========================================================================
    
    [Header("Server Connection")]
    public string laptopIP = "192.168.100.14";
    public int port = 9999;

    [Header("Camera Settings")]
    public int cameraWidth = 480;
    public int cameraHeight = 360;
    public int cameraFPS = 20;

    [Header("Streaming")]
    public float sendFPS = 10f;
    [Range(1, 100)]
    public int jpegQuality = 35;

    [Header("Floor Visualization")]
    public float floorSize = 5f;
    public Color floorColor = new Color(0f, 1f, 0f, 0.4f);
    
    [Header("Play Area Drawing")]
    public bool enableDrawing = true;
    public Color areaColor = new Color(0f, 0.5f, 1f, 0.7f);
    public float areaLineWidth = 0.05f;
    public float pointSize = 0.1f;
    
    [Header("Hand Visualization")]
    public bool showHandJoints = true;
    public Color handColor = new Color(1f, 0f, 1f, 0.8f);
    public float handJointSize = 0.02f;
    
    [Header("Pose Visualization")]
    public bool showPoseSkeleton = false;
    public Color poseColor = new Color(1f, 1f, 0f, 0.8f);
    public float poseJointSize = 0.03f;

    // ========================================================================
    // PRIVATE FIELDS
    // ========================================================================
    
    // Network
    private WebCamTexture webcam;
    private TcpClient client;
    private NetworkStream stream;
    private Thread sendThread;
    private Thread receiveThread;
    private bool isRunning = false;
    private bool isConnected = false;

    // Camera
    private Texture2D sendTexture;
    private Queue<byte[]> frameQueue = new Queue<byte[]>();
    private object queueLock = new object();
    
    // Visualization objects
    private GameObject floorPlane;
    private GameObject playAreaContainer;
    private GameObject handVisualsContainer;
    private GameObject poseVisualsContainer;
    private Camera mainCamera;
    
    // State
    private GuardianResult latestResult;
    private object resultLock = new object();
    private int framesSent = 0;
    private string statusMessage = "Initializing...";
    
    // Play area drawing
    private List<GameObject> areaPointObjects = new List<GameObject>();
    private List<GameObject> areaLineObjects = new List<GameObject>();
    private bool isDrawingArea = false;
    
    // Hand tracking
    private Dictionary<string, GameObject> handJoints = new Dictionary<string, GameObject>();
    
    // Pose tracking
    private List<GameObject> poseJoints = new List<GameObject>();
    
    // Performance
    private float actualFPS = 0f;
    private int framesThisSecond = 0;
    private float fpsTimer = 0f;
    private float lastFrameTime = 0f;

    // ========================================================================
    // UNITY LIFECYCLE
    // ========================================================================

    void Start()
    {
        Application.targetFrameRate = 60;
        Screen.sleepTimeout = SleepTimeout.NeverSleep;
        
        Debug.Log("═══════════════════════════════════════");
        Debug.Log("VR GUARDIAN SYSTEM v3.0 - MOBILE");
        Debug.Log("═══════════════════════════════════════");

        SetupMainCamera();

#if UNITY_ANDROID
        if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            Permission.RequestUserPermission(Permission.Camera);
            statusMessage = "Requesting camera...";
            return;
        }
#endif

        CreateVisualizationObjects();
        InitializeCamera();
        Invoke("ConnectToServer", 2f);
    }

    void Update()
    {
        CalculateFPS();
        
        if (!isConnected || webcam == null || !webcam.isPlaying)
            return;

        // Capture frames at target FPS
        float timeSinceLastFrame = Time.time - lastFrameTime;
        if (timeSinceLastFrame >= (1f / sendFPS))
        {
            lastFrameTime = Time.time;
            CaptureAndQueueFrame();
        }

        // Update all visualizations
        UpdateAllVisualizations();
        
        // Handle user input
        HandleInput();
    }

    void OnApplicationQuit()
    {
        Cleanup();
    }

    void OnDestroy()
    {
        Cleanup();
    }

    // ========================================================================
    // INITIALIZATION
    // ========================================================================

    void SetupMainCamera()
    {
        mainCamera = Camera.main;
        
        if (mainCamera == null)
        {
            Camera[] cameras = FindObjectsOfType<Camera>();
            if (cameras.Length > 0)
            {
                mainCamera = cameras[0];
                mainCamera.tag = "MainCamera";
                Debug.Log($"[CAMERA] Tagged '{mainCamera.name}' as MainCamera");
            }
            else
            {
                GameObject camObj = new GameObject("Main Camera");
                mainCamera = camObj.AddComponent<Camera>();
                mainCamera.tag = "MainCamera";
                mainCamera.clearFlags = CameraClearFlags.SolidColor;
                mainCamera.backgroundColor = new Color(0.2f, 0.2f, 0.2f);
                mainCamera.transform.position = new Vector3(0, 1, -3);
                Debug.Log("[CAMERA] Created main camera");
            }
        }
    }

    void CreateVisualizationObjects()
    {
        // Floor plane
        floorPlane = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floorPlane.name = "GuardianFloor";
        floorPlane.transform.localScale = new Vector3(floorSize * 0.1f, 1f, floorSize * 0.1f);
        
        Collider col = floorPlane.GetComponent<Collider>();
        if (col != null) Destroy(col);
        
        Material floorMat = new Material(Shader.Find("Standard"));
        SetupTransparentMaterial(floorMat, floorColor);
        floorPlane.GetComponent<MeshRenderer>().material = floorMat;
        floorPlane.SetActive(false);
        
        // Containers
        playAreaContainer = new GameObject("PlayAreaContainer");
        handVisualsContainer = new GameObject("HandVisualsContainer");
        poseVisualsContainer = new GameObject("PoseVisualsContainer");
        
        Debug.Log("[VISUALS] Visualization objects created");
    }

    void SetupTransparentMaterial(Material mat, Color color)
    {
        mat.SetFloat("_Mode", 3);
        mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        mat.SetInt("_ZWrite", 0);
        mat.DisableKeyword("_ALPHATEST_ON");
        mat.EnableKeyword("_ALPHABLEND_ON");
        mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
        mat.renderQueue = 3000;
        mat.color = color;
        mat.EnableKeyword("_EMISSION");
        mat.SetColor("_EmissionColor", color * 0.3f);
    }

    void InitializeCamera()
    {
        Debug.Log("[CAMERA] Initializing...");
        
        WebCamDevice[] devices = WebCamTexture.devices;
        
        if (devices.Length == 0)
        {
            Debug.LogError("[CAMERA] No devices found!");
            statusMessage = "No camera";
            Invoke("InitializeCamera", 2f);
            return;
        }

        int selectedCamera = -1;
        for (int i = 0; i < devices.Length; i++)
        {
            Debug.Log($"[CAMERA] Device {i}: {devices[i].name} (Front: {devices[i].isFrontFacing})");
            if (!devices[i].isFrontFacing && selectedCamera == -1)
            {
                selectedCamera = i;
            }
        }

        if (selectedCamera == -1) selectedCamera = 0;

        try
        {
            webcam = new WebCamTexture(
                devices[selectedCamera].name,
                cameraWidth,
                cameraHeight,
                cameraFPS
            );
            
            webcam.Play();
            sendTexture = new Texture2D(cameraWidth, cameraHeight, TextureFormat.RGB24, false);
            StartCoroutine(WaitForCamera());
            
            Debug.Log($"[CAMERA] Started: {devices[selectedCamera].name}");
            statusMessage = "Camera starting...";
        }
        catch (Exception e)
        {
            Debug.LogError($"[CAMERA] Failed: {e.Message}");
            statusMessage = "Camera error";
            Invoke("InitializeCamera", 3f);
        }
    }

    System.Collections.IEnumerator WaitForCamera()
    {
        float timeout = 5f;
        float elapsed = 0f;
        
        while (elapsed < timeout)
        {
            if (webcam != null && webcam.isPlaying && webcam.width > 16)
            {
                Debug.Log($"[CAMERA] ✓ Active! {webcam.width}x{webcam.height}");
                statusMessage = "Camera ready";
                yield break;
            }
            
            yield return new WaitForSeconds(0.2f);
            elapsed += 0.2f;
        }
        
        Debug.LogError("[CAMERA] Timeout");
        statusMessage = "Camera timeout";
        Invoke("InitializeCamera", 2f);
    }

    // ========================================================================
    // NETWORK
    // ========================================================================

    void ConnectToServer()
    {
        if (isConnected) return;

        Debug.Log($"[NETWORK] Connecting to {laptopIP}:{port}...");
        statusMessage = "Connecting...";
        
        try
        {
            client = new TcpClient();
            client.Connect(laptopIP, port);
            stream = client.GetStream();
            
            client.ReceiveBufferSize = 262144;
            client.SendBufferSize = 262144;
            client.NoDelay = true;
            
            isConnected = true;
            isRunning = true;
            
            sendThread = new Thread(SendThread);
            sendThread.IsBackground = true;
            sendThread.Start();
            
            receiveThread = new Thread(ReceiveThread);
            receiveThread.IsBackground = true;
            receiveThread.Start();
            
            statusMessage = "Connected";
            Debug.Log("[NETWORK] ✓✓✓ Connected!");
        }
        catch (Exception e)
        {
            Debug.LogError($"[NETWORK] Failed: {e.Message}");
            statusMessage = "Connection failed";
            Invoke("ConnectToServer", 3f);
        }
    }

    void SendThread()
    {
        Debug.Log("[SEND] Thread started");
        
        while (isRunning)
        {
            try
            {
                byte[] frameData = null;
                
                lock (queueLock)
                {
                    if (frameQueue.Count > 0)
                    {
                        frameData = frameQueue.Dequeue();
                    }
                }
                
                if (frameData != null)
                {
                    byte[] sizeInfo = BitConverter.GetBytes((uint)frameData.Length);
                    if (BitConverter.IsLittleEndian)
                        Array.Reverse(sizeInfo);
                    
                    stream.Write(sizeInfo, 0, 4);
                    stream.Write(frameData, 0, frameData.Length);
                    stream.Flush();
                    
                    framesSent++;
                    
                    if (framesSent % 100 == 0)
                    {
                        Debug.Log($"[SEND] Frames sent: {framesSent}");
                    }
                }
                else
                {
                    Thread.Sleep(5);
                }
            }
            catch (Exception e)
            {
                if (isRunning)
                {
                    Debug.LogError($"[SEND] Error: {e.Message}");
                    isConnected = false;
                    break;
                }
            }
        }
        
        Debug.Log("[SEND] Thread stopped");
    }

    void ReceiveThread()
    {
        Debug.Log("[RECEIVE] Thread started");
        StreamReader reader = new StreamReader(stream, Encoding.UTF8);
        
        while (isRunning)
        {
            try
            {
                string line = reader.ReadLine();
                if (line == null) break;
                
                GuardianResult result = JsonUtility.FromJson<GuardianResult>(line);
                
                lock (resultLock)
                {
                    latestResult = result;
                }
                
                if (result.frame % 50 == 0 && result.ground != null && result.ground.detected)
                {
                    Debug.Log($"[RECEIVE] Floor detected! Confidence: {result.ground.confidence}%");
                }
            }
            catch (Exception e)
            {
                if (isRunning)
                {
                    Debug.LogError($"[RECEIVE] Error: {e.Message}");
                }
                break;
            }
        }
        
        Debug.Log("[RECEIVE] Thread stopped");
    }

    // ========================================================================
    // FRAME CAPTURE
    // ========================================================================

    void CalculateFPS()
    {
        framesThisSecond++;
        fpsTimer += Time.deltaTime;
        
        if (fpsTimer >= 1f)
        {
            actualFPS = framesThisSecond / fpsTimer;
            framesThisSecond = 0;
            fpsTimer = 0f;
        }
    }

    void CaptureAndQueueFrame()
    {
        if (webcam == null || !webcam.isPlaying || !webcam.didUpdateThisFrame) return;

        try
        {
            sendTexture.SetPixels32(webcam.GetPixels32());
            sendTexture.Apply(false);
            
            byte[] jpegData = sendTexture.EncodeToJPG(jpegQuality);
            
            lock (queueLock)
            {
                if (frameQueue.Count < 3)
                {
                    frameQueue.Enqueue(jpegData);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[CAPTURE] Error: {e.Message}");
        }
    }

    void HandleInput()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            isDrawingArea = !isDrawingArea;
            Debug.Log($"[DRAWING] {(isDrawingArea ? "ENABLED" : "DISABLED")}");
        }
        
        if (Input.GetKeyDown(KeyCode.C))
        {
            ClearPlayArea();
            Debug.Log("[DRAWING] Area cleared");
        }
    }

    // ========================================================================
    // VISUALIZATION UPDATES
    // ========================================================================

    void UpdateAllVisualizations()
    {
        GuardianResult current = null;
        lock (resultLock)
        {
            current = latestResult;
        }

        if (current == null) return;

        UpdateFloorVisualization(current.ground);
        UpdateHandVisualization(current.hands);
        UpdatePoseVisualization(current.pose);
        UpdatePlayAreaVisualization(current.play_area);
    }

    void UpdateFloorVisualization(GroundData ground)
    {
        if (ground == null || !ground.detected || ground.confidence < 40)
        {
            if (floorPlane != null && floorPlane.activeSelf)
            {
                floorPlane.SetActive(false);
            }
            return;
        }

        if (floorPlane == null) return;
        floorPlane.SetActive(true);

        Vector3 cameraPos = mainCamera.transform.position;
        Vector3 cameraForward = mainCamera.transform.forward;
        cameraForward.y = 0;
        cameraForward.Normalize();

        Vector3 floorPos = cameraPos + cameraForward * 2f;

        if (ground.plane != null && ground.plane.Length >= 4)
        {
            float a = ground.plane[0];
            float b = ground.plane[1];
            float c = ground.plane[2];
            float d = ground.plane[3];

            if (Mathf.Abs(b) > 0.01f)
            {
                float groundY = -(a * floorPos.x + c * floorPos.z + d) / b;
                floorPos.y = groundY + 0.01f;
            }
            else
            {
                floorPos.y = cameraPos.y - 1.5f;
            }

            floorPlane.transform.position = floorPos;

            Vector3 normal = new Vector3(a, b, c).normalized;
            floorPlane.transform.up = normal;
        }
        else
        {
            floorPos.y = cameraPos.y - 1.5f;
            floorPlane.transform.position = floorPos;
        }

        float alpha = Mathf.Clamp01(ground.confidence / 100f) * floorColor.a;
        float pulse = Mathf.Sin(Time.time * 2f) * 0.2f + 0.8f;
        alpha *= pulse;

        MeshRenderer renderer = floorPlane.GetComponent<MeshRenderer>();
        if (renderer != null && renderer.material != null)
        {
            Color finalColor = floorColor;
            finalColor.a = alpha;
            renderer.material.color = finalColor;
            renderer.material.SetColor("_EmissionColor", finalColor * 0.3f);
        }
    }

    void UpdateHandVisualization(HandsData hands)
    {
        if (!showHandJoints || hands == null || !hands.detected || hands.hands == null)
        {
            foreach (var joint in handJoints.Values)
            {
                if (joint != null) joint.SetActive(false);
            }
            return;
        }

        for (int i = 0; i < hands.hands.Count; i++)
        {
            HandData hand = hands.hands[i];
            if (hand == null || hand.index_finger == null) continue;
            
            string key = $"hand_{i}_index";
            GameObject joint = GetOrCreateJoint(key, handVisualsContainer.transform, handColor, handJointSize);
            
            Vector3 pos = ScreenToWorld(hand.index_finger.x, hand.index_finger.y, hand.index_finger.z);
            joint.transform.position = pos;
            joint.SetActive(true);
            
            if (hand.is_pointing)
            {
                joint.transform.localScale = Vector3.one * handJointSize * 3f;
                MeshRenderer rend = joint.GetComponent<MeshRenderer>();
                if (rend != null && rend.material != null)
                {
                    rend.material.color = Color.yellow;
                }
            }
            else
            {
                joint.transform.localScale = Vector3.one * handJointSize;
                MeshRenderer rend = joint.GetComponent<MeshRenderer>();
                if (rend != null && rend.material != null)
                {
                    rend.material.color = handColor;
                }
            }
        }
    }

    void UpdatePoseVisualization(PoseData pose)
    {
        if (!showPoseSkeleton || pose == null || !pose.detected || pose.landmarks == null)
        {
            foreach (var joint in poseJoints)
            {
                if (joint != null) joint.SetActive(false);
            }
            return;
        }

        int[] keyLandmarks = { 0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28 };
        
        for (int i = 0; i < keyLandmarks.Length; i++)
        {
            int idx = keyLandmarks[i];
            if (idx < pose.landmarks.Count)
            {
                PoseLandmark lm = pose.landmarks[idx];
                if (lm == null) continue;
                
                if (lm.visibility > 0.5f)
                {
                    string key = $"pose_{idx}";
                    GameObject joint = GetOrCreateJoint(key, poseVisualsContainer.transform, poseColor, poseJointSize);
                    
                    Vector3 pos = ScreenToWorld(lm.x, lm.y, lm.z);
                    joint.transform.position = pos;
                    joint.SetActive(true);
                }
            }
        }
    }

    void UpdatePlayAreaVisualization(PlayAreaData playArea)
    {
        if (playArea == null || playArea.points == null)
            return;

        if (playArea.points.Count != areaPointObjects.Count)
        {
            ClearPlayAreaVisuals();
        }

        for (int i = 0; i < playArea.points.Count; i++)
        {
            float[] point = playArea.points[i];
            if (point == null || point.Length < 3) continue;
            
            Vector3 pos = new Vector3(point[0], point[1], point[2]);
            
            GameObject pointObj = GetOrCreateAreaPoint(i);
            pointObj.transform.position = mainCamera.transform.position + pos;
            pointObj.SetActive(true);
        }

        if (playArea.points.Count > 1)
        {
            for (int i = 0; i < playArea.points.Count - 1; i++)
            {
                if (playArea.points[i] == null || playArea.points[i + 1] == null) continue;
                if (playArea.points[i].Length < 3 || playArea.points[i + 1].Length < 3) continue;
                
                Vector3 p1 = new Vector3(playArea.points[i][0], playArea.points[i][1], playArea.points[i][2]);
                Vector3 p2 = new Vector3(playArea.points[i + 1][0], playArea.points[i + 1][1], playArea.points[i + 1][2]);
                
                GameObject lineObj = GetOrCreateAreaLine(i);
                LineRenderer lr = lineObj.GetComponent<LineRenderer>();
                
                if (lr != null)
                {
                    lr.SetPosition(0, mainCamera.transform.position + p1);
                    lr.SetPosition(1, mainCamera.transform.position + p2);
                    lineObj.SetActive(true);
                }
            }
            
            if (playArea.is_complete && playArea.points.Count >= 3)
            {
                int lastIdx = playArea.points.Count - 1;
                if (playArea.points[lastIdx] != null && playArea.points[0] != null &&
                    playArea.points[lastIdx].Length >= 3 && playArea.points[0].Length >= 3)
                {
                    Vector3 p1 = new Vector3(playArea.points[lastIdx][0], playArea.points[lastIdx][1], playArea.points[lastIdx][2]);
                    Vector3 p2 = new Vector3(playArea.points[0][0], playArea.points[0][1], playArea.points[0][2]);
                    
                    GameObject lineObj = GetOrCreateAreaLine(lastIdx);
                    LineRenderer lr = lineObj.GetComponent<LineRenderer>();
                    
                    if (lr != null)
                    {
                        lr.SetPosition(0, mainCamera.transform.position + p1);
                        lr.SetPosition(1, mainCamera.transform.position + p2);
                        lineObj.SetActive(true);
                    }
                }
            }
        }
    }

    // ========================================================================
    // HELPER FUNCTIONS
    // ========================================================================

    Vector3 ScreenToWorld(float x, float y, float z)
    {
        Vector3 screenPos = new Vector3(x * Screen.width, (1 - y) * Screen.height, z * 2f + 1f);
        return mainCamera.ScreenToWorldPoint(screenPos);
    }

    GameObject GetOrCreateJoint(string key, Transform parent, Color color, float size)
    {
        if (!handJoints.ContainsKey(key) || handJoints[key] == null)
        {
            GameObject joint = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            joint.name = key;
            joint.transform.parent = parent;
            joint.transform.localScale = Vector3.one * size;
            
            Collider col = joint.GetComponent<Collider>();
            if (col != null) Destroy(col);
            
            Material mat = new Material(Shader.Find("Standard"));
            mat.color = color;
            mat.EnableKeyword("_EMISSION");
            mat.SetColor("_EmissionColor", color * 0.5f);
            joint.GetComponent<MeshRenderer>().material = mat;
            
            handJoints[key] = joint;
        }
        
        return handJoints[key];
    }

    GameObject GetOrCreateAreaPoint(int index)
    {
        while (areaPointObjects.Count <= index)
        {
            GameObject point = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            point.name = $"AreaPoint_{areaPointObjects.Count}";
            point.transform.parent = playAreaContainer.transform;
            point.transform.localScale = Vector3.one * pointSize;
            
            Collider col = point.GetComponent<Collider>();
            if (col != null) Destroy(col);
            
            Material mat = new Material(Shader.Find("Standard"));
            mat.color = areaColor;
            mat.EnableKeyword("_EMISSION");
            mat.SetColor("_EmissionColor", areaColor);
            point.GetComponent<MeshRenderer>().material = mat;
            
            areaPointObjects.Add(point);
        }
        
        return areaPointObjects[index];
    }

    GameObject GetOrCreateAreaLine(int index)
    {
        while (areaLineObjects.Count <= index)
        {
            GameObject line = new GameObject($"AreaLine_{areaLineObjects.Count}");
            line.transform.parent = playAreaContainer.transform;
            
            LineRenderer lr = line.AddComponent<LineRenderer>();
            lr.material = new Material(Shader.Find("Sprites/Default"));
            lr.startColor = areaColor;
            lr.endColor = areaColor;
            lr.startWidth = areaLineWidth;
            lr.endWidth = areaLineWidth;
            lr.positionCount = 2;
            
            areaLineObjects.Add(line);
        }
        
        return areaLineObjects[index];
    }

    void ClearPlayArea()
    {
        foreach (var obj in areaPointObjects)
        {
            if (obj != null) obj.SetActive(false);
        }
        
        foreach (var obj in areaLineObjects)
        {
            if (obj != null) obj.SetActive(false);
        }
    }

    void ClearPlayAreaVisuals()
    {
        foreach (var obj in areaPointObjects)
        {
            if (obj != null) Destroy(obj);
        }
        areaPointObjects.Clear();
        
        foreach (var obj in areaLineObjects)
        {
            if (obj != null) Destroy(obj);
        }
        areaLineObjects.Clear();
    }

    // ========================================================================
    // CLEANUP
    // ========================================================================

    void Cleanup()
    {
        Debug.Log("[CLEANUP] Shutting down...");
        
        isRunning = false;
        isConnected = false;
        
        if (sendThread != null && sendThread.IsAlive)
            sendThread.Join(1000);
        
        if (receiveThread != null && receiveThread.IsAlive)
            receiveThread.Join(1000);
        
        if (webcam != null && webcam.isPlaying)
            webcam.Stop();
        
        if (stream != null)
            stream.Close();
        
        if (client != null)
            client.Close();
        
        if (sendTexture != null)
            Destroy(sendTexture);
        
        Debug.Log($"[CLEANUP] Sent {framesSent} frames");
    }

    // ========================================================================
    // GUI
    // ========================================================================

    void OnGUI()
    {
        // Full screen camera preview
        if (webcam != null && webcam.isPlaying)
        {
            GUI.DrawTexture(new Rect(0, 0, Screen.width, Screen.height), webcam, ScaleMode.ScaleToFit);
        }

        // Status overlay
        int boxHeight = 280;
        GUI.Box(new Rect(10, 10, 420, boxHeight), "");
        
        GuardianResult current = null;
        lock (resultLock)
        {
            current = latestResult;
        }
        
        GUIStyle titleStyle = new GUIStyle(GUI.skin.label);
        titleStyle.fontSize = 20;
        titleStyle.fontStyle = FontStyle.Bold;
        GUI.Label(new Rect(20, 20, 400, 30), "VR Guardian v3.0", titleStyle);
        
        GUI.contentColor = isConnected ? Color.green : Color.red;
        GUIStyle statusStyle = new GUIStyle(GUI.skin.label);
        statusStyle.fontSize = 16;
        GUI.Label(new Rect(20, 50, 400, 20), $"● {statusMessage}", statusStyle);
        
        GUI.contentColor = Color.white;
        GUI.Label(new Rect(20, 75, 400, 20), $"Server: {laptopIP}:{port}");
        GUI.Label(new Rect(20, 95, 400, 20), $"Frames: {framesSent}");
        
        Color fpsColor = actualFPS >= sendFPS * 0.8f ? Color.green : 
                        actualFPS >= sendFPS * 0.5f ? Color.yellow : Color.red;
        GUI.contentColor = fpsColor;
        GUIStyle fpsStyle = new GUIStyle(GUI.skin.label);
        fpsStyle.fontStyle = FontStyle.Bold;
        GUI.Label(new Rect(20, 115, 400, 20), $"FPS: {actualFPS:F1}", fpsStyle);
        
        GUI.contentColor = Color.white;
        
        if (current != null)
        {
            // Ground status
            GUI.contentColor = current.ground != null && current.ground.detected ? Color.green : Color.yellow;
            string groundStatus = (current.ground != null && current.ground.detected) ? "✓ FLOOR DETECTED" : "Searching floor...";
            GUIStyle groundStyle = new GUIStyle(GUI.skin.label);
            groundStyle.fontStyle = FontStyle.Bold;
            GUI.Label(new Rect(20, 145, 400, 20), groundStatus, groundStyle);
            
            // Hand status
            GUI.contentColor = Color.magenta;
            if (current.hands != null && current.hands.detected && current.hands.hands != null)
            {
                GUIStyle handStyle = new GUIStyle(GUI.skin.label);
                handStyle.fontStyle = FontStyle.Bold;
                GUI.Label(new Rect(20, 170, 400, 20), $"👋 Hands: {current.hands.hands.Count}", handStyle);
                
                for (int i = 0; i < current.hands.hands.Count && i < 2; i++)
                {
                    HandData hand = current.hands.hands[i];
                    if (hand != null)
                    {
                        string pointingStatus = hand.is_pointing ? "👉 POINTING" : "🖐️ Open";
                        GUI.Label(new Rect(40, 190 + i * 20, 400, 20), 
                                 $"{hand.handedness}: {pointingStatus}");
                    }
                }
            }
            else
            {
                GUI.contentColor = Color.gray;
                GUI.Label(new Rect(20, 170, 400, 20), "No hands detected");
            }
            
            // Pose status
            GUI.contentColor = Color.yellow;
            if (current.pose != null && current.pose.detected)
            {
                GUIStyle poseStyle = new GUIStyle(GUI.skin.label);
                poseStyle.fontStyle = FontStyle.Bold;
                GUI.Label(new Rect(20, 220, 400, 20), "🧍 Pose: Detected", poseStyle);
            }
            
            // Play area status
            GUI.contentColor = Color.cyan;
            if (current.play_area != null && current.play_area.num_points > 0)
            {
                GUIStyle areaStyle = new GUIStyle(GUI.skin.label);
                areaStyle.fontStyle = FontStyle.Bold;
                GUI.Label(new Rect(20, 245, 400, 20), 
                         $"📐 Area: {current.play_area.num_points} points", areaStyle);
                
                if (current.play_area.is_complete)
                {
                    GUI.contentColor = Color.green;
                    GUI.Label(new Rect(40, 265, 400, 20), "✓ Area complete!");
                }
            }
        }
        
        // Drawing mode indicator
        if (isDrawingArea)
        {
            GUI.contentColor = Color.green;
            int indicatorY = Screen.height - 60;
            GUI.Box(new Rect(10, indicatorY, 200, 50), "");
            GUIStyle drawStyle = new GUIStyle(GUI.skin.label);
            drawStyle.fontSize = 18;
            drawStyle.fontStyle = FontStyle.Bold;
            GUI.Label(new Rect(20, indicatorY + 15, 180, 30), "🎨 DRAWING MODE", drawStyle);
        }
        
        // Instructions
        GUI.contentColor = Color.white;
        int instrY = Screen.height - 120;
        GUI.Box(new Rect(Screen.width - 310, instrY, 300, 110), "");
        GUIStyle instrStyle = new GUIStyle(GUI.skin.label);
        instrStyle.fontStyle = FontStyle.Bold;
        GUI.Label(new Rect(Screen.width - 300, instrY + 10, 280, 20), "Controls:", instrStyle);
        GUI.Label(new Rect(Screen.width - 300, instrY + 30, 280, 20), "SPACE - Toggle drawing");
        GUI.Label(new Rect(Screen.width - 300, instrY + 50, 280, 20), "C - Clear area");
        GUI.Label(new Rect(Screen.width - 300, instrY + 70, 280, 20), "Point finger to draw");
        
        GUI.contentColor = Color.white;
    }

    // ========================================================================
    // PUBLIC API
    // ========================================================================

    public WebCamTexture GetWebcamTexture()
    {
        return webcam;
    }
    
    public bool IsFloorDetected()
    {
        lock (resultLock)
        {
            return latestResult != null && latestResult.ground != null && latestResult.ground.detected;
        }
    }
    
    public bool AreHandsDetected()
    {
        lock (resultLock)
        {
            return latestResult != null && latestResult.hands != null && latestResult.hands.detected;
        }
    }
    
    public bool IsPoseDetected()
    {
        lock (resultLock)
        {
            return latestResult != null && latestResult.pose != null && latestResult.pose.detected;
        }
    }
    
    public void EnableDrawingMode(bool enable)
    {
        isDrawingArea = enable;
    }
    
    public float GetActualFPS()
    {
        return actualFPS;
    }
}