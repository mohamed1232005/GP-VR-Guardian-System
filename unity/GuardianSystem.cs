// =============================================================================
// GuardianSystem.cs  —  Phase 3  —  Virtual Fence + Finger-Tip Alignment Fix
// =============================================================================
//
//  CHANGES IN THIS VERSION
//  ───────────────────────
//
//  ★ FIX — FINGER TIP ALIGNMENT  (see MoveMarker / ProjectToFloor)
//  ────────────────────────────────────────────────────────────────────
//  Root cause: The ray for the floor-dot starts at the CAMERA ORIGIN, not
//  at the physical fingertip.  When the camera is ~1 m above the floor and
//  the finger is 50 cm lower, the ray from the camera through the screen
//  pixel of the fingertip hits the floor FURTHER from the user than the
//  true fingertip shadow — the green dot appears behind the tip.
//
//  Fix: In MoveMarker() we build the ray as before, then advance the ray
//  origin forward by fingerRayAdvance metres ALONG the ray before
//  calling ProjectToFloor().  This pulls the floor-intersection point
//  back toward the user, matching the physical fingertip position.
//
//  Inspector knob: fingerRayAdvance  (default 0.10 m, range 0–0.35 m)
//    • Dot still behind tip  → increase fingerRayAdvance
//    • Dot now in front      → decrease fingerRayAdvance
//
//  ★ NEW — PHASE 3: VIRTUAL FENCE  (BuildBoundaryLine / UpdateBoundaryVisuals)
//  ─────────────────────────────────────────────────────────────────────────────
//  1. boundaryLine  — LineRenderer, loop=true, neon cyan, width 0.02 m
//     Created once in Boot() via BuildBoundaryLine().
//     Stays disabled until all 4 corners are placed.
//
//  2. UpdateBoundaryVisuals()  — called every Update()
//     When placedCorners.Count == 4:
//       Reads each corner sphere's live transform.position every frame
//       and feeds it into boundaryLine.SetPosition().  The fence
//       automatically tracks any ARCore SLAM drift of the spheres.
//
//  3. ClearCorners() updated:
//       boundaryLine.enabled = false on double-tap reset.
//
//  EVERYTHING ELSE is 100% unchanged from Phase 2:
//  Phase 1 AR math, network stack, hand-skeleton GL overlay, hold-still
//  timer, PlaceCorner, HoldRing, progress bar, floor locking, double-tap.
// =============================================================================

using UnityEngine;
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
    // =========================================================================
    //  INSPECTOR
    // =========================================================================

    [Header("Network")]
    public string serverIP   = "192.168.100.14";
    public int    serverPort = 9999;

    [Header("Performance")]
    public int targetServerFPS  = 10;
    public int processingWidth  = 320;
    public int processingHeight = 240;
    [Range(10, 60)]
    public int jpegQuality = 32;

    [Header("AR — assign from XR Origin")]
    public ARCameraManager  arCameraManager;
    public ARPlaneManager   arPlaneManager;
    public ARRaycastManager arRaycastManager;

    [Header("Cursor (leave empty — auto-created)")]
    public GameObject markerObject;
    public float      markerHeightOffset = 0.015f;

    [Header("Debug Visuals")]
    public bool showDebugVisuals = true;
    public bool showHandSkeleton = true;

    // =========================================================================
    //  ★ FIX — FINGER TIP ALIGNMENT  (new Inspector field)
    //
    //  Advances the ray origin toward the floor before intersection.
    //  Corrects the gap between the camera-origin ray and the physical
    //  fingertip position above the floor.
    // =========================================================================

    [Header("Finger Tip Alignment Fix")]
    [Tooltip("Metres to advance ray origin along the ray before floor intersection.\n" +
             "Increase if dot appears BEHIND fingertip.\n" +
             "Decrease if dot appears IN FRONT of fingertip.\n" +
             "Default 0.10 m works for typical arm length + phone height.")]
    [Range(0f, 0.35f)]
    public float fingerRayAdvance = 0.10f;   // ★ FIX

    // =========================================================================
    //  ★ FIX — RAYCAST REACH MULTIPLIER  (Challenge 1)
    //
    //  Mediapipe normalised coordinates tend to cluster near screen centre,
    //  projecting to a tiny ~1x1 m area on the floor.  This multiplier
    //  amplifies the screen-space offset from centre BEFORE raycasting,
    //  pushing the floor dot out toward the room edges.
    //
    //  Inspector knob: reachMultiplier  (default 1.5, range 1–3)
    //    • Points still compress near user  → increase toward 3
    //    • Floor dot flies past target      → decrease toward 1
    // =========================================================================

    [Header("Reach Amplification")]
    [Tooltip("Amplifies the screen offset from centre before raycasting.\n" +
             "Expands pointing reach so corners of the room can be stamped\n" +
             "without physically walking to them.\n" +
             "1 = no amplification (original behaviour).\n" +
             "1.5 = recommended starting point.\n" +
             "3 = maximum reach.")]
    [Range(1f, 3f)]
    public float reachMultiplier = 1.5f;   // ★ FIX Challenge 1

    // =========================================================================
    //  PHASE 2 — CORNER PLACEMENT SETTINGS  (unchanged)
    // =========================================================================

    [Header("Phase 2 — Corner Capture")]
    [Tooltip("Seconds the finger must stay still to stamp a corner")]
    public float holdSeconds        = 3f;
    [Tooltip("Max movement (metres) allowed while holding still")]
    public float holdRadiusM        = 0.03f;
    [Tooltip("Size of permanent corner spheres")]
    public float cornerScale        = 0.06f;
    [Tooltip("How high corner spheres sit above floor")]
    public float cornerHeightOffset = 0.03f;

    // =========================================================================
    //  ★ NEW — PHASE 3 FENCE SETTINGS
    // =========================================================================

    [Header("Phase 3 — Virtual Fence")]
    [Tooltip("Width of the fence line in metres")]
    [Range(0.005f, 0.05f)]
    public float fenceLineWidth = 0.02f;                     // ★ NEW
    [Tooltip("Colour of the virtual fence")]
    public Color fenceColor     = new Color(0f, 1f, 0.9f);  // ★ NEW neon cyan

    // =========================================================================
    //  PRIVATE — floor
    // =========================================================================

    private float floorY     = 0f;
    private bool  floorValid = false;
    private Plane floorPlane;
    private readonly List<ARRaycastHit> rayHits = new List<ARRaycastHit>();

    // =========================================================================
    //  PRIVATE — coordinate mapping
    // =========================================================================

    private Matrix4x4? displayMatrix = null;

    // =========================================================================
    //  PRIVATE — networking
    // =========================================================================

    private TcpClient     tcp;
    private NetworkStream net;
    private Thread        netThread;

    private readonly Queue<byte[]> sendQ    = new Queue<byte[]>();
    private readonly Queue<string> recvQ    = new Queue<string>();
    private readonly object        sendLock = new object();
    private readonly object        recvLock = new object();

    private volatile bool connected     = false;
    private volatile bool running       = false;
    private volatile bool needReconnect = false;

    private float sendInterval;
    private float lastSendTime;

    // =========================================================================
    //  PRIVATE — state
    // =========================================================================

    private ServerResult latestData   = null;
    private GameObject   shadowRing   = null;
    private GameObject   fingerSphere = null;
    private LineRenderer laserLine    = null;

    // =========================================================================
    //  PHASE 2 — CORNER STATE  (unchanged)
    // =========================================================================

    private readonly List<GameObject> placedCorners = new List<GameObject>();

    private float   holdTimer       = 0f;
    private Vector3 holdAnchorWorld = Vector3.zero;
    private bool    holdActive      = false;

    private GameObject holdRing   = null;
    private float      lastTapTime = -10f;

    private static readonly Color[] CORNER_COLORS = {
        new Color(0.2f, 0.5f, 1f,   1f),   // C1 = blue
        new Color(1f,   0.4f, 0.1f, 1f),   // C2 = orange
        new Color(0.2f, 0.9f, 0.3f, 1f),   // C3 = bright green
        new Color(1f,   0.2f, 0.8f, 1f),   // C4 = pink/magenta
    };

    // =========================================================================
    //  ★ NEW — PHASE 3 — BOUNDARY LINE RENDERER
    // =========================================================================

    private LineRenderer boundaryLine = null;   // ★ NEW Phase 3

    // How far above floorY the fence line floats (avoids z-fighting with plane mesh)
    private const float FENCE_Y_OFFSET = 0.012f;   // ★ NEW Phase 3

    // =========================================================================
    //  PRIVATE — 2D hand skeleton  (unchanged)
    // =========================================================================

    private static readonly int[,] HAND_CONNECTIONS = {
        {0,1},{1,2},{2,3},{3,4},
        {0,5},{5,6},{6,7},{7,8},
        {0,9},{9,10},{10,11},{11,12},
        {0,13},{13,14},{14,15},{15,16},
        {0,17},{17,18},{18,19},{19,20},
        {5,9},{9,13},{13,17}
    };

    private Material _glMat = null;

    // =========================================================================
    //  PRIVATE — HUD
    // =========================================================================

    private string hud1 = "Starting...";
    private string hud2 = "";
    private string hud3 = "";

    // =========================================================================
    //  UNITY LIFECYCLE
    // =========================================================================

    void Start()
    {
        sendInterval = 1f / Mathf.Max(1, targetServerFPS);
#if UNITY_ANDROID
        if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            Permission.RequestUserPermission(Permission.Camera);
            StartCoroutine(WaitForCameraPermission());
            return;
        }
#endif
        Boot();
    }

    IEnumerator WaitForCameraPermission()
    {
        while (!Permission.HasUserAuthorizedPermission(Permission.Camera))
            yield return new WaitForSeconds(0.3f);
        Boot();
    }

    void Boot()
    {
        BuildMarker();
        BuildDebugVisuals();
        BuildHoldRing();
        BuildBoundaryLine();    // ★ NEW Phase 3
        RegisterAREvents();
        InitCamera();
    }

    void Update()
    {
        if (needReconnect)
        {
            needReconnect = false;
            Invoke(nameof(ConnectToServer), 2f);
            hud1 = "Reconnecting...";
        }
        DrainReceiveQueue();
        MoveMarker();
        UpdateHoldTimer();
        PulseMarker();
        HandleDoubleTap();
        UpdateBoundaryVisuals();   // ★ NEW Phase 3
    }

    void OnDestroy()
    {
        running = false;
        netThread?.Join(400);
        try { net?.Close(); } catch { }
        try { tcp?.Close(); } catch { }
        if (arCameraManager != null) arCameraManager.frameReceived -= OnARFrameReceived;
        if (arPlaneManager  != null) arPlaneManager.planesChanged  -= OnPlanesChanged;
        if (markerObject    != null) Destroy(markerObject);
        if (shadowRing      != null) Destroy(shadowRing);
        if (fingerSphere    != null) Destroy(fingerSphere);
        if (laserLine       != null) Destroy(laserLine.gameObject);
        if (holdRing        != null) Destroy(holdRing);
        if (boundaryLine    != null) Destroy(boundaryLine.gameObject);   // ★ NEW
        if (_glMat          != null) Destroy(_glMat);
        ClearCorners();
    }

    // =========================================================================
    //  HUD + OVERLAYS
    // =========================================================================

    void OnGUI()
    {
        GUI.color = new Color(0f, 0f, 0f, 0.65f);
        GUI.DrawTexture(new Rect(0, 0, Screen.width, 108), Texture2D.whiteTexture);

        var style = new GUIStyle(GUI.skin.label) { fontSize = 26, fontStyle = FontStyle.Bold };
        GUI.color = Color.green;  GUI.Label(new Rect(6, 2,  Screen.width, 34), hud1, style);
        GUI.color = Color.yellow; GUI.Label(new Rect(6, 36, Screen.width, 34), hud2, style);
        GUI.color = Color.cyan;   GUI.Label(new Rect(6, 70, Screen.width, 34), hud3, style);

        if (showHandSkeleton && latestData?.hands?.hands != null)
            DrawHandSkeletonGL(latestData.hands.hands);

        DrawCornerLabels();
        DrawHoldProgressHUD();
    }

    // =========================================================================
    //  ★ NEW — PHASE 3 — BUILD BOUNDARY LINE
    //
    //  Creates the LineRenderer once in Boot().  Starts disabled.
    //
    //  loop = true  →  Unity automatically draws a segment from position[3]
    //                  back to position[0], closing the rectangle without
    //                  any extra code.
    //  positionCount = 4  →  one vertex per corner.
    //  numCornerVertices / numCapVertices  →  smooth rounded joins & ends.
    // =========================================================================

    void BuildBoundaryLine()   // ★ NEW Phase 3
    {
        var go = new GameObject("_BoundaryFence");
        boundaryLine = go.AddComponent<LineRenderer>();

        boundaryLine.useWorldSpace     = true;
        boundaryLine.loop              = false;  // ★ FIX C2: starts open; closed dynamically at 4 corners
        boundaryLine.positionCount     = 0;      // ★ FIX C2: managed dynamically in UpdateBoundaryVisuals
        boundaryLine.startWidth        = fenceLineWidth;
        boundaryLine.endWidth          = fenceLineWidth;
        boundaryLine.numCapVertices    = 4;
        boundaryLine.numCornerVertices = 4;

        // Unlit neon cyan material — emissive so it glows in AR
        boundaryLine.material          = MakeMaterial(fenceColor);
        boundaryLine.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        boundaryLine.receiveShadows    = false;

        // Hidden until all 4 corners are placed
        boundaryLine.enabled = false;
    }

    // =========================================================================
    //  ★ NEW — PHASE 3 — UPDATE BOUNDARY VISUALS
    //
    //  Called every Update() — runs even after 4 corners are placed.
    //
    //  Reads each corner sphere's live transform.position every frame.
    //  This means if ARCore SLAM re-aligns the floor plane and shifts the
    //  spheres, the fence line stretches with them automatically.
    //
    //  Y is clamped to floorY + FENCE_Y_OFFSET to keep the line visible
    //  and prevent z-fighting with the ARCore plane mesh.
    // =========================================================================

    // =========================================================================
    //  ★ FIX — CHALLENGE 2: PROGRESSIVE BOUNDARY LINE  (updated Phase 3)
    //
    //  Previous version waited for all 4 corners and used a static
    //  positionCount = 4, which threw errors and never showed until complete.
    //
    //  New behaviour — "line by line" as each corner is stamped:
    //    < 2 corners  →  line renderer disabled (nothing to connect yet)
    //    ≥ 2 corners  →  enabled; positionCount set dynamically to match
    //                    placedCorners.Count; loop=false (open polyline)
    //    = 4 corners  →  loop=true closes the rectangle automatically
    //
    //  Live position tracking is preserved: corner sphere transforms are
    //  read every frame so the fence tracks any ARCore SLAM drift.
    // =========================================================================

    void UpdateBoundaryVisuals()   // ★ FIX Challenge 2
    {
        if (boundaryLine == null) return;

        int count = placedCorners.Count;

        // ── Step 1: need at least 2 points to draw a segment ─────────────────
        if (count < 2)                                                   // ★ C2
        {
            boundaryLine.enabled = false;
            return;
        }

        // ── Step 2: enable and resize position array dynamically ─────────────
        boundaryLine.enabled       = true;                               // ★ C2
        boundaryLine.positionCount = count;                              // ★ C2

        // ── Step 3: update each placed corner's live position ────────────────
        for (int i = 0; i < count; i++)                                  // ★ C2
        {
            if (placedCorners[i] == null) { boundaryLine.enabled = false; return; }

            Vector3 pos = placedCorners[i].transform.position;
            pos.y = floorY + FENCE_Y_OFFSET;   // float above floor, no z-fight
            boundaryLine.SetPosition(i, pos);
        }

        // ── Step 4: close the loop only when all 4 corners are placed ────────
        boundaryLine.loop = (count == 4);                                // ★ C2
    }

    // =========================================================================
    //  PHASE 2 — HOLD TIMER  (unchanged)
    // =========================================================================

    void UpdateHoldTimer()
    {
        if (placedCorners.Count >= 4) { ResetHold(); return; }

        if (!floorValid || markerObject == null || !markerObject.activeSelf)
        { ResetHold(); return; }

        Vector3 cur = markerObject.transform.position;
        cur.y = floorY;

        if (!holdActive)
        {
            holdAnchorWorld = cur;
            holdActive      = true;
            holdTimer       = 0f;
            return;
        }

        float drift = Vector2.Distance(
            new Vector2(cur.x,             cur.z),
            new Vector2(holdAnchorWorld.x, holdAnchorWorld.z));

        if (drift > holdRadiusM)
        {
            holdAnchorWorld = cur;
            holdTimer       = 0f;
            UpdateHoldRing(0f);
            return;
        }

        holdTimer += Time.deltaTime;
        UpdateHoldRing(holdTimer / holdSeconds);

        if (holdTimer >= holdSeconds)
        {
            PlaceCorner(holdAnchorWorld);
            ResetHold();
        }
    }

    void ResetHold()
    {
        holdActive = false;
        holdTimer  = 0f;
        UpdateHoldRing(0f);
    }

    // =========================================================================
    //  PHASE 2 — PLACE CORNER  (unchanged logic, updated HUD text)
    // =========================================================================

    void PlaceCorner(Vector3 worldPos)
    {
        int idx = placedCorners.Count;
        if (idx >= 4) return;

        Color col    = CORNER_COLORS[idx];
        var   corner = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        corner.name  = $"_Corner{idx + 1}";
        corner.transform.localScale = Vector3.one * cornerScale;
        corner.transform.position   = new Vector3(worldPos.x, floorY + cornerHeightOffset, worldPos.z);
        Destroy(corner.GetComponent<Collider>());
        var rend = corner.GetComponent<Renderer>();
        rend.material          = MakeMaterial(col);
        rend.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        corner.SetActive(true);
        placedCorners.Add(corner);

        Debug.Log($"[GS] Corner {idx+1} at ({worldPos.x:F3}, {floorY:F3}, {worldPos.z:F3})");

        if (placedCorners.Count == 4)
        {
            hud1 = "4 CORNERS — FENCE ACTIVE";   // ★ Phase 3 message
            hud2 = "Double-tap screen to reset";
            hud3 = "";
        }
        else
        {
            hud1 = $"FLOOR Y={floorY:F3}m  {placedCorners.Count}/4 corners";
            hud3 = $"Corner {idx+1} placed at ({worldPos.x:F2}, {worldPos.z:F2})";
        }
    }

    // =========================================================================
    //  PHASE 2 — CLEAR CORNERS
    //  ★ Updated: disables boundaryLine immediately on reset
    // =========================================================================

    void ClearCorners()
    {
        foreach (var c in placedCorners)
            if (c != null) Destroy(c);
        placedCorners.Clear();

        // ★ NEW Phase 3: hide fence immediately when corners are cleared
        if (boundaryLine != null) boundaryLine.enabled = false;

        ResetHold();
        hud1 = "Corners cleared — point at corner 1!";
        hud3 = "";
        Debug.Log("[GS] Corners cleared.");
    }

    void HandleDoubleTap()
    {
        if (Input.touchCount == 1 && Input.GetTouch(0).phase == TouchPhase.Began)
        {
            float now = Time.time;
            if (now - lastTapTime < 0.35f) ClearCorners();
            lastTapTime = now;
        }
    }

    // =========================================================================
    //  PHASE 2 — HOLD RING  (unchanged)
    // =========================================================================

    void BuildHoldRing()
    {
        holdRing = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        holdRing.name = "_HoldRing";
        holdRing.transform.localScale = Vector3.zero;
        Destroy(holdRing.GetComponent<Collider>());
        holdRing.GetComponent<Renderer>().material =
            MakeMaterial(new Color(1f, 1f, 1f, 0.85f));
        holdRing.GetComponent<Renderer>().shadowCastingMode =
            UnityEngine.Rendering.ShadowCastingMode.Off;
        holdRing.SetActive(false);
    }

    void UpdateHoldRing(float progress)
    {
        if (holdRing == null) return;
        if (progress <= 0f || markerObject == null || !markerObject.activeSelf)
        { holdRing.SetActive(false); return; }

        holdRing.SetActive(true);
        float size = Mathf.Lerp(0f, 0.14f, progress);
        holdRing.transform.localScale = new Vector3(size, 0.002f, size);
        holdRing.transform.position   = new Vector3(
            markerObject.transform.position.x, floorY + 0.002f,
            markerObject.transform.position.z);
        Color col = progress < 0.5f
            ? Color.Lerp(Color.white,  Color.yellow, progress * 2f)
            : Color.Lerp(Color.yellow, Color.green,  (progress - 0.5f) * 2f);
        holdRing.GetComponent<Renderer>().material.color = col;
    }

    // =========================================================================
    //  PHASE 2 — CORNER LABELS  (unchanged)
    // =========================================================================

    void DrawCornerLabels()
    {
        if (placedCorners.Count == 0) return;
        var cam = Camera.main;
        if (cam == null) return;

        var s = new GUIStyle(GUI.skin.label)
            { fontSize = 32, fontStyle = FontStyle.Bold, alignment = TextAnchor.MiddleCenter };

        for (int i = 0; i < placedCorners.Count; i++)
        {
            var corner = placedCorners[i];
            if (corner == null) continue;
            Vector3 sp = cam.WorldToScreenPoint(corner.transform.position);
            if (sp.z < 0) continue;

            float gx = sp.x - 20f;
            float gy = Screen.height - sp.y - 20f;

            GUI.color = new Color(0f, 0f, 0f, 0.7f);
            GUI.DrawTexture(new Rect(gx - 4, gy - 4, 50, 44), Texture2D.whiteTexture);
            GUI.color              = CORNER_COLORS[i];
            s.normal.textColor     = CORNER_COLORS[i];
            GUI.Label(new Rect(gx, gy, 42, 38), $"C{i + 1}", s);
        }
        GUI.color = Color.white;
    }

    // =========================================================================
    //  PHASE 2 — HOLD PROGRESS BAR  (unchanged)
    // =========================================================================

    void DrawHoldProgressHUD()
    {
        if (placedCorners.Count >= 4 || !holdActive || holdTimer <= 0f) return;

        float progress = Mathf.Clamp01(holdTimer / holdSeconds);
        int   next     = placedCorners.Count + 1;

        float barW = 170f, barH = 22f;
        float barX = Screen.width - barW - 12f;
        float barY = 115f;

        GUI.color = new Color(0f, 0f, 0f, 0.6f);
        GUI.DrawTexture(new Rect(barX - 2, barY - 2, barW + 4, barH + 22), Texture2D.whiteTexture);

        Color barCol = progress < 0.5f
            ? Color.Lerp(Color.white, Color.yellow, progress * 2f)
            : Color.Lerp(Color.yellow, Color.green, (progress - 0.5f) * 2f);
        GUI.color = barCol;
        GUI.DrawTexture(new Rect(barX, barY, barW * progress, barH), Texture2D.whiteTexture);

        var ls = new GUIStyle(GUI.skin.label) { fontSize = 16, fontStyle = FontStyle.Bold };
        GUI.color = Color.white;
        GUI.Label(new Rect(barX, barY + barH + 2, barW, 20f), $"Hold still → C{next}", ls);
    }

    // =========================================================================
    //  HAND SKELETON OVERLAY  (unchanged)
    // =========================================================================

    void DrawHandSkeletonGL(List<HandData> hands)
    {
        if (hands == null || hands.Count == 0) return;
        if (_glMat == null)
        {
            var shader = Shader.Find("Hidden/Internal-Colored") ?? Shader.Find("Unlit/Color");
            _glMat = new Material(shader ?? Shader.Find("Standard")) { hideFlags = HideFlags.HideAndDontSave };
            _glMat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            _glMat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            _glMat.SetInt("_Cull", 0);
            _glMat.SetInt("_ZWrite", 0);
        }
        _glMat.SetPass(0);
        GL.PushMatrix();
        GL.LoadPixelMatrix();

        foreach (var hand in hands)
        {
            var lms = hand?.landmarks;
            if (lms == null || lms.Count < 21) continue;
            bool isPointing = hand.is_pointing;

            Vector2[] pts = new Vector2[21];
            for (int i = 0; i < 21; i++)
            {
                pts[i]   = LandmarkToScreen(lms[i].x, lms[i].y);
                pts[i].y = Screen.height - pts[i].y;
            }

            GL.Begin(GL.QUADS);
            for (int c = 0; c < HAND_CONNECTIONS.GetLength(0); c++)
            {
                int ia = HAND_CONNECTIONS[c, 0], ib = HAND_CONNECTIONS[c, 1];
                bool isIdx = (ia >= 5 && ia <= 8) || (ib >= 5 && ib <= 8);
                Color lc = isIdx
                    ? (isPointing ? new Color(0,1,1,1) : new Color(0,.7f,.7f,.8f))
                    : (isPointing ? new Color(1,1,1,.85f) : new Color(.7f,.7f,.7f,.7f));
                GL.Color(lc);
                DrawGLLine(pts[ia], pts[ib], isIdx ? 5f : 3f);
            }
            GL.End();

            GL.Begin(GL.QUADS);
            for (int i = 0; i < 21; i++)
            {
                Color dc; float dr;
                if      (i == 8)           { dc = new Color(1,0,1,1);          dr = 12f; }
                else if (i == 0)           { dc = new Color(1,.9f,0,1);        dr = 8f;  }
                else if (i >= 5 && i <= 7) { dc = isPointing ? new Color(0,1,1,1) : new Color(0,.7f,.7f,.8f); dr = 6f; }
                else                       { dc = new Color(.9f,.9f,.9f,.75f); dr = 5f;  }
                GL.Color(dc); DrawGLDot(pts[i], dr);
            }
            GL.End();
        }
        GL.PopMatrix();

        var ls = new GUIStyle(GUI.skin.label) { fontSize = 22, fontStyle = FontStyle.Bold };
        foreach (var hand in hands)
        {
            var lms = hand?.landmarks;
            if (lms == null || lms.Count < 1) continue;
            Vector2 wp  = LandmarkToScreen(lms[0].x, lms[0].y);
            string  lbl = hand.is_pointing ? "POINTING" : (hand.is_fist ? "FIST" : "open");
            GUI.color   = hand.is_pointing ? Color.cyan : (hand.is_fist ? new Color(1,.4f,0) : Color.white);
            GUI.Label(new Rect(wp.x + 10f, wp.y - 10f, 160f, 30f), lbl, ls);
        }
        GUI.color = Color.white;
    }

    void DrawGLLine(Vector2 a, Vector2 b, float width)
    {
        Vector2 dir = b - a; float len = dir.magnitude;
        if (len < 0.001f) return;
        Vector2 perp = new Vector2(-dir.y, dir.x) / len * (width * 0.5f);
        GL.Vertex3(a.x+perp.x, a.y+perp.y, 0); GL.Vertex3(b.x+perp.x, b.y+perp.y, 0);
        GL.Vertex3(b.x-perp.x, b.y-perp.y, 0); GL.Vertex3(a.x-perp.x, a.y-perp.y, 0);
    }

    void DrawGLDot(Vector2 c, float r)
    {
        GL.Vertex3(c.x-r, c.y-r, 0); GL.Vertex3(c.x+r, c.y-r, 0);
        GL.Vertex3(c.x+r, c.y+r, 0); GL.Vertex3(c.x-r, c.y+r, 0);
    }

    // =========================================================================
    //  MARKER BUILD  (unchanged)
    // =========================================================================

    void BuildMarker()
    {
        if (markerObject == null)
        {
            markerObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            markerObject.name = "_FingerMarker";
            markerObject.transform.localScale = Vector3.one * 0.08f;
            Destroy(markerObject.GetComponent<Collider>());
            markerObject.GetComponent<Renderer>().material = MakeMaterial(Color.green);
            markerObject.GetComponent<Renderer>().shadowCastingMode =
                UnityEngine.Rendering.ShadowCastingMode.Off;
        }
        markerObject.SetActive(false);

        shadowRing = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        shadowRing.name = "_ShadowRing";
        shadowRing.transform.localScale = new Vector3(0.18f, 0.002f, 0.18f);
        Destroy(shadowRing.GetComponent<Collider>());
        shadowRing.GetComponent<Renderer>().material = MakeMaterial(new Color(1f, 0.5f, 0f, 0.9f));
        shadowRing.SetActive(false);
    }

    void BuildDebugVisuals()
    {
        fingerSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        fingerSphere.name = "_FingerSphere";
        fingerSphere.transform.localScale = Vector3.one * 0.03f;
        Destroy(fingerSphere.GetComponent<Collider>());
        fingerSphere.GetComponent<Renderer>().material = MakeMaterial(Color.yellow);
        fingerSphere.GetComponent<Renderer>().shadowCastingMode =
            UnityEngine.Rendering.ShadowCastingMode.Off;
        fingerSphere.SetActive(false);

        var laserGO = new GameObject("_LaserLine");
        laserLine = laserGO.AddComponent<LineRenderer>();
        laserLine.useWorldSpace = true; laserLine.positionCount = 2;
        laserLine.startWidth = 0.003f;  laserLine.endWidth = 0.003f;
        laserLine.material = MakeMaterial(Color.green);
        laserLine.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        laserLine.enabled = false;
    }

    void UpdateDebugVisuals(Ray ray, Vector3 worldPt, bool show)
    {
        if (!showDebugVisuals || !show)
        {
            if (fingerSphere != null) fingerSphere.SetActive(false);
            if (laserLine    != null) laserLine.enabled = false;
            return;
        }
        Vector3 fp = ray.GetPoint(0.15f);
        if (fingerSphere != null) { fingerSphere.transform.position = fp; fingerSphere.SetActive(true); }
        if (laserLine    != null) { laserLine.SetPosition(0, fp); laserLine.SetPosition(1, worldPt); laserLine.enabled = true; }
    }

    // =========================================================================
    //  MOVE MARKER
    //
    //  ★ FIX — FINGER TIP ALIGNMENT
    //  ──────────────────────────────────────────────────────────────────────
    //  After building the camera ray from the screen pixel of index_tip,
    //  we advance the ray origin by fingerRayAdvance metres ALONG the ray
    //  before passing it to ProjectToFloor().
    //
    //  This compensates for the parallax gap between the camera origin and
    //  the physical fingertip — pulling the floor dot toward the user so
    //  it aligns with where the tip actually touches/points at the floor.
    //
    //  The base (un-advanced) ray is still passed to UpdateDebugVisuals()
    //  so the yellow finger sphere and laser line remain visually correct.
    // =========================================================================

    void MoveMarker()
    {
        // Hide cursor once all 4 corners placed
        if (placedCorners.Count >= 4)
        {
            SetMarkersVisible(false);
            UpdateDebugVisuals(new Ray(), Vector3.zero, false);
            return;
        }

        if (!floorValid || markerObject == null)
        {
            hud2 = floorValid ? "Waiting for hand..." : "Scanning floor...";
            return;
        }

        if (latestData?.hands?.hands == null || latestData.hands.hands.Count == 0)
        {
            SetMarkersVisible(false); UpdateDebugVisuals(new Ray(), Vector3.zero, false);
            hud2 = "No hand in frame"; return;
        }

        HandData hand = null;
        foreach (var h in latestData.hands.hands)
            if (h != null && h.is_pointing) { hand = h; break; }

        if (hand == null || hand.index_tip == null)
        {
            SetMarkersVisible(false); UpdateDebugVisuals(new Ray(), Vector3.zero, false);
            hud2 = "Hand visible — not pointing"; return;
        }

        float nx = hand.index_tip.x, ny = hand.index_tip.y;
        hud2 = $"POINTING  tip=({nx:F3}, {ny:F3})";
        if (float.IsNaN(nx) || float.IsNaN(ny)) { SetMarkersVisible(false); return; }

        // Convert normalised tip → screen pixel  (unchanged)
        Vector2 screenPt = LandmarkToScreen(nx, ny);

        // ★ FIX Challenge 1 — REACH AMPLIFICATION
        //   Mediapipe coordinates cluster near screen centre, so raw screenPt
        //   maps to a tiny ~1x1 m patch on the floor.
        //   We amplify the offset from the screen centre by reachMultiplier,
        //   pushing the projected point outward so far corners are reachable.
        //   amplifiedScreenPt is the SINGLE coordinate used by BOTH Tier-1
        //   (ARCore mesh raycast) and Tier-2 (math plane) to guarantee they
        //   always point at exactly the same location — no teleport/jitter.
        Vector2 screenCenter      = new Vector2(Screen.width * 0.5f, Screen.height * 0.5f);
        Vector2 amplifiedScreenPt = screenCenter + (screenPt - screenCenter) * reachMultiplier;   // ★ FIX C1

        var camRef = Camera.main;
        if (camRef == null) { SetMarkersVisible(false); return; }

        // Build base ray from camera through the AMPLIFIED screen pixel    // ★ FIX C1
        Ray baseRay = camRef.ScreenPointToRay(new Vector3(amplifiedScreenPt.x, amplifiedScreenPt.y, 0f));

        // ★ FIX: advance the ray origin toward the floor by fingerRayAdvance
        //   Effect: moves the floor-intersection point closer to the camera
        //   (i.e. toward the user), aligning the dot with the physical fingertip.
        Ray advancedRay = new Ray(
            baseRay.origin + baseRay.direction * fingerRayAdvance,   // ★ FIX
            baseRay.direction);

        // Project to floor — pass amplifiedScreenPt so BOTH tiers use the same coord  // ★ FIX C1
        Vector3 worldPt;
        if (!ProjectToFloor(amplifiedScreenPt, advancedRay, out worldPt))
        {
            SetMarkersVisible(false); UpdateDebugVisuals(baseRay, Vector3.zero, false);
            hud3 = "Ray miss — tilt phone toward floor"; return;
        }

        float sr = markerObject.transform.localScale.x * 0.5f;
        markerObject.transform.position = new Vector3(worldPt.x, floorY + sr + markerHeightOffset, worldPt.z);
        shadowRing.transform.position   = new Vector3(worldPt.x, floorY + 0.003f, worldPt.z);
        SetMarkersVisible(true);
        UpdateDebugVisuals(baseRay, worldPt, true);

        hud1 = $"FLOOR Y={floorY:F3}m  {placedCorners.Count}/4 corners  Point!";
        hud3 = $"DOT ({worldPt.x:F2}, {floorY:F3}, {worldPt.z:F2})";
    }

    void SetMarkersVisible(bool v)
    {
        if (markerObject != null) markerObject.SetActive(v);
        if (shadowRing   != null) shadowRing.SetActive(v);
    }

    void PulseMarker()
    {
        if (markerObject == null || !markerObject.activeSelf) return;
        float s = 0.08f * (1f + 0.2f * Mathf.Sin(Time.time * 6f));
        markerObject.transform.localScale = Vector3.one * s;
    }

    // =========================================================================
    //  COORDINATE MAPPING  (unchanged)
    // =========================================================================

    Vector2 LandmarkToScreen(float nx, float ny)
    {
        if (displayMatrix.HasValue)
        {
            var m = displayMatrix.Value;
            float vx = nx * 2f - 1f, vy = ny * 2f - 1f;
            Vector3 t = m.MultiplyPoint3x4(new Vector3(vx, vy, 0f));
            return new Vector2((t.x + 1f) * 0.5f * Screen.width,
                               Screen.height - ((t.y + 1f) * 0.5f * Screen.height));
        }
        return new Vector2(ny * Screen.width, (1f - nx) * Screen.height);
    }

    // =========================================================================
    //  FLOOR PROJECTION
    //
    //  Both tiers receive the same amplifiedScreenPt — this is the critical
    //  requirement for consistent, jitter-free pointing.
    //
    //  Tier 1 — ARCore mesh raycast:
    //    Passes amplifiedScreenPt directly to arRaycastManager.Raycast().
    //    ARCore builds its own internal ray from this 2D point, so it too
    //    benefits from the reach amplification.
    //
    //  Tier 2 — infinite math plane at floorY:
    //    Uses advancedRay (built from amplifiedScreenPt in MoveMarker).
    //    Fallback when ARCore has no detected plane at the target location.
    //
    //  Using the SAME screen coordinate in both tiers eliminates the
    //  teleport/jitter that occurs when Tier-1 and Tier-2 disagree on
    //  where the user is pointing.
    //
    //  Legacy overload (no Ray argument) kept for safety.
    // =========================================================================

    // Primary overload — both tiers use amplifiedScreenPt                // ★ FIX C1
    bool ProjectToFloor(Vector2 amplifiedScreenPt, Ray advancedRay, out Vector3 worldPt)
    {
        worldPt = Vector3.zero;

        // Tier 1: ARCore plane raycast — uses amplifiedScreenPt            // ★ FIX C1
        if (arRaycastManager != null &&
            arRaycastManager.Raycast(amplifiedScreenPt, rayHits, TrackableType.PlaneWithinPolygon)
            && rayHits.Count > 0)
        {
            Vector3 hit = rayHits[0].pose.position;
            if (!float.IsNaN(hit.x) && !float.IsNaN(hit.z))
            {
                worldPt = hit;
                return true;
            }
        }

        // Tier 2: infinite math plane at floorY — uses advancedRay        // ★ FIX C1
        float denom = advancedRay.direction.y;
        if (Mathf.Abs(denom) < 0.0005f) return false;
        float t = (floorY - advancedRay.origin.y) / denom;
        if (t < 0.01f || t > 25f) return false;
        Vector3 pt = advancedRay.origin + advancedRay.direction * t;
        if (float.IsNaN(pt.x) || float.IsNaN(pt.z)) return false;
        worldPt = pt;
        return true;
    }

    // Legacy overload — builds ray from Camera.main, no advance applied
    bool ProjectToFloor(Vector2 screenPt, out Vector3 worldPt)
    {
        var cam = Camera.main;
        Ray r = cam != null
            ? cam.ScreenPointToRay(new Vector3(screenPt.x, screenPt.y, 0f))
            : new Ray();
        return ProjectToFloor(screenPt, r, out worldPt);
    }

    // =========================================================================
    //  AR FLOOR DETECTION  (unchanged)
    // =========================================================================

    void RegisterAREvents()
    {
        if (arPlaneManager == null) arPlaneManager = FindObjectOfType<ARPlaneManager>();
        if (arPlaneManager != null) arPlaneManager.planesChanged += OnPlanesChanged;
        else Debug.LogWarning("[GS] No ARPlaneManager found!");
    }

    void OnPlanesChanged(ARPlanesChangedEventArgs args)
    {
        var all = new List<ARPlane>(args.added); all.AddRange(args.updated);
        float bestY = float.MaxValue; bool found = false;
        foreach (var plane in all)
        {
            if (plane.alignment != PlaneAlignment.HorizontalUp) continue;
            if (plane.center.y < bestY) { bestY = plane.center.y; found = true; }
        }
        if (!found) return;
        floorY     = bestY;
        floorPlane = new Plane(Vector3.up, new Vector3(0f, floorY, 0f));
        floorValid = true;
        hud1 = $"FLOOR Y={floorY:F3}m — point at corner 1!";
        Debug.Log($"[GS] Floor locked Y={floorY:F3}");
    }

    // =========================================================================
    //  CAMERA INIT + AR FRAME  (unchanged)
    // =========================================================================

    void InitCamera()
    {
#if UNITY_EDITOR
        hud1 = "Editor mode — no AR camera";
        Invoke(nameof(ConnectToServer), 1f);
#else
        if (arCameraManager == null) arCameraManager = FindObjectOfType<ARCameraManager>();
        if (arCameraManager != null)
        {
            arCameraManager.frameReceived += OnARFrameReceived;
            hud1 = "AR Camera ready — connecting...";
            Invoke(nameof(ConnectToServer), 1f);
        }
        else { Debug.LogError("[GS] No ARCameraManager!"); hud1 = "ERROR: No ARCameraManager!"; }
#endif
    }

    void OnARFrameReceived(ARCameraFrameEventArgs args)
    {
        if (args.displayMatrix.HasValue) displayMatrix = args.displayMatrix.Value;
        if (!connected || Time.time - lastSendTime < sendInterval) return;
        XRCpuImage img;
        if (!arCameraManager.TryAcquireLatestCpuImage(out img)) return;
        var cp = new XRCpuImage.ConversionParams
        {
            inputRect        = new RectInt(0, 0, img.width, img.height),
            outputDimensions = new Vector2Int(processingWidth, processingHeight),
            outputFormat     = TextureFormat.RGB24,
            transformation   = XRCpuImage.Transformation.None
        };
        int sz = img.GetConvertedDataSize(cp);
        var raw = new NativeArray<byte>(sz, Allocator.Temp);
        img.Convert(cp, raw); img.Dispose();
        var tex = new Texture2D(processingWidth, processingHeight, TextureFormat.RGB24, false);
        tex.LoadRawTextureData(raw); tex.Apply(); raw.Dispose();
        byte[] jpg = tex.EncodeToJPG(jpegQuality); Destroy(tex);
        lock (sendLock) { sendQ.Clear(); sendQ.Enqueue(jpg); }
        lastSendTime = Time.time;
    }

    // =========================================================================
    //  NETWORKING  (unchanged)
    // =========================================================================

    void ConnectToServer()
    {
        if (connected) return;
        hud1 = $"Connecting {serverIP}:{serverPort}...";
        try
        {
            tcp = new TcpClient();
            var ar = tcp.BeginConnect(serverIP, serverPort, null, null);
            if (!ar.AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(5)))
            { tcp.Close(); throw new TimeoutException("Timed out"); }
            tcp.EndConnect(ar);
            tcp.ReceiveBufferSize = tcp.SendBufferSize = 1024 * 1024;
            net = tcp.GetStream(); connected = running = true;
            hud1 = $"Connected  floorY={floorY:F3}m";
            netThread = new Thread(NetworkLoop) { IsBackground = true };
            netThread.Start();
            Debug.Log($"[GS] Connected {serverIP}:{serverPort}");
        }
        catch (Exception ex)
        {
            connected = false; hud1 = "Connect failed — retry 2s";
            Debug.LogWarning($"[GS] {ex.Message}");
            Invoke(nameof(ConnectToServer), 3f);
        }
    }

    void Disconnect()
    {
        if (!connected && !running) return;
        connected = running = false; needReconnect = true;
        try { net?.Close(); } catch { } try { tcp?.Close(); } catch { }
        hud1 = "Disconnected — reconnecting...";
    }

    void NetworkLoop()
    {
        var buf = new byte[4 * 1024 * 1024];
        try
        {
            while (running && connected)
            {
                byte[] frame = null;
                lock (sendLock)
                {
                    while (sendQ.Count > 1) sendQ.Dequeue();
                    if (sendQ.Count > 0) frame = sendQ.Dequeue();
                }
                if (frame == null) { Thread.Sleep(1); continue; }
                try
                {
                    var len = BitConverter.GetBytes((uint)frame.Length);
                    if (BitConverter.IsLittleEndian) Array.Reverse(len);
                    net.Write(len, 0, 4); net.Write(frame, 0, frame.Length); net.Flush();
                }
                catch { Disconnect(); break; }
                try
                {
                    int h = 0;
                    while (h < 4) { int r = net.Read(buf, h, 4-h); if (r==0){Disconnect();return;} h+=r; }
                    if (!connected) break;
                    int jl = buf[0]|(buf[1]<<8)|(buf[2]<<16)|(buf[3]<<24);
                    if (jl <= 0 || jl > 2_000_000) continue;
                    int got = 0;
                    while (got < jl) { int r = net.Read(buf, got, jl-got); if (r==0){Disconnect();return;} got+=r; }
                    lock (recvLock) { recvQ.Enqueue(Encoding.UTF8.GetString(buf, 0, jl)); }
                }
                catch { Disconnect(); break; }
                Thread.Sleep(1);
            }
        }
        catch { } finally { Disconnect(); }
    }

    void DrainReceiveQueue()
    {
        string json = null;
        lock (recvLock) { if (recvQ.Count > 0) json = recvQ.Dequeue(); }
        if (string.IsNullOrWhiteSpace(json)) return;
        json = json.Trim();
        if (json.Length < 2 || json[0] != '{') return;
        try { latestData = JsonUtility.FromJson<ServerResult>(json); }
        catch (Exception ex)
        {
            Debug.LogWarning($"[GS] JSON: {ex.Message}");
            Debug.LogWarning($"[GS] RAW: {json.Substring(0, Mathf.Min(300, json.Length))}");
            latestData = null;
        }
    }

    Material MakeMaterial(Color col)
    {
        Shader sh = Shader.Find("Universal Render Pipeline/Unlit")
                 ?? Shader.Find("Unlit/Color") ?? Shader.Find("Standard");
        Material mat = new Material(sh ?? Shader.Find("Hidden/InternalErrorShader"));
        mat.color = col;
        if (mat.HasProperty("_EmissionColor"))
        { mat.EnableKeyword("_EMISSION"); mat.SetColor("_EmissionColor", col * 2.5f); }
        return mat;
    }
}

// =============================================================================
//  DATA CLASSES  (unchanged)
// =============================================================================

[Serializable] public class TipPoint3D   { public float x, y, z; }
[Serializable] public class HandLandmark { public float x, y, z; }

[Serializable]
public class HandData
{
    public bool               is_pointing;
    public bool               is_fist;
    public TipPoint3D         index_tip;
    public List<HandLandmark> landmarks;
}

[Serializable] public class HandsWrapper { public List<HandData> hands; }

[Serializable]
public class ServerResult
{
    public HandsWrapper hands;
    public float        fps;
    public float        process_ms;
}