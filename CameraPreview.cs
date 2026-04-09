// ============================================================================
// CameraPreview.cs  —  v8  Data classes + ARCore Health Check helper
// ============================================================================
// CHANGES FROM previous version:
//   1. PoseData + PoseLandmark classes ADDED (missing in uploaded v6)
//      — These are required for RenderPose() in GuardianSystem.cs
//   2. ARCoreHealthCheck static class added
//      — Call ARCoreHealthCheck.RunAll() from GuardianSystem.Start()
//        to log all ARCore requirements to Unity Console at launch
//   3. ServerResult.pose field added to match server.py v8 JSON output
//
// JSON contract with Python server (GuardianSystem v22 fixed-rectangle):
//   Required for hand-driven UX: "hands" with at least one entry containing
//   is_pointing, is_fist, index_tip {x,y,z}, and optionally landmarks[21].
//   Extra keys (floor, pose, play_area, etc.) are ignored if present.
//
// Device debug output (written by GuardianSystem):
//   persistentDataPath/last_server_json.txt — full raw server JSON each frame
//   persistentDataPath/guardian_boundary_state.json — saved rectangle (see BoundaryPersistData)
// ============================================================================

using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

// ── JSON Data Classes ──────────────────────────────────────────────────────
// [Serializable]
// public class ServerResult
// {
//     public int          frame;
//     public float        fps;
//     public float        process_ms;
//     public FloorData    floor;
//     public HandsData    hands;
//     public PoseData     pose;        // ADDED v8 — matches server 'pose' key
//     public PlayAreaData play_area;
// }

[Serializable]
public class FloorData
{
    public bool    detected;
    public float[] plane;        // [a, b, c, d] RANSAC plane equation
    public int     confidence;   // 0-100
}

[Serializable]
public class HandsData
{
    public bool           detected;
    public List<HandInfo> hands;
}

[Serializable]
public class HandInfo
{
    public int               id;
    public bool              is_pointing;
    public bool              is_fist;         // STOP drawing gesture (all fingers curled)
    // point_conf is sent by Python but Unity JsonUtility silently ignores
    // unknown JSON fields — this is safe, no action needed.
    public Vector3Data       index_tip;
    public List<Vector3Data> landmarks;   // 21 MediaPipe Hand landmarks
}

// ADDED v8 — Pose estimation (33 MediaPipe Pose landmarks)
[Serializable]
public class PoseData
{
    public bool               detected;
    public List<PoseLandmark> landmarks;
}

[Serializable]
public class PoseLandmark
{
    public float x, y, z;
    public float visibility;   // 0.0-1.0 — skip landmark if < 0.5
}

[Serializable]
public class Vector3Data
{
    public float x, y, z;
}

[Serializable]
public class PlayAreaData
{
    public List<Vector3Data> points;
    public int               num_points;
    public bool              is_complete;
}


// ── ARCore Health Check ────────────────────────────────────────────────────
// Place this static class call in GuardianSystem.Start():
//   ARCoreHealthCheck.RunAll();
// It logs every requirement to the Unity Console — green = ok, red = issue.
//
public static class ARCoreHealthCheck
{
    public static void RunAll()
    {
        Debug.Log("=== ARCore Health Check ===");

        // 1. AR Foundation packages
        CheckARFoundation();

        // 2. Scene components
        CheckSceneComponents();

        // 3. Camera settings
        CheckCameraSettings();

        // 4. Build settings (runtime warning only)
        CheckBuildSettings();

        Debug.Log("=== ARCore Health Check Complete ===");
    }

    static void CheckARFoundation()
    {
        // FIXED: UNITY_XR_ARFOUNDATION / UNITY_XR_ARCORE are NOT real Unity
        // scripting defines. The old #if checks ALWAYS fired the LogError
        // even when AR Foundation IS installed. Using runtime reflection now.

        bool arFoundationPresent = false;
        bool arCorePresent       = false;

        try
        {
            var t = typeof(UnityEngine.XR.ARFoundation.ARSession);
            arFoundationPresent = (t != null);
        }
        catch { }

        try
        {
            var t = System.Type.GetType(
                "UnityEngine.XR.ARCore.ARCoreSessionSubsystem, Unity.XR.ARCore",
                throwOnError: false);
            arCorePresent = (t != null);
        }
        catch { }

        if (arFoundationPresent)
            Debug.Log("[ARCore] ✓ AR Foundation present (runtime check)");
        else
            Debug.LogError("[ARCore] ✗ AR Foundation NOT detected. " +
                           "Install via Window → Package Manager → AR Foundation");

        if (arCorePresent)
            Debug.Log("[ARCore] ✓ ARCore XR Plugin present (runtime check)");
        else
            Debug.LogWarning("[ARCore] ⚠ ARCore XR Plugin assembly not found. " +
                             "Normal in Editor. On device: Project Settings → " +
                             "XR Plug-in Management → Android → ARCore ✓");
    }

    static void CheckSceneComponents()
    {
        // AR Session
        var arSession = GameObject.FindObjectOfType<ARSession>();
        if (arSession == null)
            Debug.LogError("[ARCore] ✗ No ARSession in scene! " +
                           "Add via GameObject → XR → AR Session");
        else
            Debug.Log($"[ARCore] ✓ ARSession: '{arSession.name}'");

        // ARPlaneManager
        var planeMgr = GameObject.FindObjectOfType<ARPlaneManager>();
        if (planeMgr == null)
            Debug.LogError("[ARCore] ✗ No ARPlaneManager in scene! " +
                           "Add to XR Origin. Floor boundary will not work.");
        else
        {
            Debug.Log($"[ARCore] ✓ ARPlaneManager: '{planeMgr.name}'");
            if (planeMgr.requestedDetectionMode != PlaneDetectionMode.Horizontal)
                Debug.LogWarning("[ARCore] ⚠ ARPlaneManager Detection Mode is not Horizontal. " +
                                 "Change to Horizontal to detect floor only.");
            else
                Debug.Log("[ARCore] ✓ ARPlaneManager Detection Mode = Horizontal");
        }

        // ARRaycastManager
        var raycastMgr = GameObject.FindObjectOfType<ARRaycastManager>();
        if (raycastMgr == null)
            Debug.LogError("[ARCore] ✗ No ARRaycastManager in scene! " +
                           "Add to XR Origin. Boundary placement will not work.");
        else
            Debug.Log($"[ARCore] ✓ ARRaycastManager: '{raycastMgr.name}'");

        // ARCameraManager
        var camMgr = GameObject.FindObjectOfType<ARCameraManager>();
        if (camMgr == null)
            Debug.LogError("[ARCore] ✗ No ARCameraManager in scene! " +
                           "Add to the AR Camera (child of XR Origin).");
        else
            Debug.Log($"[ARCore] ✓ ARCameraManager: '{camMgr.name}'");

        // ARCameraBackground
        var camBg = GameObject.FindObjectOfType<ARCameraBackground>();
        if (camBg == null)
            Debug.LogWarning("[ARCore] ⚠ No ARCameraBackground. " +
                             "Camera feed won't show on device without it.");
        else
            Debug.Log($"[ARCore] ✓ ARCameraBackground: '{camBg.name}'");

        // AudioListener count
        var listeners = GameObject.FindObjectsOfType<AudioListener>();
        if (listeners.Length > 1)
            Debug.LogError($"[ARCore] ✗ {listeners.Length} AudioListeners found! " +
                           "Should be exactly 1. Remove AudioListener from old Camera.");
        else if (listeners.Length == 1)
            Debug.Log($"[ARCore] ✓ AudioListener count = 1 ('{listeners[0].name}')");
        else
            Debug.LogWarning("[ARCore] ⚠ No AudioListener in scene.");
    }

    static void CheckCameraSettings()
    {
        var cam = Camera.main;
        if (cam == null)
        {
            Debug.LogError("[ARCore] ✗ Camera.main is null! " +
                           "Tag your AR Camera as 'MainCamera'.");
            return;
        }
        Debug.Log($"[ARCore] ✓ Camera.main = '{cam.name}'");

        if (cam.clearFlags != CameraClearFlags.SolidColor)
            Debug.LogWarning("[ARCore] ⚠ Camera.clearFlags should be SolidColor, " +
                             $"currently: {cam.clearFlags}. " +
                             "AR Camera Background needs black clear.");
        else
            Debug.Log("[ARCore] ✓ Camera.clearFlags = SolidColor");
    }

    static void CheckBuildSettings()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        Debug.Log("[ARCore] ✓ Running on Android device");
#elif UNITY_EDITOR
        Debug.Log("[ARCore] ℹ Running in Editor — AR features use WebCam fallback");
        Debug.Log("[ARCore] ℹ Remember: Build Settings → Scripting Backend = IL2CPP, Architecture = ARM64");
#else
        Debug.LogWarning("[ARCore] ⚠ Not Android — ARCore only works on Android 7.0+ phones");
#endif
    }
}