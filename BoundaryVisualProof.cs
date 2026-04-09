// =============================================================================
// BoundaryVisualProof.cs — Visual proof that BoundaryLogic 3D math is correct
// =============================================================================
// USAGE:
//   1. In Unity Editor, create an Empty GameObject
//   2. Drag this script onto it
//   3. Open the Scene view (not Game view)
//   4. You will immediately see a glowing 3D room boundary drawn on the floor!
//
// This script uses the EXACT SAME BoundaryLogic code that passed all 96 tests.
// What you see in the Scene view is what you'll get on the phone.
// =============================================================================

using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

[ExecuteInEditMode]
public class BoundaryVisualProof : MonoBehaviour
{
    [Header("Simulated Camera")]
    [Tooltip("Height of phone camera above floor (metres)")]
    public float cameraHeight = 1.5f;

    [Tooltip("Phone tilt angle downward (degrees)")]
    [Range(20f, 70f)]
    public float cameraTiltDown = 45f;

    [Header("Simulated Room")]
    [Tooltip("How many points per side of the room")]
    [Range(4, 20)]
    public int pointsPerSide = 8;

    [Header("Visual")]
    public Color lineColor = Color.green;
    public Color fillColor = new Color(0f, 1f, 0.3f, 0.3f);
    public Color cornerColor = Color.yellow;
    public float wallHeight = 0.3f;

    // Internal
    BoundaryLogic.PointAccumulator acc;
    List<Vector3> cachedPoints;
    int[] cachedTris;
    const int SW = 1080, SH = 1920;

    // ── The EXACT projection pipeline from the tests ─────────────────────

    Vector3 SimulateFingerToWorld(float nx, float ny, float mpZ)
    {
        Vector2 sp = BoundaryLogic.LandmarkToScreen(nx, ny, SW, SH);
        float halfFov = 30f * Mathf.Deg2Rad;
        float halfW = SW * 0.5f, halfH = SH * 0.5f;
        float rx = (sp.x - halfW) / halfH * Mathf.Tan(halfFov);
        float ry = (sp.y - halfH) / halfH * Mathf.Tan(halfFov);
        Vector3 localDir = new Vector3(rx, ry, 1f).normalized;
        Vector3 dir = Quaternion.Euler(cameraTiltDown, 0f, 0f) * localDir;
        Vector3 orig = new Vector3(0f, cameraHeight, 0f);
        Vector3 hit;
        if (BoundaryLogic.RayFloorIntersection(orig, dir, 0f, out hit)) return hit;
        float depth = BoundaryLogic.MediaPipeZToDepth(mpZ);
        return orig + dir * depth;
    }

    // ── Build the simulated room boundary ────────────────────────────────

    void RebuildBoundary()
    {
        acc = new BoundaryLogic.PointAccumulator
        {
            MinPointDistance = 0.04f,
            LineHeightOffset = 0.04f
        };

        var path = new List<float[]>();
        int n = pointsPerSide;

        // Trace a rectangle: finger moves around screen edges pointing at floor
        // Side 1: left→right along bottom of screen (far wall)
        for (int i = 0; i <= n; i++)
            path.Add(new float[] { Mathf.Lerp(0.2f, 0.8f, (float)i / n), 0.75f, 0.15f });
        // Side 2: bottom→middle on right (right wall, getting closer)
        for (int i = 1; i <= n; i++)
            path.Add(new float[] { 0.8f, Mathf.Lerp(0.75f, 0.55f, (float)i / n), Mathf.Lerp(0.15f, 0.08f, (float)i / n) });
        // Side 3: right→left along middle of screen (near wall)
        for (int i = 1; i <= n; i++)
            path.Add(new float[] { Mathf.Lerp(0.8f, 0.2f, (float)i / n), 0.55f, 0.08f });
        // Side 4: middle→bottom on left (left wall, going back)
        for (int i = 1; i <= n; i++)
            path.Add(new float[] { 0.2f, Mathf.Lerp(0.55f, 0.75f, (float)i / n), Mathf.Lerp(0.08f, 0.15f, (float)i / n) });

        // Feed through the EXACT same pipeline
        var pyPts = new List<Vector3Data>();
        foreach (var fp in path)
            pyPts.Add(new Vector3Data { x = fp[0], y = fp[1], z = fp[2] });

        BoundaryLogic.ProcessNewPoints(pyPts, 0, acc, SimulateFingerToWorld);
        acc.MarkComplete();

        cachedPoints = new List<Vector3>(acc.Points);
        if (cachedPoints.Count >= 3)
            cachedTris = BoundaryLogic.EarClip(cachedPoints.ToArray(), cachedPoints.Count);
        else
            cachedTris = new int[0];
    }

    // ── Draw it in the Scene view ────────────────────────────────────────

#if UNITY_EDITOR
    void OnDrawGizmos()
    {
        // Rebuild each frame so inspector changes are live
        RebuildBoundary();
        if (cachedPoints == null || cachedPoints.Count < 3) return;

        int count = cachedPoints.Count;

        // ── Floor outline (thick green line) ──
        Gizmos.color = lineColor;
        for (int i = 0; i < count; i++)
        {
            Vector3 a = cachedPoints[i];
            Vector3 b = cachedPoints[(i + 1) % count];
            Gizmos.DrawLine(a, b);
            // Draw thicker by offset lines
            Vector3 offset = Vector3.up * 0.005f;
            Gizmos.DrawLine(a + offset, b + offset);
            Gizmos.DrawLine(a - offset, b - offset);
        }

        // ── Vertical walls ──
        Gizmos.color = new Color(lineColor.r, lineColor.g, lineColor.b, 0.5f);
        for (int i = 0; i < count; i++)
        {
            Vector3 bottom = cachedPoints[i];
            Vector3 top = bottom + Vector3.up * wallHeight;
            Gizmos.DrawLine(bottom, top);
            Vector3 nextBottom = cachedPoints[(i + 1) % count];
            Vector3 nextTop = nextBottom + Vector3.up * wallHeight;
            Gizmos.DrawLine(top, nextTop);
        }

        // ── Corner posts (yellow spheres) ──
        Gizmos.color = cornerColor;
        for (int i = 0; i < count; i++)
        {
            Gizmos.DrawSphere(cachedPoints[i], 0.06f);
            Gizmos.DrawSphere(cachedPoints[i] + Vector3.up * wallHeight, 0.04f);
        }

        // ── Floor fill (triangulated mesh drawn as green triangles) ──
        Gizmos.color = fillColor;
        if (cachedTris != null && cachedTris.Length >= 3)
        {
            for (int i = 0; i < cachedTris.Length; i += 3)
            {
                Vector3 v0 = cachedPoints[cachedTris[i]];
                Vector3 v1 = cachedPoints[cachedTris[i + 1]];
                Vector3 v2 = cachedPoints[cachedTris[i + 2]];
                Gizmos.DrawLine(v0, v1);
                Gizmos.DrawLine(v1, v2);
                Gizmos.DrawLine(v2, v0);
            }
        }

        // ── Camera position indicator ──
        Gizmos.color = Color.cyan;
        Vector3 camPos = new Vector3(0f, cameraHeight, 0f);
        Gizmos.DrawWireSphere(camPos, 0.1f);
        Vector3 lookDir = Quaternion.Euler(cameraTiltDown, 0f, 0f) * Vector3.forward;
        Gizmos.DrawRay(camPos, lookDir * 1.5f);

        // ── Label with stats ──
        float area = Mathf.Abs(BoundaryLogic.ComputeSignedAreaXZ(cachedPoints));
        float arc = BoundaryLogic.ComputeArcLength(cachedPoints);
        float winding = BoundaryLogic.ComputeWindingConsistency(cachedPoints);
        Handles.Label(camPos + Vector3.up * 0.3f,
            $"BOUNDARY PROOF\n" +
            $"Points: {count}\n" +
            $"Area: {area:F2} m²\n" +
            $"Perimeter: {arc:F2} m\n" +
            $"Winding: {winding:F0}%\n" +
            $"Triangles: {cachedTris.Length / 3}",
            new GUIStyle
            {
                fontSize = 14,
                fontStyle = FontStyle.Bold,
                normal = new GUIStyleState { textColor = Color.white }
            });
    }
#endif
}
