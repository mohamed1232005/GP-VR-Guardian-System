// =============================================================================
// BoundaryLogic.cs — Pure-logic boundary helpers (unit-testable, no MonoBehaviour)
// =============================================================================
// Extracts testable boundary logic from GuardianSystem.cs so NUnit tests can
// validate coordinate transforms, point accumulation, distance gating,
// ear-clip triangulation, and spatial-path validation WITHOUT needing AR/camera.
// =============================================================================

using System.Collections.Generic;
using UnityEngine;

// NOTE: Vector3Data is already defined in CameraPreview.cs (class with x, y, z fields).
// BoundaryLogic uses that same type — do NOT redefine it here.

/// <summary>
/// Pure-logic boundary system. No MonoBehaviour, no AR, no UI — just math.
/// GuardianSystem delegates to these functions at runtime.
/// Tests call them directly.
/// </summary>
public static class BoundaryLogic
{
    // ── Coordinate conversion ────────────────────────────────────────────────

    /// <summary>
    /// Convert normalised landmark (0-1) to screen pixels.
    /// MediaPipe y=0 is top, Unity screen y=0 is bottom, so we flip Y.
    /// </summary>
    public static Vector2 LandmarkToScreen(float nx, float ny, int screenW, int screenH)
    {
        return new Vector2(nx * screenW, (1f - ny) * screenH);
    }

    // ── Distance gating ──────────────────────────────────────────────────────

    /// <summary>
    /// Returns true if <paramref name="candidate"/> is far enough from
    /// <paramref name="lastPt"/> to be accepted as a new boundary point.
    /// </summary>
    public static bool PassesDistanceGate(Vector3 candidate, Vector3 lastPt, float minDist)
    {
        return Vector3.Distance(candidate, lastPt) >= minDist;
    }

    // ── Height offset ────────────────────────────────────────────────────────

    /// <summary>
    /// Apply the line-height offset to lift boundary points above the floor
    /// surface to prevent z-fighting with the AR plane.
    /// </summary>
    public static Vector3 ApplyHeightOffset(Vector3 pt, float offset)
    {
        pt.y += offset;
        return pt;
    }

    // ── Tier-2 floor ray intersection ────────────────────────────────────────

    /// <summary>
    /// Given a ray from the camera, compute the intersection with the
    /// horizontal plane at y = <paramref name="floorY"/>.
    /// Returns <c>true</c> if a valid intersection exists (t in [0.1, 20]).
    /// </summary>
    public static bool RayFloorIntersection(
        Vector3 rayOrigin, Vector3 rayDir, float floorY,
        out Vector3 hitPt, float tMin = 0.1f, float tMax = 20f)
    {
        hitPt = Vector3.zero;
        if (Mathf.Abs(rayDir.y) < 1e-5f) return false;
        float t = (floorY - rayOrigin.y) / rayDir.y;
        if (t < tMin || t > tMax) return false;
        hitPt = rayOrigin + rayDir * t;
        return true;
    }

    // ── Tier-3 depth projection ──────────────────────────────────────────────

    /// <summary>
    /// Map MediaPipe hand-z (relative to wrist in palm-size units) to a
    /// real-world camera depth in metres.  Returns clamped value in [0.5, 4.0].
    /// </summary>
    public static float MediaPipeZToDepth(float mpZ)
    {
        return Mathf.Clamp(0.5f + mpZ * 4.0f, 0.5f, 4.0f);
    }

    // ── Point accumulator ────────────────────────────────────────────────────

    /// <summary>
    /// Stateful accumulator that mirrors MaybeAdd logic.
    /// Call <see cref="TryAdd"/> for each candidate world point.
    /// </summary>
    public class PointAccumulator
    {
        public readonly List<Vector3> Points = new List<Vector3>();
        public Vector3 LastPoint { get; private set; }
        public bool AreaComplete { get; private set; }

        public float MinPointDistance = 0.04f;
        public float LineHeightOffset = 0.04f;

        /// <summary>Try to add a world point. Returns true if accepted.</summary>
        public bool TryAdd(Vector3 wp)
        {
            if (AreaComplete) return false;
            // Compare XZ (floor-plane) distance only — Y has the height offset
            // which inflates 3D distance and would let too-close points pass.
            if (Points.Count > 0)
            {
                float dx = wp.x - LastPoint.x;
                float dz = wp.z - LastPoint.z;
                if (Mathf.Sqrt(dx * dx + dz * dz) < MinPointDistance) return false;
            }
            wp = ApplyHeightOffset(wp, LineHeightOffset);
            Points.Add(wp);
            LastPoint = wp;
            return true;
        }

        /// <summary>Mark the boundary as complete. Requires >= 3 points.</summary>
        public bool MarkComplete()
        {
            if (Points.Count < 3) return false;
            AreaComplete = true;
            return true;
        }

        /// <summary>Reset all state.</summary>
        public void Clear()
        {
            Points.Clear();
            LastPoint = Vector3.zero;
            AreaComplete = false;
        }
    }

    // ── ProcessPlayArea logic ────────────────────────────────────────────────

    /// <summary>
    /// Given a list of Python play_area points and a tracking index, compute
    /// how many new points were added. Returns the new tracking index.
    /// <paramref name="projector"/> converts each (nx, ny, z) → world Vector3.
    /// </summary>
    public delegate Vector3 PointProjector(float nx, float ny, float z);

    public static int ProcessNewPoints(
        List<Vector3Data> pyPoints, int lastProcessedCount,
        PointAccumulator acc, PointProjector projector)
    {
        if (pyPoints == null || acc.AreaComplete) return lastProcessedCount;
        for (int i = lastProcessedCount; i < pyPoints.Count; i++)
        {
            var p = pyPoints[i];
            Vector3 wp = projector(p.x, p.y, p.z);
            acc.TryAdd(wp);
        }
        return pyPoints.Count;
    }

    // ── Spatial path validation ──────────────────────────────────────────────

    /// <summary>
    /// Compute total arc length of a polyline.
    /// </summary>
    public static float ComputeArcLength(List<Vector3> pts)
    {
        float len = 0f;
        for (int i = 1; i < pts.Count; i++)
            len += Vector3.Distance(pts[i], pts[i - 1]);
        return len;
    }

    /// <summary>
    /// Compute the signed area on the XZ plane (positive = CCW, negative = CW).
    /// Uses the shoelace formula.
    /// </summary>
    public static float ComputeSignedAreaXZ(List<Vector3> pts)
    {
        float area = 0f;
        int n = pts.Count;
        for (int i = 0; i < n; i++)
        {
            var a = pts[i];
            var b = pts[(i + 1) % n];
            area += (b.x - a.x) * (b.z + a.z);
        }
        return area * 0.5f;
    }

    /// <summary>
    /// Check whether consecutive points form a roughly monotonic path
    /// (each step moves in a consistent rotational direction around the centroid).
    /// Returns the fraction of steps that agree with the dominant winding.
    /// 1.0 = perfect loop, 0.5 = random.
    /// </summary>
    public static float ComputeWindingConsistency(List<Vector3> pts)
    {
        if (pts.Count < 3) return 0f;

        // Centroid
        Vector3 c = Vector3.zero;
        foreach (var p in pts) c += p;
        c /= pts.Count;

        int cwCount = 0, ccwCount = 0;
        for (int i = 0; i < pts.Count; i++)
        {
            var a = pts[i] - c;
            var b = pts[(i + 1) % pts.Count] - c;
            float cross = a.x * b.z - a.z * b.x;
            if (cross > 0f) ccwCount++;
            else if (cross < 0f) cwCount++;
        }
        int total = cwCount + ccwCount;
        if (total == 0) return 0f;
        return Mathf.Max(cwCount, ccwCount) / (float)total;
    }

    // ── Ear-clip triangulation (same algorithm as GuardianSystem) ─────────────

    public static int[] EarClip(Vector3[] v, int n)
    {
        var idx = new List<int>(n);
        for (int i = 0; i < n; i++) idx.Add(i);
        var t = new List<int>();
        float area = 0;
        for (int i = 0; i < n; i++)
        {
            var a = v[idx[i]];
            var b = v[idx[(i + 1) % n]];
            area += (b.x - a.x) * (b.z + a.z);
        }
        bool cw = area > 0;
        int safe = n * n + 10;
        while (idx.Count > 3 && safe-- > 0)
        {
            bool found = false;
            for (int i = 0; i < idx.Count; i++)
            {
                int p = (i - 1 + idx.Count) % idx.Count, nx = (i + 1) % idx.Count;
                Vector3 A = v[idx[p]], B = v[idx[i]], C = v[idx[nx]];
                float cross = (B.x - A.x) * (C.z - A.z) - (B.z - A.z) * (C.x - A.x);
                if (cw ? cross > 0 : cross < 0) continue;
                bool ear = true;
                for (int k = 0; k < idx.Count; k++)
                {
                    if (k == p || k == i || k == nx) continue;
                    if (PointInTriangle(v[idx[k]], A, B, C)) { ear = false; break; }
                }
                if (!ear) continue;
                t.Add(idx[p]); t.Add(idx[i]); t.Add(idx[nx]);
                idx.RemoveAt(i); found = true; break;
            }
            if (!found) break;
        }
        if (idx.Count == 3) { t.Add(idx[0]); t.Add(idx[1]); t.Add(idx[2]); }
        return t.ToArray();
    }

    public static bool PointInTriangle(Vector3 P, Vector3 A, Vector3 B, Vector3 C)
    {
        float d1 = Sign(P, A, B), d2 = Sign(P, B, C), d3 = Sign(P, C, A);
        return !((d1 < 0 || d2 < 0 || d3 < 0) && (d1 > 0 || d2 > 0 || d3 > 0));
    }

    public static float Sign(Vector3 a, Vector3 b, Vector3 c)
    {
        return (a.x - c.x) * (b.z - c.z) - (b.x - c.x) * (a.z - c.z);
    }

    // ── Close-loop detection (Python-side mirror for validation) ──────────────

    /// <summary>
    /// Check if adding <paramref name="candidate"/> would close the loop
    /// (i.e. candidate is near the first point, after min points collected).
    /// Uses squared-distance on the XY (normalised image) plane.
    /// </summary>
    public static bool WouldCloseLoop(
        List<Vector3Data> points, Vector3Data candidate,
        int minPointsToClose, float closeLoopThreshSq)
    {
        if (points == null || points.Count < minPointsToClose) return false;
        var first = points[0];
        float dx = candidate.x - first.x;
        float dy = candidate.y - first.y;
        return (dx * dx + dy * dy) < closeLoopThreshSq;
    }
}
