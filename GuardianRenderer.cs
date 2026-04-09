// ============================================================================
// GuardianRenderer.cs  —  DISABLED (kept for compile compatibility)
// ============================================================================
// WHY THIS IS DISABLED:
//
// GuardianRenderer was trying to render hands and boundary using:
//   • 3D Sphere GameObjects placed with arCamera.ViewportToWorldPoint()
//   • A LineRenderer in world space
//
// Neither of these is visible in a 2D AR app where the "AR view" is a
// RawImage on a Canvas (Screen Space Overlay). The 3D spheres render behind
// the Canvas, and the LineRenderer renders in 3D world space where there is
// nothing to see (the camera shows only the Canvas, not the 3D scene).
//
// GuardianSystem.cs ALREADY contains the correct 2D Canvas-based rendering:
//   • 42 UI Image "joints" on the AR overlay Canvas  (DrawHands)
//   • 46 UI Image "bones" for the skeleton lines      (DrawHands)
//   • Green UI Image dots + lines for boundary        (DrawBoundary)
//
// This script is intentionally left as an empty stub so that any
// GuardianRenderer component already in the scene compiles without error.
// Simply REMOVE the GuardianRenderer component from your scene objects —
// or leave it and it will do nothing.
// ============================================================================

using UnityEngine;

public class GuardianRenderer : MonoBehaviour
{
    // All visualization is now handled inside GuardianSystem.cs.
    // This component does nothing and can be safely removed from the scene.
    void Start()
    {
        Debug.Log("[GuardianRenderer] DISABLED — visualization is handled by GuardianSystem.cs");
        enabled = false;
    }
}