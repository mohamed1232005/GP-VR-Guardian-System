"""Quick functional test for the world-space FloorAccumulator (Phase 14).
Run: venv/Scripts/python.exe _test_accum.py
"""
import numpy as np
from geometry.floor_accumulator import FloorAccumulator


def make_opencv_inliers_for_world(world_xz, floor_y, M):
    """Given desired WORLD floor points (x,z) at height floor_y and a pose M
    (Unity-cam -> world), produce the OpenCV-camera inlier points that the
    accumulator will transform back to exactly those world points."""
    W = np.column_stack([world_xz[:, 0],
                         np.full(len(world_xz), floor_y),
                         world_xz[:, 1],
                         np.ones(len(world_xz))])           # (N,4) world homog
    M_inv = np.linalg.inv(M)
    unity_cam = (M_inv @ W.T).T[:, :3]                       # world -> unity cam
    opencv = unity_cam.copy()
    opencv[:, 1] *= -1.0                                     # unity cam -> opencv cam (flip Y)
    return opencv


def make_opencv_inliers_for_world_pts(world_pts, M):
    """Like make_opencv_inliers_for_world but for arbitrary per-point world Y
    (used to build a TILTED floor)."""
    W = np.column_stack([world_pts, np.ones(len(world_pts))])
    M_inv = np.linalg.inv(M)
    unity_cam = (M_inv @ W.T).T[:, :3]
    opencv = unity_cam.copy()
    opencv[:, 1] *= -1.0
    return opencv


def sample_floor(w, d, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-w / 2, w / 2, n)
    zs = rng.uniform(1.0, 1.0 + d, n)                        # in front of camera
    return np.column_stack([xs, zs])


def rigid_M(yaw_deg, t):
    a = np.radians(yaw_deg)
    R = np.array([[np.cos(a), 0, np.sin(a)],
                  [0, 1, 0],
                  [-np.sin(a), 0, np.cos(a)]])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def test_lock_caps_and_stability():
    acc = FloorAccumulator(min_frames=6, min_cells=150, stability_window=5,
                           max_size_m=3.0)
    M = rigid_M(20.0, [4.0, 1.6, 2.0])     # arbitrary world pose
    floor_y = 0.0
    locked_rect = None
    for i in range(12):
        wxz = sample_floor(2.0, 2.5, n=1600, seed=i)       # 2.0 x 2.5 m floor
        # add small per-frame noise so it's realistic but stable
        wxz = wxz + np.random.default_rng(100 + i).normal(0, 0.01, wxz.shape)
        oc = make_opencv_inliers_for_world(wxz, floor_y, M)
        rect = acc.add_frame(oc, M, confidence=0.8, frame_id=str(i))
        if acc.ready_to_lock():
            locked_rect = acc.lock()
            print(f"LOCKED at frame {i}: w={locked_rect['width']:.2f} "
                  f"d={locked_rect['depth']:.2f} area={locked_rect['area']:.2f} "
                  f"floorY={locked_rect['floor_y']:.3f} yaw={locked_rect['yaw_deg']:.1f} "
                  f"cells={locked_rect['cell_count']}")
            break
    assert locked_rect is not None, "accumulator never reached a stable lock"
    # sides should recover ~2.0 and ~2.5 (order-independent)
    sides = sorted([locked_rect['width'], locked_rect['depth']])
    assert 1.7 < sides[0] < 2.3, f"short side off: {sides}"
    assert 2.2 < sides[1] < 2.8, f"long side off: {sides}"
    assert abs(locked_rect['floor_y']) < 0.05, f"floorY off: {locked_rect['floor_y']}"
    # freeze: further frames must NOT change the rect
    before = acc.current_rect()
    acc.add_frame(make_opencv_inliers_for_world(sample_floor(2.9, 2.9, seed=99),
                                                0.0, M), M, frame_id="post")
    assert acc.current_rect() is before, "rect changed after lock (freeze violated)"
    print("PASS: stability lock + freeze")


def test_cap_3x3():
    acc = FloorAccumulator(min_frames=4, min_cells=150, stability_window=4,
                           max_size_m=3.0)
    M = rigid_M(0.0, [0, 1.5, 0])
    for i in range(10):
        wxz = sample_floor(5.0, 5.0, n=3000, seed=i)        # 5x5 floor -> must cap
        oc = make_opencv_inliers_for_world(wxz, 0.0, M)
        acc.add_frame(oc, M, confidence=0.9, frame_id=str(i))
        if acc.ready_to_lock():
            r = acc.lock()
            print(f"CAP lock: w={r['width']:.2f} d={r['depth']:.2f}")
            assert r['width'] <= 3.01 and r['depth'] <= 3.01, "exceeded 3x3 cap"
            print("PASS: 3x3 cap")
            return
    raise AssertionError("never locked in cap test")


def test_world_camera_roundtrip():
    # _world_corners_to_camera lives on the worker; replicate the math here.
    M = rigid_M(35.0, [1.2, 1.7, -0.4])
    world = [[0.5, 0.0, 2.0], [-0.5, 0.0, 2.0], [-0.5, 0.0, 3.0], [0.5, 0.0, 3.0]]
    M_inv = np.linalg.inv(M)
    for c in world:
        cam = M_inv @ np.array([c[0], c[1], c[2], 1.0])
        back = M @ np.array([cam[0], cam[1], cam[2], 1.0])
        assert np.allclose(back[:3], c, atol=1e-9), f"roundtrip failed {c} -> {back[:3]}"
    print("PASS: world<->camera roundtrip")


def test_wall_contamination_floorY():
    """Frames see floor (low world-Y) + wall (high world-Y) mixed together.
    The locked rectangle must sit on the FLOOR (floorY low, all corners ~flat),
    NOT half-way up the wall. This reproduces the reported bug where the cube
    climbed the wall (boundary world_y spanned -0.07..1.75 m)."""
    acc = FloorAccumulator(min_frames=5, min_cells=150, stability_window=4,
                           max_size_m=3.0, plane_band_m=0.06)
    M = rigid_M(15.0, [2.0, 1.5, 1.0])
    FLOOR_Y = 0.0
    locked = None
    for i in range(12):
        # 40% floor points at Y=0, 60% wall points at Y=0.4..1.6 (above floor)
        floor_xz = sample_floor(2.0, 2.2, n=900, seed=i)
        wall_xz = sample_floor(2.0, 2.2, n=1400, seed=100 + i)
        floor_w = make_opencv_inliers_for_world(floor_xz, FLOOR_Y, M)
        # wall points: same XZ footprint but lifted to random heights up the wall
        wall_world = np.column_stack([
            wall_xz[:, 0],
            np.random.default_rng(7 + i).uniform(0.4, 1.6, len(wall_xz)),
            wall_xz[:, 1], np.ones(len(wall_xz))])
        Minv = np.linalg.inv(M)
        wall_uc = (Minv @ wall_world.T).T[:, :3]
        wall_uc[:, 1] *= -1.0
        oc = np.vstack([floor_w, wall_uc])
        acc.add_frame(oc, M, confidence=0.8, frame_id=str(i))
        if acc.ready_to_lock():
            locked = acc.lock()
            break
    assert locked is not None, "never locked under wall contamination"
    print(f"WALL-CONTAM lock: floorY={locked['floor_y']:.3f} "
          f"w={locked['width']:.2f} d={locked['depth']:.2f}")
    # floorY must be on the floor (~0), NOT mid-wall (~0.8)
    assert abs(locked['floor_y'] - FLOOR_Y) < 0.08, \
        f"floorY climbed the wall: {locked['floor_y']}"
    # all corners flat at floorY (small vertical spread)
    ys = [c[1] for c in locked['corners_world']]
    assert max(ys) - min(ys) < 0.05, f"corners not flat: spread={max(ys)-min(ys):.3f}"
    print("PASS: floor-only lock under wall contamination (cube stays flat on floor)")


def test_tilted_floor_full_capture():
    """Depth Anything curls the FAR floor up (~16°), so the floor comes back
    TILTED in world. The plane-distance band must keep the WHOLE floor (near +
    far), not just the lowest near strip. Reproduces the reported 2.2x0.4
    thin-cube bug — the locked rectangle must recover the full ~2x2 footprint."""
    acc = FloorAccumulator(min_frames=5, min_cells=150, stability_window=4,
                           max_size_m=3.0, plane_band_m=0.10)
    M = rigid_M(10.0, [1.0, 1.5, 0.5])
    theta = np.radians(16.0)
    n_world = np.array([0.0, np.cos(theta), -np.sin(theta)])   # tilted floor normal
    locked = None
    for i in range(12):
        xz = sample_floor(2.0, 2.0, n=2200, seed=i)            # 2x2 floor footprint
        x = xz[:, 0]; z = xz[:, 1]
        y = np.tan(theta) * (z - z.mean())                     # far floor curls UP
        y = y + np.random.default_rng(5 + i).normal(0, 0.01, len(y))
        world_pts = np.column_stack([x, y, z])
        oc = make_opencv_inliers_for_world_pts(world_pts, M)
        acc.add_frame(oc, M, normal_world=n_world, confidence=0.8, frame_id=str(i))
        if acc.ready_to_lock():
            locked = acc.lock()
            break
    assert locked is not None, "tilted floor never locked (thin strip never stabilised)"
    sides = sorted([locked['width'], locked['depth']])
    print(f"TILTED lock: w={locked['width']:.2f} d={locked['depth']:.2f} floorY={locked['floor_y']:.3f}")
    assert sides[0] > 1.5, f"floor collapsed to a thin strip: {sides}"   # NOT 0.4
    print("PASS: tilted floor captured in FULL (no thin strip, lock achieved)")


def test_area_growth_gate():
    """While the scanned floor keeps GROWING, the accumulator must NOT lock — it
    waits until the area plateaus, then locks the LARGEST floor (not an early
    small patch). Reproduces 'locked at 1.77x1.86 while more floor was visible'."""
    acc = FloorAccumulator(min_frames=4, min_cells=80, stability_window=3,
                           max_size_m=3.0, plane_band_m=0.10,
                           area_growth_frames=4, area_growth_eps_m2=0.06)
    M = rigid_M(0.0, [0, 1.5, 0])
    locked_early = False
    locked = None
    for i in range(24):
        size = min(2.6, 1.0 + i * 0.3)          # floor grows each frame, caps at 2.6
        xz = sample_floor(size, size, n=2400, seed=i)
        oc = make_opencv_inliers_for_world(xz, 0.0, M)
        acc.add_frame(oc, M, normal_world=[0.0, 1.0, 0.0], confidence=0.9, frame_id=str(i))
        if acc.ready_to_lock():
            r = acc.current_rect()
            if size < 2.5:                      # still growing → must NOT be lockable
                locked_early = True
            locked = acc.lock()
            print(f"GROWTH-GATE lock at frame {i}: size_in={size:.2f} "
                  f"w={locked['width']:.2f} d={locked['depth']:.2f} "
                  f"bestArea={acc.best_area:.2f}")
            break
    assert not locked_early, "locked while floor still growing (area-growth gate failed)"
    assert locked is not None, "never locked after floor plateaued"
    sides = sorted([locked['width'], locked['depth']])
    assert sides[0] > 2.0, f"did not grow to the largest floor: {sides}"
    print("PASS: area-growth gate waited for the largest stable floor")


if __name__ == "__main__":
    test_world_camera_roundtrip()
    test_lock_caps_and_stability()
    test_cap_3x3()
    test_wall_contamination_floorY()
    test_tilted_floor_full_capture()
    test_area_growth_gate()
    print("ALL TESTS PASSED")
