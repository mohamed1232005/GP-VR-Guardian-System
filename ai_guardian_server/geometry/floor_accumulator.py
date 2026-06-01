# ===========================================================================
# geometry/floor_accumulator.py — Phase 14 [WORLD_FLOOR_ACCUMULATION]
#
# Build the locked guardian rectangle from ALL trusted floor points seen during
# scanning, not just the last frame.
#
# Pipeline (per accepted frame):
#   1. Take the RANSAC inlier floor points (OpenCV camera space, metres).
#   2. Convert OpenCV cam -> Unity cam (flip Y) -> WORLD using the frame's
#      camera_to_world matrix M.   (M maps Unity-cam -> world: world = R·cam + t)
#   3. Keep only points within PLANE_BAND of the running trusted floor Y
#      (req 5 — points near the trusted RANSAC floor plane).
#   4. Splat them into a world XZ occupancy grid (cells accumulate hit counts +
#      summed Y).  Cells below MIN_CELL_HITS are treated as noise (req 4).
#   5. Take the largest connected blob of occupied cells (drops disconnected
#      outlier clusters), fit an oriented min-area rectangle, and clamp it to
#      MAX_SIZE × MAX_SIZE (req 6 + 7 — final rect from accumulated area, ≤3×3).
#   6. Track (center, rotation, floorY, width, depth) over a sliding window and
#      only allow a lock once all five are stable (req 8).
#
# After lock() the result is frozen and never recomputed (req 9 + 10 — the
# caller stops updating the boundary).
#
# Coordinate notes:
#   OpenCV cam : X=right, Y=down, Z=forward
#   Unity  cam : X=right, Y=up,   Z=forward      (OpenCV -> Unity cam = flip Y)
#   World      : Y=up (gravity).  world = M @ [x, -y_cv, z, 1]
# ===========================================================================

import math
import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - cv2 is always present in this server
    cv2 = None


def _circular_spread_deg(vals, period=90.0):
    """Minimal arc (in degrees) that covers all angle samples on a circle of
    circumference `period`. Used so a 0°/89° pair (≈ same rectangle, since a
    rectangle is 90°-symmetric) reads as a *small* rotation drift, not a huge one."""
    v = np.sort(np.mod(np.asarray(vals, dtype=np.float64), period))
    if v.size <= 1:
        return 0.0
    gaps = np.diff(v)
    wrap_gap = (v[0] + period) - v[-1]
    largest_gap = max(float(gaps.max()), float(wrap_gap))
    return float(period - largest_gap)


class FloorAccumulator:
    """World-space multi-frame floor point accumulator + stability lock gate."""

    def __init__(
        self,
        cell_size_m: float = 0.05,
        max_size_m: float = 3.0,
        min_cell_hits: int = 2,
        plane_band_m: float = 0.06,
        min_frames: int = 6,
        min_cells: int = 150,
        min_area_m2: float = 1.0,
        stability_window: int = 5,
        center_tol_m: float = 0.10,
        yaw_tol_deg: float = 6.0,
        floory_tol_m: float = 0.05,
        size_tol_m: float = 0.15,
        max_cells: int = 40000,
        area_growth_frames: int = 6,
        area_growth_eps_m2: float = 0.06,
    ):
        self._area_growth_frames = int(area_growth_frames)
        self._area_growth_eps    = float(area_growth_eps_m2)
        self._cell_size       = float(cell_size_m)
        self._max_size        = float(max_size_m)
        self._min_cell_hits   = int(min_cell_hits)
        self._plane_band      = float(plane_band_m)
        self._min_frames      = int(min_frames)
        self._min_cells       = int(min_cells)
        self._min_area        = float(min_area_m2)
        self._window          = int(stability_window)
        self._center_tol      = float(center_tol_m)
        self._yaw_tol         = float(yaw_tol_deg)
        self._floory_tol      = float(floory_tol_m)
        self._size_tol        = float(size_tol_m)
        self._max_cells       = int(max_cells)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        # occupancy grid: (gx, gz) -> [hit_count, summed_world_y]
        self._cells = {}
        self._floor_y = None          # running trusted floor Y (world)
        self._frame_count = 0
        self._conf_sum = 0.0
        self._conf_n = 0
        self._history = []            # recent (cx, cz, yaw, floorY, w_long, w_short)
        self._last_rect = None
        self._locked = False
        self._frozen_rect = None
        # area-growth gate
        self._best_area = 0.0
        self._growth_stable_frames = 0

    @property
    def locked(self) -> bool:
        return self._locked

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def cell_count(self) -> int:
        return len(self._cells)

    @property
    def best_area(self) -> float:
        return self._best_area

    @property
    def current_area(self) -> float:
        return self._last_rect["area"] if self._last_rect is not None else 0.0

    @property
    def growth_stable_frames(self) -> int:
        return self._growth_stable_frames

    @property
    def area_growth_frames(self) -> int:
        return self._area_growth_frames

    def current_rect(self):
        return self._frozen_rect if self._locked else self._last_rect

    # ------------------------------------------------------------------
    # Per-frame ingest
    # ------------------------------------------------------------------
    def add_frame(self, inlier_points_cv, M, plane=None, normal_world=None,
                  confidence=0.0, frame_id="?"):
        """Accumulate one accepted frame's trusted floor points. Returns the
        current accumulated rectangle dict (or None if not enough data yet).

        normal_world: the pose-derived floor normal in WORLD (≈ up). Used to band
        points by distance to the tilted floor PLANE rather than constant world-Y,
        so a floor that Depth Anything curves/tilts up at the far edge is kept in
        full (not clipped into a thin near strip)."""
        if self._locked:
            return self._frozen_rect
        if inlier_points_cv is None or M is None:
            return self._last_rect

        pts = np.asarray(inlier_points_cv, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 10 or pts.shape[1] < 3:
            return self._last_rect

        M = np.asarray(M, dtype=np.float64).reshape(4, 4)

        # --- OpenCV cam -> Unity cam (flip Y) -> world ---
        unity_cam = pts[:, :3].copy()
        unity_cam[:, 1] *= -1.0
        homog = np.concatenate([unity_cam, np.ones((unity_cam.shape[0], 1))], axis=1)
        world = (M @ homog.T).T[:, :3]
        ys = world[:, 1]

        # --- keep points near the trusted floor PLANE (gravity-aligned) ---
        # The floor often comes back TILTED in world (Depth Anything curves the
        # far floor up — here normal_world had a ~0.27 Z component, ~16°). A
        # constant-Y band then keeps only the lowest near edge and throws the far
        # floor away → a thin 0.4 m strip that never stabilises. Instead band by
        # perpendicular distance to the floor PLANE (point p0 on the floor,
        # normal n = gravity-up from the pose): the whole tilted floor (near AND
        # far) is within the band, while the wall — which is perpendicular to the
        # floor — is far from the plane and dropped.
        if normal_world is not None:
            n = np.asarray(normal_world, dtype=np.float64).reshape(3)
            nn = np.linalg.norm(n)
            n = n / nn if nn > 1e-9 else np.array([0.0, 1.0, 0.0])
        else:
            n = np.array([0.0, 1.0, 0.0])
        if n[1] < 0.0:
            n = -n                                  # ensure it points UP

        # Reference point ON the floor: centroid of the lowest band of points
        # (robust to wall/sofa contamination that sits above the floor).
        floor_y_low = float(np.percentile(ys, 20))
        near_floor  = ys < (floor_y_low + 0.15)
        if int(near_floor.sum()) >= 8:
            p0 = np.median(world[near_floor], axis=0)
        else:
            p0 = np.median(world, axis=0)

        plane_dist = (world - p0) @ n               # signed distance to floor plane
        keep = np.abs(plane_dist) < self._plane_band
        world = world[keep]
        if world.shape[0] < 8:
            return self._last_rect

        # cube base height = robust median Y of the kept floor (flat cube base)
        frame_floor_y = float(np.median(world[:, 1]))
        if self._floor_y is None:
            self._floor_y = frame_floor_y
        else:
            self._floor_y = 0.8 * self._floor_y + 0.2 * frame_floor_y

        # --- splat into world XZ occupancy grid (vectorised per-frame merge) ---
        cs = self._cell_size
        gx = np.round(world[:, 0] / cs).astype(np.int64)
        gz = np.round(world[:, 2] / cs).astype(np.int64)
        wy = world[:, 1]
        cell_xz = np.stack([gx, gz], axis=1)
        uniq, inv = np.unique(cell_xz, axis=0, return_inverse=True)
        counts = np.bincount(inv)
        sum_y = np.bincount(inv, weights=wy)
        for j in range(uniq.shape[0]):
            key = (int(uniq[j, 0]), int(uniq[j, 1]))
            cell = self._cells.get(key)
            if cell is None:
                if len(self._cells) >= self._max_cells:
                    continue
                self._cells[key] = [int(counts[j]), float(sum_y[j])]
            else:
                cell[0] += int(counts[j])
                cell[1] += float(sum_y[j])

        self._frame_count += 1
        self._conf_sum += float(confidence)
        self._conf_n += 1

        rect = self._fit_rect()
        if rect is not None:
            # area-growth gate: while the area keeps growing (user still scanning
            # new floor) reset the plateau counter; only when it stops growing do
            # we let the lock proceed — so we lock the LARGEST floor, not an early
            # small patch.
            area = rect["area"]
            if area > self._best_area + self._area_growth_eps:
                self._best_area = area
                self._growth_stable_frames = 0
            else:
                self._best_area = max(self._best_area, area)
                self._growth_stable_frames += 1
            self._last_rect = rect
            self._push_history(rect)
        return self._last_rect

    # ------------------------------------------------------------------
    # Rectangle fit from accumulated occupancy
    # ------------------------------------------------------------------
    def _occupied_cells(self):
        keep = [k for k, v in self._cells.items() if v[0] >= self._min_cell_hits]
        if not keep:
            return None
        return np.array(keep, dtype=np.int64)

    def _fit_rect(self):
        occ = self._occupied_cells()
        if occ is None or occ.shape[0] < 8:
            return None

        # Drop disconnected outlier clusters: keep the largest connected blob.
        occ = self._largest_blob(occ)
        if occ is None or occ.shape[0] < 8:
            return None

        cs = self._cell_size
        pts_xz = occ.astype(np.float32) * cs            # world X, Z (metres)
        (cx, cz), (w, d), yaw = self._min_area_rect(pts_xz)

        # Clamp to the 3×3 cap, shrinking symmetrically around the centre along
        # the rectangle's own axes (req 7).
        w_c = min(float(w), self._max_size)
        d_c = min(float(d), self._max_size)
        corners = self._rect_corners(cx, cz, w_c, d_c, yaw)

        floor_y = float(self._floor_y) if self._floor_y is not None else 0.0
        conf = (self._conf_sum / self._conf_n) if self._conf_n > 0 else 0.0

        return {
            "center":        (float(cx), float(cz)),
            "yaw_deg":       float(yaw),
            "width":         float(w_c),
            "depth":         float(d_c),
            "floor_y":       floor_y,
            "area":          float(w_c * d_c),
            "confidence":    float(min(1.0, max(0.0, conf))),
            "corners_world": [[float(x), floor_y, float(z)] for (x, z) in corners],
            "cell_count":    int(occ.shape[0]),
            "frame_count":   int(self._frame_count),
        }

    def _min_area_rect(self, pts_xz):
        """Return ((cx,cz),(w,d),yaw_deg). Oriented via cv2.minAreaRect; falls
        back to a world-axis-aligned box if OpenCV is unavailable."""
        if cv2 is not None and pts_xz.shape[0] >= 3:
            (cx, cz), (w, h), ang = cv2.minAreaRect(pts_xz.astype(np.float32))
            return (cx, cz), (w, h), ang
        x_lo, x_hi = float(pts_xz[:, 0].min()), float(pts_xz[:, 0].max())
        z_lo, z_hi = float(pts_xz[:, 1].min()), float(pts_xz[:, 1].max())
        return ((x_lo + x_hi) * 0.5, (z_lo + z_hi) * 0.5), (x_hi - x_lo, z_hi - z_lo), 0.0

    @staticmethod
    def _rect_corners(cx, cz, w, d, yaw_deg):
        a = math.radians(yaw_deg)
        ux, uz = math.cos(a), math.sin(a)       # width axis
        vx, vz = -math.sin(a), math.cos(a)      # depth axis
        hw, hd = w * 0.5, d * 0.5
        corners = []
        for sw, sd in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            x = cx + sw * hw * ux + sd * hd * vx
            z = cz + sw * hw * uz + sd * hd * vz
            corners.append((x, z))
        return corners

    def _largest_blob(self, occ):
        """occ: (N,2) int grid coords. Rasterise to a binary image, close small
        gaps, and keep the largest 8-connected component."""
        if cv2 is None:
            return occ
        gx = occ[:, 0]
        gz = occ[:, 1]
        minx, minz = int(gx.min()), int(gz.min())
        W = int(gx.max()) - minx + 3
        H = int(gz.max()) - minz + 3
        if W <= 0 or H <= 0 or (W * H) > 4_000_000:
            return occ
        img = np.zeros((H, W), dtype=np.uint8)
        img[(gz - minz + 1), (gx - minx + 1)] = 255
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, labels = cv2.connectedComponents(img, connectivity=8)
        if n <= 2:
            return occ
        counts = np.bincount(labels.ravel())
        counts[0] = 0                            # ignore background
        best = int(counts.argmax())
        ys, xs = np.where(labels == best)
        return np.stack([xs + minx - 1, ys + minz - 1], axis=1).astype(np.int64)

    # ------------------------------------------------------------------
    # Stability tracking + lock gate (req 8)
    # ------------------------------------------------------------------
    def _push_history(self, rect):
        cx, cz = rect["center"]
        w, d = rect["width"], rect["depth"]
        # store flip-invariant long/short sides so a 90° rect symmetry swap
        # doesn't read as a width/depth jump.
        w_long, w_short = (max(w, d), min(w, d))
        self._history.append((cx, cz, rect["yaw_deg"], rect["floor_y"], w_long, w_short))
        if len(self._history) > self._window:
            self._history.pop(0)

    def stability(self):
        n = len(self._history)
        if n < self._window:
            return {"stable": False, "frames": n}
        H = np.array(self._history, dtype=np.float64)
        cx, cz = H[:, 0], H[:, 1]
        yaw, fy = H[:, 2], H[:, 3]
        w_long, w_short = H[:, 4], H[:, 5]

        d_center = float(math.hypot(cx.max() - cx.min(), cz.max() - cz.min()))
        d_yaw = _circular_spread_deg(yaw, period=90.0)
        d_floory = float(fy.max() - fy.min())
        d_w = float(w_long.max() - w_long.min())
        d_d = float(w_short.max() - w_short.min())

        stable = (
            d_center <= self._center_tol and
            d_yaw    <= self._yaw_tol and
            d_floory <= self._floory_tol and
            d_w      <= self._size_tol and
            d_d      <= self._size_tol
        )
        return {
            "stable": bool(stable), "frames": n,
            "d_center": d_center, "d_yaw": d_yaw, "d_floory": d_floory,
            "d_w": d_w, "d_d": d_d,
        }

    def ready_to_lock(self) -> bool:
        """True only when enough trusted floor has accumulated AND the rectangle
        (center, rotation, floorY, width, depth) has been stable (req 8)."""
        if self._locked or self._last_rect is None:
            return False
        if self._frame_count < self._min_frames:
            return False
        if self.cell_count < self._min_cells:
            return False
        if self._last_rect["area"] < self._min_area:
            return False
        # area-growth gate: don't lock while the floor area is still growing.
        if self._growth_stable_frames < self._area_growth_frames:
            return False
        return self.stability().get("stable", False)

    def lock(self):
        """Freeze the current accumulated rectangle. Idempotent (req 9 + 10)."""
        if not self._locked:
            self._locked = True
            self._frozen_rect = self._last_rect
        return self._frozen_rect
