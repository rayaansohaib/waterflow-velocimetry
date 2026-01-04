"""
Water Surface Velocity Pipeline (v1)
-----------------------------------
Turns a monocular video of surface tracers (floaters/foam/bubbles) + optional IMU/GNSS
into water-surface velocity in m/s, along with overlays and CSV outputs.

Quick start
===========
1) Install deps (Python 3.10+ recommended):
   pip install -r requirements.txt
   # minimal: opencv-python numpy pyyaml scipy

2) Calibrate homography + scale (one-time per site/view):
   python pipeline.py calibrate \
       --video path/to/video.mp4 \
       --roi out/roi.json \
       --homography out/H.npy \
       --scale out/scale.json
   # This launches two small GUIs:
   #  (a) ROI polygon click (right-click to close, 'ENTER' to save)
   #  (b) Homography click (click 4 coplanar points clockwise, 'ENTER' to compute)
   #  (C) Scale selection after warp (click two points with known real distance), input for distance made to cli

3.) Run pipeline:(with optional IMU stabilization)
python pipeline.py run \
  --video kayaking.mp4 \
  --roi out/roi.json \
  --homography out/H.npy \
  --scale out/scale.json \
  --imu out/imu_orientation.csv \
  --stabilize imu \
  --out_dir out/session_imu


Outputs
=======
- out/session_001/velocities.csv         # frame_time, mean_mps, median_mps, iqr_mps, n_vectors
- out/session_001/overlay.mp4            # visual arrows on warped ROI
- out/session_001/summary.json           # QC metrics and parameters

Notes
=====
- IMU (CSV inputs) to improve stabilization .
- All geometry happens on a homography-rectified (top-down) plane to make px↔meters uniform.


"""
from __future__ import annotations
import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from scipy import stats

# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_json(obj, path: str):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


# -----------------------------
# Configuration structures
# -----------------------------

@dataclass
class LKParams:
    max_corners: int = 3000
    quality_level: float = 0.02
    min_distance: int = 8
    block_size: int = 7

    win_size: Tuple[int, int] = (21, 21)
    max_level: int = 3
    criteria_eps: float = 0.03
    criteria_iters: int = 30

    fb_thresh_px: float = 1.0  # forward-backward check threshold (px)


@dataclass
class RansacParams:
    reproj_thresh_px: float = 1.5
    min_inlier_frac: float = 0.6


@dataclass
class ScaleModel:
    meters_per_px: float
    baselines: List[Dict]


# -----------------------------
# ROI selection tool
# -----------------------------

class ROISelector:
  

    def __init__(self, window_name: str = 'ROI Selector'):
        self.window_name = window_name
        self.points: List[Tuple[int, int]] = []
        self.closed = False
        self._window_open = False

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.closed:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.closed = True
       

    def select(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        # Create and manage window robustly; handle OS-level close events.
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._window_open = True
        cv2.setMouseCallback(self.window_name, self._mouse_cb)

        def _finalize_and_close():
            # Remove callbacks and destroy window safely; give time to close on some platforms
            cv2.setMouseCallback(self.window_name, lambda *args: None)
            if self._window_open:
                try:
                    cv2.destroyWindow(self.window_name)
                except cv2.error:
                    pass
            cv2.waitKey(1)  # ensure the window loop pumps once

        while True:
            disp = frame.copy()
            # draw current poly
            for i, p in enumerate(self.points):
                cv2.circle(disp, p, 3, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(disp, self.points[i - 1], p, (0, 255, 0), 2)
            if (self.closed or len(self.points) >= 3):
                cv2.polylines(disp, [np.array(self.points, np.int32)], True, (0, 255, 255), 2)
            cv2.putText(
                disp,
                "Left:add  Right/'c':close  's'/ENTER:save  'r':reset  ESC/'q':quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(self.window_name, disp)

            k = cv2.waitKey(16) & 0xFF
            # Handle user keys
            if k == ord('r'):
                self.points = []
                self.closed = False
            elif k in (ord('c'),):
                if len(self.points) >= 3:
                    self.closed = True
            elif k in (ord('s'), 13, 10, 141, 13 & 0xFF):  # 's', Enter variants
                if len(self.points) >= 3:
                    # Auto-close if user forgot to right-click
                    self.closed = True
                    _finalize_and_close()
                    return self.points
            elif k in (27, ord('q')):  # ESC or 'q'
                _finalize_and_close()
                raise SystemExit('ROI selection aborted.')

            # Detect manual window close (clicking the ‘X’)
            try:
                prop = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
                if prop < 1:
                    self._window_open = False
                    raise SystemExit('ROI window closed by user.')
            except cv2.error:
                self._window_open = False
                raise SystemExit('ROI window closed unexpectedly.')
       


# -----------------------------
# Homography calibration tool
# -----------------------------

class HomographyCalibrator:
    """Interactively compute H: click 4+ image points and provide their rectified target positions.
    For practicality we let user specify the target as a rectangle of known size (meters),
    or auto-rectify an arbitrary quad to a canonical rectangle.
    """

    def __init__(self, reproj_thresh_px: float = 1.5):
        self.reproj_thresh_px = reproj_thresh_px

    @staticmethod
    def _order_quad(pts: np.ndarray) -> np.ndarray:
        # Order points (tl, tr, br, bl)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).ravel()
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def compute_h_from_quad(self, frame: np.ndarray) -> np.ndarray:
        # Click 4 points (coplanar) in image, map to canonical rectangle sized from pixel distances
        clicks: List[Tuple[int, int]] = []

        def cb(event, x, y, flags, param):
            nonlocal clicks
            if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
                clicks.append((x, y))

        win = 'Click 4 planar points (clockwise)'; cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, cb)
        while True:
            disp = frame.copy()
            for p in clicks:
                cv2.circle(disp, p, 5, (0, 0, 255), -1)
            cv2.putText(disp, "Click 4 points → ENTER to compute, 'r' reset, ESC abort", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow(win, disp)
            k = cv2.waitKey(16) & 0xFF
            if k == ord('r'):
                clicks = []
            if k in (27, ord('q')):
                cv2.destroyWindow(win); cv2.waitKey(1)
                raise SystemExit('Homography selection aborted')
            if len(clicks) == 4 and k in (13, ord('s')):
                cv2.setMouseCallback(win, lambda *args: None)
                cv2.destroyWindow(win); cv2.waitKey(1)
                break

        src = np.array(clicks, dtype=np.float32)
        src = self._order_quad(src)
        # Target rectangle: width = distance top edge, height = distance left edge (in pixels for now)
        w = np.linalg.norm(src[1] - src[0])
        h = np.linalg.norm(src[3] - src[0])
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)  # ALWAYS returns 3x3 float64 for 4 points
        H = H.astype(np.float64)
        return H



# -----------------------------
# Scaling tool (px → meters)
# -----------------------------

class ScaleCalibrator:
    def __init__(self):
        pass

    def compute_scale_interactive(self, warped_frame: np.ndarray, known_distance_m: Optional[float] = None) -> ScaleModel:
        """Let user click two points in the warped plane and enter real distance (m)."""
        clicks: List[Tuple[int, int]] = []

        def cb(event, x, y, flags, param):
            nonlocal clicks
            if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
                clicks.append((x, y))

        win = 'Scale: click 2 points (ENTER/\'s\' to confirm, ESC abort)'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, cb)

        meters = known_distance_m
        while True:
            disp = warped_frame.copy()
            for p in clicks:
                cv2.circle(disp, p, 5, (255, 0, 0), -1)
            if len(clicks) == 2:
                cv2.line(disp, clicks[0], clicks[1], (255, 0, 0), 2)
                d_px = float(np.linalg.norm(np.array(clicks[1]) - np.array(clicks[0])))
                cv2.putText(disp, f"d_px = {d_px:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                if meters is None:
                    cv2.putText(disp, "Press 'd' to type distance in meters", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.imshow(win, disp)
            k = cv2.waitKey(16) & 0xFF
            if k in (27, ord('q')):
                cv2.destroyWindow(win); cv2.waitKey(1)
                raise SystemExit('Scale selection aborted')
            if len(clicks) == 2:
                if meters is None and k == ord('d'):
                    try:
                        meters = float(input('Enter real known distance in meters: '))
                    except Exception:
                        meters = None
                if meters and meters > 0 and k in (13, ord('s')):
                    cv2.setMouseCallback(win, lambda *args: None)
                    cv2.destroyWindow(win); cv2.waitKey(1)
                    d_px = float(np.linalg.norm(np.array(clicks[1]) - np.array(clicks[0])))
                    m_per_px = meters / d_px
                    return ScaleModel(meters_per_px=m_per_px,
                                      baselines=[{"px": d_px, "m": meters}])


# -----------------------------
# Stabilization (visual-only with optional IMU/GNSS)
# -----------------------------

class Stabilizer:
    def __init__(self, ransac: RansacParams):
        self.ransac = ransac
        self.prev_gray: Optional[np.ndarray] = None
        self.cum_H = np.eye(3, dtype=np.float32)

    def reset(self):
        self.prev_gray = None
        self.cum_H = np.eye(3, dtype=np.float32)

    def step(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate inter-frame homography and accumulate to stabilize video.
        Returns (stabilized_frame, cumulative_H).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame, self.cum_H

        # Detect & match features between prev and current
        pts_prev = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=2000, qualityLevel=0.01, minDistance=8, blockSize=7)
        if pts_prev is None:
            self.prev_gray = gray
            return frame, self.cum_H
        pts_curr, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts_prev, None,
                                                     winSize=(21, 21), maxLevel=3,
                                                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))
        ok = (st.reshape(-1) == 1)
        src = pts_prev.reshape(-1, 2)[ok]
        dst = pts_curr.reshape(-1, 2)[ok]
        if len(src) < 10:
            self.prev_gray = gray
            return frame, self.cum_H
        H, mask = cv2.findHomography(dst, src, cv2.RANSAC, self.ransac.reproj_thresh_px)
        if H is None:
            self.prev_gray = gray
            return frame, self.cum_H
        self.cum_H = self.cum_H @ H.astype(np.float32)
        h, w = frame.shape[:2]
        stabilized = cv2.warpPerspective(frame, self.cum_H, (w, h))
        self.prev_gray = gray
        return stabilized, self.cum_H


# -----------------------------
# Feature tracker on warped ROI
# -----------------------------

class FlowTracker:
    def __init__(self, lk: LKParams):
        self.lk = lk
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None

    def seed(self, gray_roi: np.ndarray, mask: Optional[np.ndarray] = None):
        self.prev_gray = gray_roi
        self.prev_pts = cv2.goodFeaturesToTrack(gray_roi,
                                               maxCorners=self.lk.max_corners,
                                               qualityLevel=self.lk.quality_level,
                                               minDistance=self.lk.min_distance,
                                               blockSize=self.lk.block_size,
                                               mask=mask)

    def step(self, gray_roi: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 30:
            self.seed(gray_roi, mask)
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

        next_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray_roi, self.prev_pts, None,
                                                     winSize=self.lk.win_size,
                                                     maxLevel=self.lk.max_level,
                                                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                               self.lk.criteria_iters, self.lk.criteria_eps))
        # Forward-backward check
        back_pts, st2, err2 = cv2.calcOpticalFlowPyrLK(gray_roi, self.prev_gray, next_pts, None,
                                                       winSize=self.lk.win_size,
                                                       maxLevel=self.lk.max_level,
                                                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                                 self.lk.criteria_iters, self.lk.criteria_eps))
        fb = np.linalg.norm(self.prev_pts - back_pts, axis=2).reshape(-1)
        ok = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < self.lk.fb_thresh_px)
        p0 = self.prev_pts.reshape(-1, 2)[ok]
        p1 = next_pts.reshape(-1, 2)[ok]

        # update
        self.prev_gray = gray_roi
        self.prev_pts = p1.reshape(-1, 1, 2)
        return p0, p1

class PlanarVO2D:
    """
    Estimate camera planar translation on the rectified (warped) plane using static-background features.

    We assume:
      - After IMU yaw stabilization, rotation is mostly removed.
      - After homography warp, motion is well-approximated by 2D translation on the plane.
    """

    def __init__(self, lk: LKParams, ransac: RansacParams):
        self.lk = lk
        self.ransac = ransac
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None

    def reset(self):
        self.prev_gray = None
        self.prev_pts = None

    def _seed(self, gray: np.ndarray, static_mask: np.ndarray):
        self.prev_gray = gray
        self.prev_pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=min(1500, self.lk.max_corners),
            qualityLevel=max(0.01, self.lk.quality_level),
            minDistance=self.lk.min_distance,
            blockSize=self.lk.block_size,
            mask=static_mask
        )

    def step(self, gray: np.ndarray, static_mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Returns:
          dp_px: (2,) translation of *static points* from prev->curr in pixels/frame (median robust).
                 This is v_static_obs in px/frame.
          n_inliers: inlier count used for dp estimate
        """
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 50:
            self._seed(gray, static_mask)
            return np.zeros(2, dtype=np.float32), 0

        next_pts, st, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None,
            winSize=self.lk.win_size,
            maxLevel=self.lk.max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      self.lk.criteria_iters, self.lk.criteria_eps)
        )

        ok = (st.reshape(-1) == 1)
        p0 = self.prev_pts.reshape(-1, 2)[ok]
        p1 = next_pts.reshape(-1, 2)[ok]

        if len(p0) < 50:
            self._seed(gray, static_mask)
            return np.zeros(2, dtype=np.float32), 0

        # Robustly fit translation using RANSAC 
        
        M, inliers = cv2.estimateAffinePartial2D(
            p0, p1,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac.reproj_thresh_px,
            maxIters=2000,
            confidence=0.99
        )

        if M is None or inliers is None:
            self._seed(gray, static_mask)
            return np.zeros(2, dtype=np.float32), 0

        inliers = inliers.reshape(-1).astype(bool)
        n_in = int(inliers.sum())
        if n_in < 30:
            self._seed(gray, static_mask)
            return np.zeros(2, dtype=np.float32), n_in

        # Translation from affine (p1 ≈ A p0 + t)
        tx = float(M[0, 2])
        ty = float(M[1, 2])

        # Update for next step (keep inlier points only)
        self.prev_gray = gray
        self.prev_pts = p1[inliers].reshape(-1, 1, 2)

        return np.array([tx, ty], dtype=np.float32), n_in


def make_static_mask(gray_warped: np.ndarray, water_mask: np.ndarray) -> np.ndarray:
    # static = outside water ROI
    static = cv2.bitwise_not(water_mask)

    # avoid blank regions: require some intensity
    valid = (gray_warped > 5).astype(np.uint8) * 255

    return cv2.bitwise_and(static, valid)

# -----------------------------
# Robust vector statistics
# -----------------------------

def robust_speed_stats(vecs_mps: np.ndarray) -> Dict[str, float]:
    """vecs_mps: (N,2) velocities in m/s. Returns robust summary along dominant direction.
    - Estimate dominant flow direction via PCA, project speeds, compute median & IQR.
    """
    if vecs_mps.size == 0:
        return {"mean": 0.0, "median": 0.0, "iqr": 0.0, "n": 0}
    # PCA direction
    X = vecs_mps - vecs_mps.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    d = Vt[0]  # principal direction
    speeds = vecs_mps @ d  # signed projection
    median = float(np.median(speeds))
    iqr = float(np.subtract(*np.percentile(speeds, [75, 25])))
    mean = float(np.mean(speeds))
    return {"mean": mean, "median": median, "iqr": iqr, "n": int(len(speeds))}


# -----------------------------
# Mask helpers
# -----------------------------

def polygon_mask(shape: Tuple[int, int], poly: List[Tuple[int, int]]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(poly) >= 3:
        cv2.fillPoly(mask, [np.array(poly, np.int32)], 255)
    return mask

# -----------------------------
# IMU loading + AHRS (gyro+accel+mag) + yaw stabilization
# -----------------------------

import csv
import numpy as np
import cv2

def load_imu_csv(path: str,
                 time_col_candidates=("t", "time", "timestamp", "time_s", "t_sec", "t_s"),
                 gx="gx", gy="gy", gz="gz",
                 ax="ax", ay="ay", az="az",
                 mx="mx", my="my", mz="mz"):
    """
    Load IMU CSV with gyro+accel+mag.

    Expected columns (names can be edited in args):
      time: seconds (preferred). If your time is ms/us, convert before export or adjust below.
      gyro: gx,gy,gz in deg/s or rad/s (we will convert later based on a flag)
      accel: ax,ay,az in m/s^2 or g (we normalize anyway)
      mag: mx,my,mz in any units (we normalize anyway)

    Returns:
      t (N,), gyr (N,3), acc (N,3), mag (N,3)
    """
    # read header, detect time column
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        # pick time col
        time_col = None
        for c in time_col_candidates:
            if c in cols:
                time_col = c
                break
        if time_col is None:
            raise ValueError(f"IMU CSV missing a time column. Tried: {time_col_candidates}. Found: {cols}")

        ts, gxs, gys, gzs, axs, ays, azs, mxs, mys, mzs = [], [], [], [], [], [], [], [], [], []
        for row in r:
            ts.append(float(row[time_col]))
            gxs.append(float(row[gx])); gys.append(float(row[gy])); gzs.append(float(row[gz]))
            axs.append(float(row[ax])); ays.append(float(row[ay])); azs.append(float(row[az]))
            mxs.append(float(row[mx])); mys.append(float(row[my])); mzs.append(float(row[mz]))

    t = np.asarray(ts, dtype=np.float64)
    gyr = np.stack([gxs, gys, gzs], axis=1).astype(np.float64)
    acc = np.stack([axs, ays, azs], axis=1).astype(np.float64)
    mag = np.stack([mxs, mys, mzs], axis=1).astype(np.float64)

    # sort by time (safety)
    order = np.argsort(t)
    return t[order], gyr[order], acc[order], mag[order]


def _norm(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / (n + eps)


def _quat_normalize(q):
    return _norm(q)


def _quat_mul(q1, q2):
    # q = q1 ⊗ q2, each [w,x,y,z]
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def _quat_conj(q):
    w,x,y,z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)


def _quat_to_yaw(q):
    """
    yaw (rad) from quaternion [w,x,y,z] using ZYX convention:
      yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    """
    w,x,y,z = q
    return float(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))


def madgwick_ahrs_step(q, omega_rad_s, acc, mag, dt, beta=0.08):
    """
    One Madgwick AHRS update using gyro+accel+mag.
    - omega_rad_s: (3,) gyro in rad/s
    - acc: (3,) accel (any units; normalized internally)
    - mag: (3,) mag (any units; normalized internally)
    - dt: seconds
    - beta: correction gain (bigger = more accel/mag trust; smaller = more gyro trust)
    Returns updated unit quaternion [w,x,y,z].
    """
    q = _quat_normalize(q)
    w,x,y,z = q
    gx,gy,gz = omega_rad_s

    # normalize measurements
    if np.linalg.norm(acc) < 1e-9 or np.linalg.norm(mag) < 1e-9:
        # fall back to gyro-only integration (no correction)
        q_dot = 0.5 * np.array([
            -x*gx - y*gy - z*gz,
             w*gx + y*gz - z*gy,
             w*gy - x*gz + z*gx,
             w*gz + x*gy - y*gx
        ], dtype=np.float64)
        return _quat_normalize(q + q_dot*dt)

    ax,ay,az = _norm(acc)
    mx,my,mz = _norm(mag)

    # compute reference direction of magnetic field
    m_quat = np.array([0.0, mx, my, mz], dtype=np.float64)
    h = _quat_mul(_quat_mul(q, m_quat), _quat_conj(q))
    bx = np.sqrt(h[1]*h[1] + h[2]*h[2])
    bz = h[3]

    # objective function (gravity + magnetic)
    f = np.array([
        2*(x*z - w*y) - ax,
        2*(w*x + y*z) - ay,
        2*(0.5 - x*x - y*y) - az,
        2*bx*(0.5 - y*y - z*z) + 2*bz*(x*z - w*y) - mx,
        2*bx*(x*y - w*z)       + 2*bz*(w*x + y*z) - my,
        2*bx*(w*y + x*z)       + 2*bz*(0.5 - x*x - y*y) - mz
    ], dtype=np.float64)

    # Jacobian
    J = np.array([
        [-2*y,               2*z,              -2*w,               2*x],
        [ 2*x,               2*w,               2*z,               2*y],
        [ 0.0,              -4*x,              -4*y,               0.0],
        [-2*bz*y,            2*bz*z,           -4*bx*y - 2*bz*w,   -4*bx*z + 2*bz*x],
        [-2*bx*z + 2*bz*x,   2*bx*y + 2*bz*w,   2*bx*x + 2*bz*z,   -2*bx*w + 2*bz*y],
        [ 2*bx*y,            2*bx*z - 4*bz*x,   2*bx*w - 4*bz*y,    2*bx*x]
    ], dtype=np.float64)

    step = J.T @ f
    step = _norm(step)

    # gyro-based quaternion derivative
    q_dot_omega = 0.5 * np.array([
        -x*gx - y*gy - z*gz,
         w*gx + y*gz - z*gy,
         w*gy - x*gz + z*gx,
         w*gz + x*gy - y*gx
    ], dtype=np.float64)

    q_dot = q_dot_omega - beta*step
    q_new = q + q_dot*dt
    return _quat_normalize(q_new)


class ImuAhrsYawStabilizer:
    """
    Precomputes yaw(t) from raw IMU and provides a per-frame 2D rotation warp
    to remove camera yaw (dominant rotational corruption for optical flow).
    """

    def __init__(self, imu_csv_path: str, gyro_units: str = "deg/s", beta: float = 0.08):
        t, gyr, acc, mag = load_imu_csv(imu_csv_path)
        self.t = t

        # convert gyro to rad/s
        if gyro_units.lower().startswith("deg"):
            gyr = gyr * (np.pi / 180.0)
        self.gyr = gyr
        self.acc = acc
        self.mag = mag
        self.beta = float(beta)

        # precompute quaternions + yaw
        self.q = np.zeros((len(t), 4), dtype=np.float64)
        self.yaw = np.zeros((len(t),), dtype=np.float64)
        qk = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # identity
        self.q[0] = qk
        self.yaw[0] = _quat_to_yaw(qk)

        for k in range(1, len(t)):
            dt = float(self.t[k] - self.t[k-1])
            dt = min(max(dt, 1e-4), 0.5)
            qk = madgwick_ahrs_step(qk, self.gyr[k], self.acc[k], self.mag[k], dt, beta=self.beta)
            self.q[k] = qk
            self.yaw[k] = _quat_to_yaw(qk)

        self.yaw0 = None  # set on first video frame

    def yaw_at(self, t_sec: float) -> float:
        # linear interpolation in yaw is OK for small frame-to-frame spacing
        # (we could unwrap; keep simple and robust)
        if t_sec <= self.t[0]:
            return float(self.yaw[0])
        if t_sec >= self.t[-1]:
            return float(self.yaw[-1])
        return float(np.interp(t_sec, self.t, self.yaw))

    def stabilize_frame(self, frame: np.ndarray, t_sec: float) -> np.ndarray:
        yaw = self.yaw_at(t_sec)
        if self.yaw0 is None:
            self.yaw0 = yaw

        # rotate by -(yaw - yaw0) around image center
        h, w = frame.shape[:2]
        center = (w / 2.0, h / 2.0)
        ang_deg = -np.degrees(yaw - self.yaw0)

        M = cv2.getRotationMatrix2D(center, ang_deg, 1.0)
        return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)

# -----------------------------
# Main pipeline
# -----------------------------

class Pipeline:
    def __init__(self, H: np.ndarray, roi_poly: List[Tuple[int, int]], scale: ScaleModel,
                 lk: LKParams, ransac: RansacParams,
                 fps: float):
        self.H = H
        self.roi_poly = roi_poly
        self.scale = scale
        self.lk = lk
        self.ransac = ransac
        self.fps = fps

    def run(self, video_path: str, out_dir: str, write_overlay: bool = True, imu_csv: str = None, imu_gyro_units: str = "deg/s", imu_beta: float = 0.08):
        vo = PlanarVO2D(self.lk, self.ransac)

        ensure_dir(out_dir)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = self.fps if self.fps > 0 else (src_fps if src_fps > 0 else 30.0)
        imu_stab = None
        if imu_csv:
            imu_stab = ImuAhrsYawStabilizer(imu_csv, gyro_units=imu_gyro_units, beta=imu_beta)

        # Prepare overlay writer lazily once we know warped size
        writer = None
        velocities_csv = []

        tracker = FlowTracker(self.lk)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            t_sec = frame_idx / fps  # you already do this later; move it up
            frame_in = frame
            if imu_stab is not None:
                frame_in = imu_stab.stabilize_frame(frame, t_sec)

            warped = cv2.warpPerspective(frame_in, self.H, (w, h))

            gray_full = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            mask_full = polygon_mask((h, w), self.roi_poly)     # water ROI mask
            static_mask = make_static_mask(gray_full, mask_full)

            # --- VO step: estimate static observed translation (px/frame) ---
            dp_static_px, n_vo = vo.step(gray_full, static_mask)

            # convert static flow to m/s
            v_static_m_s = dp_static_px * fps * self.scale.meters_per_px

            # camera velocity is the negative of static observed motion
            v_cam_m_s = -v_static_m_s   # (vx, vy) in world plane

            roi = cv2.bitwise_and(warped, warped, mask=mask_full)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            p0, p1 = tracker.step(gray, mask_full)
            if p0.shape[0] > 0:
                disp = (p1 - p0)                   # px/frame (water obs)
                v_px_s = disp * fps
                S = self.scale.meters_per_px
                v_water_rel = v_px_s * S           # v_{w/c}

                # Absolute water velocity
                v_water_abs = v_water_rel + v_cam_m_s.reshape(1, 2)

                stats_rel = robust_speed_stats(v_water_rel)
                stats_abs = robust_speed_stats(v_water_abs)
            else:
                stats_rel = {"mean": 0.0, "median": 0.0, "iqr": 0.0, "n": 0}
                stats_abs = {"mean": 0.0, "median": 0.0, "iqr": 0.0, "n": 0}


            t_sec = frame_idx / fps
            velocities_csv.append({
                "t": t_sec,
                "cam_vx": float(v_cam_m_s[0]),
                "cam_vy": float(v_cam_m_s[1]),
                "vo_inliers": int(n_vo),

                "rel_mean": stats_rel["mean"],
                "rel_median": stats_rel["median"],
                "rel_iqr": stats_rel["iqr"],
                "rel_n": stats_rel["n"],

                "abs_mean": stats_abs["mean"],
                "abs_median": stats_abs["median"],
                "abs_iqr": stats_abs["iqr"],
                "abs_n": stats_abs["n"],
            })


            # Draw overlay
            if write_overlay:
                vis = warped.copy()
                if p0.shape[0] > 0:
                    for (x0, y0), (x1, y1) in zip(p0, p1):
                        cv2.arrowedLine(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 1, tipLength=0.3)
                txt = f"median = {stats_abs['median']:.3f} m/s  (n={stats_abs['n']})"
                cv2.putText(vis, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(os.path.join(out_dir, 'overlay.mp4'), fourcc, fps, (vis.shape[1], vis.shape[0]))
                writer.write(vis)

            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        # Save CSV and summary
        import csv
        csv_path = os.path.join(out_dir, 'velocities.csv')
        with open(csv_path, 'w', newline='') as f:
            wcsv = csv.DictWriter(f, fieldnames = [
            "t", "cam_vx", "cam_vy", "vo_inliers",
            "rel_mean", "rel_median", "rel_iqr", "rel_n",
            "abs_mean", "abs_median", "abs_iqr", "abs_n",
            ]
            )
            wcsv.writeheader()
            for row in velocities_csv:
                wcsv.writerow(row)

        summary = {
            "meters_per_px": self.scale.meters_per_px,
            "roi": self.roi_poly,
            "lk": asdict(self.lk),
            "ransac": asdict(self.ransac),
            "fps": fps,
            "frames": frame_idx,
            "n_valid_frames": int(sum(1 for r in velocities_csv if r["abs_n"] > 0)),
        }
        save_json(summary, os.path.join(out_dir, 'summary.json'))


# -----------------------------
# Command handlers
# -----------------------------

def cmd_calibrate(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read first frame for calibration")

    # ROI selection
    if args.roi and not os.path.exists(args.roi):
        roi_tool = ROISelector()
        poly = roi_tool.select(frame)
        ensure_dir(os.path.dirname(args.roi))
        save_json({"polygon": poly}, args.roi)
        print(f"Saved ROI polygon to {args.roi}")
    else:
        poly = load_json(args.roi)["polygon"]

    # Homography (interactive quad → rect)
    calib = HomographyCalibrator(reproj_thresh_px=1.5)
    H = calib.compute_h_from_quad(frame)
    H = np.asarray(H)
    if H is None or H.shape != (3, 3) or H.dtype not in (np.float32, np.float64):
        print("DEBUG: H=", H)
        raise RuntimeError(f"Invalid homography returned: shape={None if H is None else H.shape}, dtype={None if H is None else H.dtype}")

    H = H.astype(np.float64)


    ensure_dir(os.path.dirname(args.homography))
    np.save(args.homography, H)
    print(f"Saved homography matrix to {args.homography}")

    # Show warped frame within ROI
    h, w = frame.shape[:2]
    warped = cv2.warpPerspective(frame, H, (w, h))
    mask = polygon_mask((h, w), poly)
    roi_warped = cv2.bitwise_and(warped, warped, mask=mask)
    cv2.imshow('Warped ROI preview', roi_warped); cv2.waitKey(300)

    # Scale calibration (click baseline on warped image)
    scaler = ScaleCalibrator()
    scale = scaler.compute_scale_interactive(warped)
    ensure_dir(os.path.dirname(args.scale))
    save_json(asdict(scale), args.scale)
    print(f"Saved scale to {args.scale} (meters_per_px={scale.meters_per_px:.6f})")

    # Ensure all OpenCV windows close cleanly
    cv2.destroyAllWindows()
    cv2.waitKey(1)



def cmd_run(args):
    H = np.load(args.homography)
    roi_poly = load_json(args.roi)["polygon"]
    scale_d = load_json(args.scale)
    scale = ScaleModel(meters_per_px=scale_d["meters_per_px"], baselines=scale_d.get("baselines", []))

    lk = LKParams()
    ransac = RansacParams()
    pipe = Pipeline(H=H, roi_poly=roi_poly, scale=scale, lk=lk, ransac=ransac,
                    fps=args.fps_override)
    pipe.run(args.video, args.out_dir, write_overlay=True,
         imu_csv=args.imu_csv,
         imu_gyro_units=args.imu_gyro_units,
         imu_beta=args.imu_beta)

    print(f"Done. Results in {args.out_dir}")


# -----------------------------
# CLI
# -----------------------------

def build_argparser():
    p = argparse.ArgumentParser(description='Water Surface Velocity Pipeline')
    sub = p.add_subparsers(dest='cmd', required=True)

    pc = sub.add_parser('calibrate', help='Select ROI, compute homography, and set scale')
    pc.add_argument('--video', required=True)
    pc.add_argument('--roi', required=True, help='Path to save/load ROI JSON')
    pc.add_argument('--homography', required=True, help='Path to save H (npy)')
    pc.add_argument('--scale', required=True, help='Path to save scale JSON')
    pc.set_defaults(func=cmd_calibrate)

    pr = sub.add_parser('run', help='Run velocity estimation')
    pr.add_argument('--video', required=True)
    pr.add_argument('--roi', required=True)
    pr.add_argument('--homography', required=True)
    pr.add_argument('--scale', required=True)
    pr.add_argument('--out_dir', required=True)
    pr.add_argument('--fps_override', type=float, default=0.0)
    pr.add_argument('--imu_csv', default=None, help='IMU CSV with columns time + gx,gy,gz + ax,ay,az + mx,my,mz')
    pr.add_argument('--imu_gyro_units', default="deg/s", choices=["deg/s", "rad/s"])
    pr.add_argument('--imu_beta', type=float, default=0.08, help='Madgwick beta gain (higher = more accel/mag correction)')

    pr.set_defaults(func=cmd_run)

    return p


def main(argv=None):
    argv = argv or sys.argv[1:]
    args = build_argparser().parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()

