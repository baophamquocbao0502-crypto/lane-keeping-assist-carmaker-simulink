import cv2
import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, Tuple

# =======================
# INPUT
# =======================
VIDEO_PATH = "test_281225_new_1.mp4"

# =======================
# Lane detection tuning
# =======================
BOTTOM_Y_TOP_RATIO = 0.55     # only process from this height to bottom
MIN_CONTOUR_AREA = 600        # remove small noise contours
EMA_ALPHA = 0.25              # smoothing for lane lines (0.15..0.35)
LINE_THICKNESS = 8

# Visualization tuning
LANE_DRAW_TOP_RATIO = 0.70    # draw lane lines only up to this height (for display)


@dataclass
class PerceptionOut:
    lane_ok: bool
    offset_px: float
    heading_deg: float
    xc1: float
    xc2: float


def circular_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Mask circular view to remove black border (IPG-like camera view)."""
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    r = int(0.49 * w)
    cv2.circle(mask, (w // 2, h // 2), r, 255, -1)
    return cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)


def white_mask_hls(frame_bgr: np.ndarray) -> np.ndarray:
    """Extract white lane markings using HLS thresholds."""
    hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(hls, (0, 165, 0), (180, 255, 70))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)
    return mask


def keep_bottom(mask: np.ndarray, y_top_ratio: float = BOTTOM_Y_TOP_RATIO) -> Tuple[np.ndarray, int]:
    """Keep only the bottom region of the mask."""
    h, w = mask.shape
    y_top = int(y_top_ratio * h)
    out = np.zeros_like(mask)
    out[y_top:, :] = mask[y_top:, :]
    return out, y_top


def contour_centroid_x(cnt: np.ndarray) -> Optional[int]:
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"])


def find_best_contour_by_side(contours, img_w: int, side: str = "left") -> Optional[np.ndarray]:
    """Pick the largest contour on the given side of the image."""
    best = None
    best_area = 0.0
    cx_split = img_w // 2

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        cx = contour_centroid_x(cnt)
        if cx is None:
            continue

        if side == "left" and cx >= cx_split:
            continue
        if side == "right" and cx <= cx_split:
            continue

        if area > best_area:
            best_area = area
            best = cnt

    return best


def fitline_to_ab(cnt: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
    """
    Fit line x = a*y + b from contour points using cv2.fitLine.
    Returns (a, b) or None.
    """
    if cnt is None:
        return None

    pts = cnt.reshape(-1, 2).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

    if abs(vy) < 1e-6:
        return None

    a = float(vx / vy)
    b = float(x0 - a * y0)
    return (a, b)


def ema_update(prev: Optional[Tuple[float, float]],
               new: Optional[Tuple[float, float]],
               alpha: float = EMA_ALPHA) -> Optional[Tuple[float, float]]:
    if new is None:
        return prev
    if prev is None:
        return new
    return (prev[0] * (1 - alpha) + new[0] * alpha,
            prev[1] * (1 - alpha) + new[1] * alpha)


def draw_line_ab(img: np.ndarray,
                 ab: Optional[Tuple[float, float]],
                 y1: int,
                 y2: int,
                 color=(0, 0, 255),
                 thickness: int = LINE_THICKNESS) -> None:
    """Draw x=a*y+b line segment."""
    if ab is None:
        return
    a, b = ab
    x1 = int(a * y1 + b)
    x2 = int(a * y2 + b)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def x_at_y(ab: Tuple[float, float], y: int) -> float:
    a, b = ab
    return a * y + b


def compute_lane_geometry(left_ab: Tuple[float, float],
                          right_ab: Tuple[float, float],
                          w: int,
                          y1: int,
                          y2: int) -> PerceptionOut:
    """
    Perception-only geometry:
    - offset_px computed at y1 (near vehicle)
    - heading_deg from centerline direction between y1 and y2
    """
    xl1, xr1 = x_at_y(left_ab, y1), x_at_y(right_ab, y1)
    xl2, xr2 = x_at_y(left_ab, y2), x_at_y(right_ab, y2)

    xc1 = 0.5 * (xl1 + xr1)
    xc2 = 0.5 * (xl2 + xr2)

    offset_px = float(xc1 - (w / 2.0))

    dx = float(xc1 - xc2)
    dy = float(y1 - y2)
    heading_deg = math.degrees(math.atan2(dx, dy))

    return PerceptionOut(
        lane_ok=True,
        offset_px=offset_px,
        heading_deg=heading_deg,
        xc1=xc1,
        xc2=xc2
    )


def draw_hud_perception(img: np.ndarray,
                        p: Optional[PerceptionOut],
                        fps: Optional[float] = None) -> None:
    """Simple HUD for perception-only signals."""
    x0, y0 = 20, 20
    box_w, box_h = 430, 165

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    tx = x0 + 15
    ty = y0 + 35
    line_h = 28

    cv2.putText(img, "PERCEPTION HUD (no controller)", (tx, ty), font, 0.7, (0, 255, 0), 2)
    ty += line_h + 4

    if p is None or (not p.lane_ok):
        cv2.putText(img, "Lane not detected", (tx, ty), font, 0.7, (0, 0, 255), 2)
        if fps is not None:
            cv2.putText(img, f"FPS: {fps:.1f}", (x0 + box_w - 120, y0 + box_h - 10),
                        font, 0.6, (200, 200, 200), 1)
        return

    cv2.putText(img, f"Heading: {p.heading_deg:+.2f} deg", (tx, ty), font, 0.65, (255, 255, 255), 2)
    ty += line_h
    cv2.putText(img, f"Offset:  {p.offset_px:+.1f} px", (tx, ty), font, 0.65, (255, 255, 255), 2)
    ty += line_h
    cv2.putText(img, f"lane_ok: {int(p.lane_ok)}", (tx, ty), font, 0.65, (255, 255, 255), 2)

    if fps is not None:
        cv2.putText(img, f"FPS: {fps:.1f}", (x0 + box_w - 120, y0 + box_h - 10),
                    font, 0.6, (200, 200, 200), 1)


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    left_ab_s = None
    right_ab_s = None

    tick_freq = cv2.getTickFrequency()
    t_prev = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w = frame.shape[:2]

        # 1) Preprocess
        frame_m = circular_mask(frame)
        mask = white_mask_hls(frame_m)
        mask_b, _ = keep_bottom(mask, BOTTOM_Y_TOP_RATIO)

        # 2) Contours
        contours, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        left_cnt = find_best_contour_by_side(contours, w, side="left")
        right_cnt = find_best_contour_by_side(contours, w, side="right")

        # 3) Fit + smooth (EMA)
        left_ab = fitline_to_ab(left_cnt)
        right_ab = fitline_to_ab(right_cnt)
        left_ab_s = ema_update(left_ab_s, left_ab)
        right_ab_s = ema_update(right_ab_s, right_ab)

        lane_ok = (left_ab_s is not None) and (right_ab_s is not None)

        out = frame.copy()

        # Optional contour debug
        if left_cnt is not None:
            cv2.drawContours(out, [left_cnt], -1, (0, 255, 0), 3)
        if right_cnt is not None:
            cv2.drawContours(out, [right_cnt], -1, (0, 255, 0), 3)

        # Draw lane lines (red) only for visualization
        y1 = h
        y2_draw = int(LANE_DRAW_TOP_RATIO * h)
        draw_line_ab(out, left_ab_s, y1, y2_draw, color=(0, 0, 255), thickness=LINE_THICKNESS)
        draw_line_ab(out, right_ab_s, y1, y2_draw, color=(0, 0, 255), thickness=LINE_THICKNESS)

        # Geometry extraction (no controller)
        y2_geom = int(BOTTOM_Y_TOP_RATIO * h)
        p_out: Optional[PerceptionOut] = None
        if lane_ok:
            p_out = compute_lane_geometry(left_ab_s, right_ab_s, w, y1, y2_geom)

            # Draw centerline (blue)
            xc1 = int(p_out.xc1)
            xc2 = int(p_out.xc2)
            cv2.line(out, (xc1, y1), (xc2, y2_geom), (255, 0, 0), 4)

        # FPS
        t_now = cv2.getTickCount()
        dt = (t_now - t_prev) / tick_freq
        t_prev = t_now
        fps = (1.0 / dt) if dt > 1e-6 else None

        # HUD
        draw_hud_perception(out, p_out, fps=fps)

        cv2.imshow("mask_bottom", mask_b)
        cv2.imshow("result", out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
