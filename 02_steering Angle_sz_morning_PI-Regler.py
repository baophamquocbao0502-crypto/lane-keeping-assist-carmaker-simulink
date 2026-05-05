import cv2
import numpy as np
import math

# VIDEO_PATH = "2_road_281225_sz_vid.mp4"
VIDEO_PATH = "test_281225_new_1.mp4"

# ====== Lane detection tuning ======
BOTTOM_Y_TOP_RATIO = 0.55     # only process from 55% height to bottom
MIN_CONTOUR_AREA = 600        # remove small noise contours
EMA_ALPHA = 0.25              # smoothing for lane lines (0.15..0.35)
LINE_THICKNESS = 8

# ====== Steering tuning (PI with anti-windup) ======
K_HEADING = 1.0               # heading gain (deg -> deg)
K_OFFSET_P = 0.03             # offset P gain (px -> deg)
K_OFFSET_I = 0.008            # offset I gain (px*s -> deg) (start small: 0.004..0.015)
MAX_STEER_DEG = 25.0          # clamp display

I_CLAMP_DEG = 10.0            # clamp I-term contribution (deg)
LANE_LOST_RESET_SEC = 0.4     # reset integral if lane lost longer than this


def circular_mask(frame_bgr):
    """Mask circular view to remove black border (IPG-like camera view)."""
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    r = int(0.49 * w)
    cv2.circle(mask, (w // 2, h // 2), r, 255, -1)
    return cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)


def white_mask_hls(frame_bgr):
    """Extract white lane markings using HLS thresholds."""
    hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(hls, (0, 165, 0), (180, 255, 70))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)
    return mask


def keep_bottom(mask, y_top_ratio=BOTTOM_Y_TOP_RATIO):
    """Keep only the bottom region of the mask."""
    h, w = mask.shape
    y_top = int(y_top_ratio * h)
    out = np.zeros_like(mask)
    out[y_top:, :] = mask[y_top:, :]
    return out, y_top


def contour_centroid_x(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"])


def find_best_contour_by_side(contours, img_w, side="left"):
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


def fitline_to_ab(cnt):
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


def ema_update(prev, new, alpha=EMA_ALPHA):
    if new is None:
        return prev
    if prev is None:
        return new
    return (
        prev[0] * (1 - alpha) + new[0] * alpha,
        prev[1] * (1 - alpha) + new[1] * alpha,
    )


def draw_line_ab(img, ab, y1, y2, color=(0, 255, 0), thickness=LINE_THICKNESS):
    if ab is None:
        return
    a, b = ab
    x1 = int(a * y1 + b)
    x2 = int(a * y2 + b)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def x_at_y(ab, y):
    a, b = ab
    return a * y + b


def compute_steering_pi(left_ab, right_ab, w, y1, y2, dt,
                        i_state, lane_ok,
                        last_lane_ok_time, now_time):
    """
    PI on offset (P+I) with anti-windup and lane-gating.
    Returns (steer_info_dict or None, updated_last_lane_ok_time).
    """
    if left_ab is None or right_ab is None:
        return None, last_lane_ok_time

    xl1, xr1 = x_at_y(left_ab, y1), x_at_y(right_ab, y1)
    xl2, xr2 = x_at_y(left_ab, y2), x_at_y(right_ab, y2)

    xc1 = 0.5 * (xl1 + xr1)
    xc2 = 0.5 * (xl2 + xr2)

    offset_px = float(xc1 - (w / 2.0))

    dx = float(xc1 - xc2)
    dy = float(y1 - y2)
    heading_deg = math.degrees(math.atan2(dx, dy))

    # Lane validity timestamp
    if lane_ok:
        last_lane_ok_time = now_time

    # Reset integral if lane lost too long
    if (now_time - last_lane_ok_time) > LANE_LOST_RESET_SEC:
        i_state["i_deg"] = 0.0

    # P parts
    p_deg = K_OFFSET_P * offset_px
    h_deg = K_HEADING * heading_deg
    i_deg = float(i_state.get("i_deg", 0.0))

    # Provisional steering
    steer_raw = h_deg + p_deg + i_deg
    steer_clamped = max(-MAX_STEER_DEG, min(MAX_STEER_DEG, steer_raw))

    # Conditional integration (anti-windup)
    if lane_ok and dt is not None and dt > 1e-4 and dt < 0.2:
        saturated = (steer_raw != steer_clamped)
        i_update = (K_OFFSET_I * offset_px) * dt  # deg

        if not saturated:
            i_deg_new = i_deg + i_update
        else:
            # Allow integration only if it helps come out of saturation
            if (steer_raw > MAX_STEER_DEG and i_update < 0) or (steer_raw < -MAX_STEER_DEG and i_update > 0):
                i_deg_new = i_deg + i_update
            else:
                i_deg_new = i_deg  # freeze

        # Clamp I-term contribution
        i_deg_new = max(-I_CLAMP_DEG, min(I_CLAMP_DEG, i_deg_new))
        i_state["i_deg"] = i_deg_new

        # Recompute with updated I
        steer_raw = h_deg + p_deg + i_state["i_deg"]
        steer_clamped = max(-MAX_STEER_DEG, min(MAX_STEER_DEG, steer_raw))

    return {
        "offset_px": offset_px,
        "heading_deg": heading_deg,
        "steer_deg": steer_clamped,
        "steer_raw_deg": steer_raw,
        "i_deg": float(i_state.get("i_deg", 0.0)),
        "xc1": xc1,
        "xc2": xc2,
    }, last_lane_ok_time


def draw_hud_box(img, steer_info, fps=None):
    """Draw a framed HUD box with semi-transparent background."""
    x0, y0 = 20, 20
    box_w, box_h = 420, 190

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (30, 30, 30), -1)
    alpha = 0.55
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    tx = x0 + 15
    ty = y0 + 35
    line_h = 30

    cv2.putText(img, "STEERING HUD (PI)", (tx, ty), font, 0.85, (0, 255, 0), 2)
    ty += line_h + 5

    if steer_info is None:
        cv2.putText(img, "Lane not detected", (tx, ty), font, 0.8, (0, 0, 255), 2)
        if fps is not None:
            cv2.putText(img, f"FPS: {fps:.1f}", (x0 + box_w - 120, y0 + box_h - 10),
                        font, 0.6, (200, 200, 200), 1)
        return

    steer = steer_info["steer_deg"]
    steer_raw = steer_info.get("steer_raw_deg", steer)
    heading = steer_info["heading_deg"]
    offset = steer_info["offset_px"]
    i_deg = steer_info.get("i_deg", 0.0)

    cv2.putText(img, f"Steer: {steer:+.2f} deg (raw {steer_raw:+.2f})", (tx, ty), font, 0.75, (0, 255, 0), 2)
    ty += line_h

    cv2.putText(img, f"Heading: {heading:+.2f} deg", (tx, ty), font, 0.75, (255, 255, 255), 2)
    ty += line_h

    cv2.putText(img, f"Offset: {offset:+.1f} px", (tx, ty), font, 0.75, (255, 255, 255), 2)
    ty += line_h

    cv2.putText(img, f"I-term: {i_deg:+.2f} deg", (tx, ty), font, 0.75, (255, 255, 255), 2)

    if fps is not None:
        cv2.putText(img, f"FPS: {fps:.1f}", (x0 + box_w - 120, y0 + box_h - 10),
                    font, 0.6, (200, 200, 200), 1)


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    left_ab_s = None
    right_ab_s = None

    # PI integral state (stores I contribution directly in degrees)
    i_state = {"i_deg": 0.0}
    last_lane_ok_time = 0.0

    tick_freq = cv2.getTickFrequency()
    t_prev = cv2.getTickCount()
    t0 = t_prev

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

        # 3) Fit + smooth
        left_ab = fitline_to_ab(left_cnt)
        right_ab = fitline_to_ab(right_cnt)
        left_ab_s = ema_update(left_ab_s, left_ab)
        right_ab_s = ema_update(right_ab_s, right_ab)

        out = frame.copy()

        # Optional contour debug
        if left_cnt is not None:
            cv2.drawContours(out, [left_cnt], -1, (0, 255, 0), 3)
        if right_cnt is not None:
            cv2.drawContours(out, [right_cnt], -1, (0, 255, 0), 3)

        # 4) Draw lane lines
        y1 = h
        y2 = int(BOTTOM_Y_TOP_RATIO * h)
        draw_line_ab(out, left_ab_s, y1, y2, (0, 255, 0), LINE_THICKNESS)
        draw_line_ab(out, right_ab_s, y1, y2, (0, 255, 0), LINE_THICKNESS)

        # FPS / dt
        t_now = cv2.getTickCount()
        dt = (t_now - t_prev) / tick_freq
        t_prev = t_now
        fps = (1.0 / dt) if dt > 1e-6 else None
        now_time = (t_now - t0) / tick_freq

        # Lane validity gating for integral
        # (basic) require both smoothed lines exist; you can tighten this if needed
        lane_ok = (left_ab_s is not None) and (right_ab_s is not None)

        # 5) Steering (PI) + centerline
        steer_info, last_lane_ok_time = compute_steering_pi(
            left_ab_s, right_ab_s, w, y1, y2,
            dt=dt,
            i_state=i_state,
            lane_ok=lane_ok,
            last_lane_ok_time=last_lane_ok_time,
            now_time=now_time
        )

        if steer_info is not None:
            xc1 = int(steer_info["xc1"])
            xc2 = int(steer_info["xc2"])
            cv2.line(out, (xc1, y1), (xc2, y2), (255, 0, 0), 4)  # centerline

        # 6) HUD box
        draw_hud_box(out, steer_info, fps=fps)

        cv2.imshow("mask_bottom", mask_b)
        cv2.imshow("result", out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
