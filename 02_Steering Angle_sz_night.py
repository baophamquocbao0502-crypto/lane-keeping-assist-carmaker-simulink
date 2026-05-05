import cv2
import numpy as np
import math

VIDEO_PATH = "2_road_sz_vid_night.mp4"
# VIDEO_PATH = "test_281225_new_1.mp4"

# =========================
# Lane detection (Night, đã chạy tốt)
# =========================
BOTTOM_Y_TOP_RATIO = 0.45      # dùng cho mask/contour (night)
MIN_CONTOUR_AREA = 120
EMA_ALPHA = 0.25
LINE_THICKNESS = 8

# Vẽ ngắn lại (đường đỏ)
LANE_DRAW_TOP_RATIO = 0.70     # 0.65..0.80

# Auto brightness
GAIN_PCTL = 95
GAIN_TARGET = 180
GAIN_MAX = 6.0

# CLAHE
USE_CLAHE = True
CLAHE_CLIPLIMIT = 3.0
CLAHE_TILE = (8, 8)

# Adaptive HLS
L_PCTL = 85
L_MIN_FLOOR = 90
S_MAX = 190

# ROI trapezoid
ROI_Y_TOP_RATIO = 0.55
ROI_X_LEFT_BOT  = 0.06
ROI_X_RIGHT_BOT = 0.94
ROI_X_LEFT_TOP  = 0.36
ROI_X_RIGHT_TOP = 0.64

# =========================
# Steering tuning (như code bạn)
# =========================
K_HEADING = 1.0
K_OFFSET = 0.03
MAX_STEER_DEG = 25.0


def circular_mask(frame_bgr):
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    r = int(0.49 * w)
    cv2.circle(mask, (w // 2, h // 2), r, 255, -1)
    return cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)


def trapezoid_roi_mask(h, w):
    y_top = int(round(h * ROI_Y_TOP_RATIO))
    pts = np.array([
        [int(round(w * ROI_X_LEFT_BOT)),  h - 1],
        [int(round(w * ROI_X_LEFT_TOP)),  y_top],
        [int(round(w * ROI_X_RIGHT_TOP)), y_top],
        [int(round(w * ROI_X_RIGHT_BOT)), h - 1],
    ], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def auto_gain_bgr(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]

    v_p = float(np.percentile(V, GAIN_PCTL))
    if v_p < 1.0:
        return frame_bgr, 1.0

    gain = GAIN_TARGET / v_p
    gain = float(np.clip(gain, 1.0, GAIN_MAX))

    out = cv2.convertScaleAbs(frame_bgr, alpha=gain, beta=0)
    return out, gain


def clahe_on_l_channel_bgr(frame_bgr):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIPLIMIT, tileGridSize=CLAHE_TILE)
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def white_mask_hls_adaptive(frame_bgr):
    hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
    _, L, _ = cv2.split(hls)

    h, w = L.shape
    y_bot = int(BOTTOM_Y_TOP_RATIO * h)
    L_roi = L[y_bot:, :]

    l_thr = int(np.percentile(L_roi, L_PCTL))
    l_thr = max(L_MIN_FLOOR, l_thr)

    mask = cv2.inRange(hls, (0, l_thr, 0), (180, 255, S_MAX))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 3)
    return mask, l_thr


def keep_bottom(mask, y_top_ratio=BOTTOM_Y_TOP_RATIO):
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
    best = None
    best_area = 0.0
    cx_split = img_w // 2
    margin = int(0.06 * img_w)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        cx = contour_centroid_x(cnt)
        if cx is None:
            continue

        if side == "left":
            if cx >= cx_split - margin or cx <= margin:
                continue
        else:
            if cx <= cx_split + margin or cx >= img_w - margin:
                continue

        if area > best_area:
            best_area = area
            best = cnt

    return best


def fitline_to_ab(cnt):
    if cnt is None:
        return None

    pts = cnt.reshape(-1, 2).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(
        pts, cv2.DIST_L2, 0, 0.01, 0.01
    ).flatten()

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


def draw_line_ab(img, ab, y1, y2, color=(0, 0, 255), thickness=LINE_THICKNESS):
    """Lane màu đỏ."""
    if ab is None:
        return
    a, b = ab
    x1 = int(a * y1 + b)
    x2 = int(a * y2 + b)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def x_at_y(ab, y):
    a, b = ab
    return a * y + b


def compute_steering(left_ab, right_ab, w, y1, y2):
    """
    Tính steering dựa vào:
      - heading: góc hướng của tâm lane giữa y1 và y2
      - offset: lệch tâm lane tại y1 so với tâm ảnh
    """
    if left_ab is None or right_ab is None:
        return None

    xl1, xr1 = x_at_y(left_ab, y1), x_at_y(right_ab, y1)
    xl2, xr2 = x_at_y(left_ab, y2), x_at_y(right_ab, y2)

    xc1 = 0.5 * (xl1 + xr1)
    xc2 = 0.5 * (xl2 + xr2)

    offset_px = float(xc1 - (w / 2.0))

    dx = float(xc1 - xc2)
    dy = float(y1 - y2)
    heading_deg = math.degrees(math.atan2(dx, dy))

    steer_deg = K_HEADING * heading_deg + K_OFFSET * offset_px
    steer_deg = max(-MAX_STEER_DEG, min(MAX_STEER_DEG, steer_deg))

    return {
        "offset_px": offset_px,
        "heading_deg": heading_deg,
        "steer_deg": steer_deg,
        "xc1": xc1,
        "xc2": xc2,
    }


def draw_hud_box(img, steer_info, fps=None, extra_text=None):
    x0, y0 = 20, 20
    box_w, box_h = 420, 190

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    tx = x0 + 15
    ty = y0 + 35
    line_h = 30

    cv2.putText(img, "STEERING HUD (Night)", (tx, ty), font, 0.9, (0, 255, 0), 2)
    ty += line_h + 5

    if steer_info is None:
        cv2.putText(img, "Lane not detected", (tx, ty), font, 0.8, (0, 0, 255), 2)
        ty += line_h
    else:
        steer = steer_info["steer_deg"]
        heading = steer_info["heading_deg"]
        offset = steer_info["offset_px"]

        cv2.putText(img, f"Steer: {steer:+.2f} deg", (tx, ty), font, 0.9, (0, 255, 0), 2)
        ty += line_h
        cv2.putText(img, f"Heading: {heading:+.2f} deg", (tx, ty), font, 0.75, (255, 255, 255), 2)
        ty += line_h
        cv2.putText(img, f"Offset: {offset:+.1f} px", (tx, ty), font, 0.75, (255, 255, 255), 2)
        ty += line_h

    if extra_text is not None:
        cv2.putText(img, extra_text, (tx, ty), font, 0.65, (0, 255, 255), 2)

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

        # FPS
        t_now = cv2.getTickCount()
        dt = (t_now - t_prev) / tick_freq
        t_prev = t_now
        fps = (1.0 / dt) if dt > 1e-6 else None

        # 1) Lane detection (night pipeline)
        frame_m = circular_mask(frame)
        frame_g, gain = auto_gain_bgr(frame_m)
        if USE_CLAHE:
            frame_g = clahe_on_l_channel_bgr(frame_g)

        roi_mask = trapezoid_roi_mask(h, w)
        frame_roi = cv2.bitwise_and(frame_g, frame_g, mask=roi_mask)

        mask, l_thr = white_mask_hls_adaptive(frame_roi)
        mask_b, _ = keep_bottom(mask, BOTTOM_Y_TOP_RATIO)

        contours, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        left_cnt = find_best_contour_by_side(contours, w, side="left")
        right_cnt = find_best_contour_by_side(contours, w, side="right")

        left_ab = fitline_to_ab(left_cnt)
        right_ab = fitline_to_ab(right_cnt)
        left_ab_s = ema_update(left_ab_s, left_ab)
        right_ab_s = ema_update(right_ab_s, right_ab)

        # 2) Draw lanes (red, shorter)
        out = frame.copy()
        y1 = h - 1
        y2_draw = int(LANE_DRAW_TOP_RATIO * h)

        draw_line_ab(out, left_ab_s, y1, y2_draw, color=(0, 0, 255), thickness=LINE_THICKNESS)
        draw_line_ab(out, right_ab_s, y1, y2_draw, color=(0, 0, 255), thickness=LINE_THICKNESS)

        # 3) Steering (dùng look-ahead y2 = vùng bắt lane, KHÔNG dùng y2_draw)
        y2_steer = int(BOTTOM_Y_TOP_RATIO * h)
        steer_info = compute_steering(left_ab_s, right_ab_s, w, y1, y2_steer)

        # Centerline (blue)
        if steer_info is not None:
            xc1 = int(steer_info["xc1"])
            xc2 = int(steer_info["xc2"])
            cv2.line(out, (xc1, y1), (xc2, y2_steer), (255, 0, 0), 4)

        # HUD + debug
        nz = int(np.count_nonzero(mask_b))
        extra = f"gain={gain:.2f}  L_thr={l_thr}  maskNZ={nz}"
        draw_hud_box(out, steer_info, fps=fps, extra_text=extra)

        cv2.imshow("mask_bottom", mask_b)
        cv2.imshow("result", out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
