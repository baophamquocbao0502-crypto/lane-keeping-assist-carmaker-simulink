import cv2
import numpy as np

VIDEO_PATH = "2_road_sz_vid_night.mp4"
# VIDEO_PATH = "test_281225_new_1.mp4"

# ====== Night tuning (giữ cấu trúc cũ) ======
BOTTOM_Y_TOP_RATIO = 0.45      # night: giữ nhiều vùng đường hơn (dùng cho mask/contour)
MIN_CONTOUR_AREA = 120         # night: contour nhỏ hơn
EMA_ALPHA = 0.25
LINE_THICKNESS = 8

# ====== VẼ NGẮN LẠI ======
LANE_DRAW_TOP_RATIO = 0.70     # vẽ từ đáy lên tới 70% chiều cao (ngắn hơn). Thử 0.65..0.80

# ====== Auto brightness (quan trọng nhất) ======
GAIN_PCTL = 95
GAIN_TARGET = 180
GAIN_MAX = 6.0

# ====== CLAHE (tăng tương phản) ======
USE_CLAHE = True
CLAHE_CLIPLIMIT = 3.0
CLAHE_TILE = (8, 8)

# ====== Adaptive HLS ======
L_PCTL = 85
L_MIN_FLOOR = 90
S_MAX = 190

# ====== ROI trapezoid ======
ROI_Y_TOP_RATIO = 0.55
ROI_X_LEFT_BOT  = 0.06
ROI_X_RIGHT_BOT = 0.94
ROI_X_LEFT_TOP  = 0.36
ROI_X_RIGHT_TOP = 0.64


def circular_mask(frame_bgr):
    """Mask vùng tròn để bỏ viền đen (IPG)."""
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    r = int(0.49 * w)
    cv2.circle(mask, (w // 2, h // 2), r, 255, -1)
    return cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)


def trapezoid_roi_mask(h, w):
    """ROI hình thang tập trung vùng mặt đường."""
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
    """
    Auto exposure đơn giản:
    - dùng kênh V (HSV)
    - scale sao cho percentile (GAIN_PCTL) đạt GAIN_TARGET
    """
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
    """
    Tách vạch trắng bằng HLS cho night:
    - L-threshold lấy theo percentile ở vùng đáy (adaptive)
    """
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
    margin = int(0.06 * img_w)  # avoid circular border

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
    """Fit line dạng x = a*y + b từ contour bằng cv2.fitLine."""
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
    """Vẽ line màu ĐỎ (BGR = 0,0,255)."""
    if ab is None:
        return
    a, b = ab
    x1 = int(a * y1 + b)
    x2 = int(a * y2 + b)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    left_ab_s = None
    right_ab_s = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w = frame.shape[:2]

        # 1) Mask tròn
        frame_m = circular_mask(frame)

        # 2) Auto-gain tăng sáng (quan trọng)
        frame_g, gain = auto_gain_bgr(frame_m)

        # 3) CLAHE tăng tương phản (tùy chọn)
        if USE_CLAHE:
            frame_g = clahe_on_l_channel_bgr(frame_g)

        # 4) ROI hình thang để tập trung vùng đường
        roi_mask = trapezoid_roi_mask(h, w)
        frame_roi = cv2.bitwise_and(frame_g, frame_g, mask=roi_mask)

        # 5) White mask HLS adaptive
        mask, l_thr = white_mask_hls_adaptive(frame_roi)

        # 6) Lấy nửa dưới
        mask_b, _ = keep_bottom(mask, BOTTOM_Y_TOP_RATIO)

        # 7) Contours
        contours, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        left_cnt = find_best_contour_by_side(contours, w, side="left")
        right_cnt = find_best_contour_by_side(contours, w, side="right")

        # 8) Fit + EMA
        left_ab = fitline_to_ab(left_cnt)
        right_ab = fitline_to_ab(right_cnt)
        left_ab_s = ema_update(left_ab_s, left_ab)
        right_ab_s = ema_update(right_ab_s, right_ab)

        # 9) Vẽ kết quả (lane màu đỏ, NGẮN LẠI)
        out = frame.copy()

        y1 = h - 1
        y2_draw = int(LANE_DRAW_TOP_RATIO * h)  # <-- điểm trên để vẽ ngắn hơn

        draw_line_ab(out, left_ab_s, y1, y2_draw, color=(0, 0, 255), thickness=LINE_THICKNESS)
        draw_line_ab(out, right_ab_s, y1, y2_draw, color=(0, 0, 255), thickness=LINE_THICKNESS)

        # Debug overlay
        nz = int(np.count_nonzero(mask_b))
        cv2.putText(out, f"gain={gain:.2f}  L_thr={l_thr}  maskNZ={nz}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("mask_bottom", mask_b)
        cv2.imshow("result", out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
