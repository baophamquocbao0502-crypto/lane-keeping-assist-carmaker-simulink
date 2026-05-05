import cv2
import numpy as np

VIDEO_PATH = "2_road_281225_sz_vid.mp4"

# ====== Tuning nhanh ======
BOTTOM_Y_TOP_RATIO = 0.55     # chỉ xử lý từ 55% chiều cao trở xuống
MIN_CONTOUR_AREA = 600        # bỏ nhiễu nhỏ
EMA_ALPHA = 0.25              # làm mượt đường (0.15..0.35)
LINE_THICKNESS = 8


def circular_mask(frame_bgr):
    """Mask vùng tròn để bỏ viền đen (IPG)."""
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    r = int(0.49 * w)
    cv2.circle(mask, (w // 2, h // 2), r, 255, -1)
    return cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)


def white_mask_hls(frame_bgr):
    """Tách vạch trắng bằng HLS."""
    hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(hls, (0, 165, 0), (180, 255, 70))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)
    return mask


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
    Fit line dạng x = a*y + b từ contour bằng cv2.fitLine.
    Trả (a, b) hoặc None.
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
    return (prev[0] * (1 - alpha) + new[0] * alpha,
            prev[1] * (1 - alpha) + new[1] * alpha)


def draw_line_ab(img, ab, y1, y2, color=(0, 255, 0), thickness=LINE_THICKNESS):
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

        # 1) Tiền xử lý: mask tròn + white mask
        frame_m = circular_mask(frame)
        mask = white_mask_hls(frame_m)

        # 2) Chỉ lấy nửa dưới
        mask_b, y_top = keep_bottom(mask, BOTTOM_Y_TOP_RATIO)

        # 3) Tìm contour
        contours, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        left_cnt = find_best_contour_by_side(contours, w, side="left")
        right_cnt = find_best_contour_by_side(contours, w, side="right")

        # 4) Fit line và làm mượt theo thời gian
        left_ab = fitline_to_ab(left_cnt)
        right_ab = fitline_to_ab(right_cnt)

        left_ab_s = ema_update(left_ab_s, left_ab)
        right_ab_s = ema_update(right_ab_s, right_ab)

        # 5) Vẽ lên ảnh gốc
        out = frame.copy()

        # (tuỳ chọn) vẽ contour xanh để debug
        if left_cnt is not None:
            cv2.drawContours(out, [left_cnt], -1, (0, 255, 0), 3)
        if right_cnt is not None:
            cv2.drawContours(out, [right_cnt], -1, (0, 255, 0), 3)

        # Vẽ 2 đường xanh kéo dài
        y1 = h
        y2 = int(BOTTOM_Y_TOP_RATIO * h)
        draw_line_ab(out, left_ab_s, y1, y2, (0, 255, 0), LINE_THICKNESS)
        draw_line_ab(out, right_ab_s, y1, y2, (0, 255, 0), LINE_THICKNESS)

        # debug: hiển thị vùng bắt đầu xử lý
        #cv2.line(out, (0, y2), (w, y2), (0, 255, 255), 2)

        cv2.imshow("mask_bottom", mask_b)
        cv2.imshow("result", out)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()




################import cv2
import numpy as np


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(0.5)) #y = mx + b
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 1e-3:

            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image,(x1, y1),(x2, y2),(255, 0, 0), 8)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(-11050, height), (1600, height), (490, 430)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

cap = cv2.VideoCapture("2_road_281225_sz_vid.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 70, np.array([]), minLineLength=10, maxLineGap=40)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result', combo_image)
    if cv2.waitKey(1) == ord('g'):
        break
cap.release()
cv2.destroyAllWindows()


#image = cv2.imread('121125.jpg')
#lane_image = np.copy(image)
#canny_image = canny(lane_image)
#cropped_image = region_of_interest(canny_image)
#lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 65, np.array([]), minLineLength=40, maxLineGap=100)
#averaged_lines = average_slope_intercept(lane_image, lines)
#line_image = display_lines(lane_image, lines)
#combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
#cv2.imshow('result', combo_image)
#cv2.waitKey(0)

#plt.imshow(canny)
#plt.show()
