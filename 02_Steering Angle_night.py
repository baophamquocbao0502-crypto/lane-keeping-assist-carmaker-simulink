import cv2
import numpy as np

# =========================================================
# 1) Tạo line dài từ (slope, intercept)
# =========================================================
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    h = image.shape[0]
    y1 = h
    y2 = int(h * 0.5)

    if abs(slope) < 1e-6:
        return None

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2], dtype=np.int32)

# =========================================================
# 2) Gom Hough segments -> lane trái + lane phải
# =========================================================
def average_slope_intercept(image, lines):
    if lines is None:
        return None, None

    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        if x2 == x1:
            continue

        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)

        # bỏ line gần ngang (nhiễu)
        if abs(slope) < 0.3:
            continue

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_line = None
    right_line = None

    if len(left_fit) > 0:
        left_avg = np.mean(left_fit, axis=0)
        left_line = make_coordinates(image, left_avg)

    if len(right_fit) > 0:
        right_avg = np.mean(right_fit, axis=0)
        right_line = make_coordinates(image, right_avg)

    return left_line, right_line

# =========================================================
# 3) Canny
# =========================================================
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # video/imread là BGR
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 20, 70)
    return edges

# =========================================================
# 4) ROI
# =========================================================
def region_of_interest(image):
    h = image.shape[0]
    polygons = np.array([[(-100, h),
                          (1100, h),
                          (430, 230)]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

# =========================================================
# 5) Vẽ line
# =========================================================
def display_lines(image, left_line, right_line):
    line_image = np.zeros_like(image)

    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 8)

    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 8)

    return line_image

# =========================================================
# 6) Tính steering angle (độ)
# =========================================================
def compute_steering_angle(image, left_line, right_line):
    h, w = image.shape[:2]
    img_center = w / 2.0

    if left_line is not None and right_line is not None:
        lane_center = (left_line[0] + right_line[0]) / 2.0
    elif left_line is not None:
        lane_center = left_line[0] + 0.5 * w
    elif right_line is not None:
        lane_center = right_line[0] - 0.5 * w
    else:
        return 0.0

    offset = img_center - lane_center
    angle_rad = np.arctan2(offset, 0.7 * h)
    return float(np.degrees(angle_rad))

# =========================================================
# 7) MAIN – chạy video
# =========================================================
def main():
    video_path = "171225_night.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        edges = canny(frame)
        roi = region_of_interest(edges)

        lines = cv2.HoughLinesP(
            roi,
            rho=2,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=40,
            maxLineGap=180
        )

        left_line, right_line = average_slope_intercept(frame, lines)
        steering_angle = compute_steering_angle(frame, left_line, right_line)

        line_img = display_lines(frame, left_line, right_line)
        combo = cv2.addWeighted(frame, 0.8, line_img, 1.0, 1.0)

        cv2.putText(combo, f"Steering Angle: {steering_angle:+.1f} deg",
                    (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                    (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Lane detection (video)", combo)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
