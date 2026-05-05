import cv2
import numpy as np

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 0.5)

    if abs(slope) < 1e-6:
        return None

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2], dtype=np.int32)

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        if x2 == x1:
            continue

        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)

        # bỏ line gần như ngang (nhiễu)
        if abs(slope) < 0.3:
            continue

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    lane_lines = []

    if len(left_fit) > 0:
        left_avg = np.mean(left_fit, axis=0)
        left_line = make_coordinates(image, left_avg)
        if left_line is not None:
            lane_lines.append(left_line)

    if len(right_fit) > 0:
        right_avg = np.mean(right_fit, axis=0)
        right_line = make_coordinates(image, right_avg)
        if right_line is not None:
            lane_lines.append(right_line)

    if len(lane_lines) == 0:
        return None

    return np.array(lane_lines, dtype=np.int32)

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # imread/video là BGR
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 20, 70)
    return edges

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(-100, height),
                          (1100, height),
                          (430, 230)]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 8)
    return line_image

# =========================
# MAIN: chạy trên video
# =========================
video_path = "171225_night.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Không mở được video: {video_path}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    edges = canny(frame)
    cropped = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        cropped,
        2,
        np.pi / 180,
        50,
        np.array([]),
        minLineLength=40,
        maxLineGap=180
    )

    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)

    combo = cv2.addWeighted(frame, 0.8, line_image, 1.0, 1.0)

    cv2.imshow("Lane detection (video)", combo)

    # Nhấn q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
