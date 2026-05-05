import cv2
import numpy as np


# =========================================================
# 1) TÍNH TOẠ ĐỘ TỪ SLOPE & INTERCEPT
# =========================================================
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]            # đáy ảnh
    y2 = int(y1 * 0.5)             # khoảng giữa ảnh
    if abs(slope) < 1e-6:
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


# =========================================================
# 2) TỐI ƯU average_slope_intercept
# =========================================================
def average_slope_intercept(image, lines, min_slope=0.3, min_length=20):
    """
    Gom các đoạn Hough thành 1 đường trái + 1 đường phải (trung bình có trọng số).
    - image: frame gốc (H x W x 3)
    - lines: output từ HoughLinesP
    - min_slope: bỏ các đoạn gần như nằm ngang (|slope| quá nhỏ)
    - min_length: bỏ các đoạn quá ngắn (nhiễu)
    """
    if lines is None:
        return None, None

    h, w, _ = image.shape

    # Đưa về dạng (N, 4): [x1, y1, x2, y2]
    segs = lines.reshape(-1, 4).astype(np.float32)
    x1 = segs[:, 0]
    y1 = segs[:, 1]
    x2 = segs[:, 2]
    y2 = segs[:, 3]

    dx = x2 - x1
    dy = y2 - y1

    # tránh chia 0
    dx[dx == 0] = 1e-6

    slopes = dy / dx
    intercepts = y1 - slopes * x1
    lengths = np.hypot(dx, dy)  # độ dài đoạn thẳng

    # bỏ đoạn quá ngang hoặc quá ngắn
    valid = (np.abs(slopes) > min_slope) & (lengths > min_length)
    if not np.any(valid):
        return None, None

    slopes = slopes[valid]
    intercepts = intercepts[valid]
    lengths = lengths[valid]
    mid_x = (x1[valid] + x2[valid]) / 2.0

    # phân loại trái/phải theo slope & vị trí trung điểm so với tâm ảnh
    center_x = w / 2.0
    left_mask  = (slopes < 0) & (mid_x < center_x)
    right_mask = (slopes > 0) & (mid_x > center_x)

    left_line = None
    right_line = None

    # ---- Lane trái ----
    if np.any(left_mask):
        s_left  = slopes[left_mask]
        b_left  = intercepts[left_mask]
        w_left  = lengths[left_mask]  # trọng số theo độ dài
        slope_l = np.average(s_left, weights=w_left)
        inter_l = np.average(b_left, weights=w_left)
        left_line = make_coordinates(image, (slope_l, inter_l))

    # ---- Lane phải ----
    if np.any(right_mask):
        s_right = slopes[right_mask]
        b_right = intercepts[right_mask]
        w_right = lengths[right_mask]
        slope_r = np.average(s_right, weights=w_right)
        inter_r = np.average(b_right, weights=w_right)
        right_line = make_coordinates(image, (slope_r, inter_r))

    return left_line, right_line


# =========================================================
# 3) CANNY
# =========================================================
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)


# =========================================================
# 4) VẼ LINE
# =========================================================
def display_lines(image, lines):
    line_img = np.zeros_like(image)
    for line in lines:
        if line is None:
            continue
        x1, y1, x2, y2 = line
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 8)
    return line_img


# =========================================================
# 5) ROI
# =========================================================
def region_of_interest(image):
    h = image.shape[0]
    polygons = np.array([
        [(-100, h), (1100, h), (430, 230)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)


# =========================================================
# 6) TÍNH GÓC LÁI
# =========================================================
def compute_steering_angle(image, left_line, right_line):
    h, w, _ = image.shape
    img_center = w / 2

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
    return np.degrees(angle_rad)


# =========================================================
# 7) MAIN – VIDEO
# =========================================================
cap = cv2.VideoCapture("171225_morning.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    canny_img = canny(frame)
    cropped = region_of_interest(canny_img)

    lines = cv2.HoughLinesP(
        cropped,
        rho=2,
        theta=np.pi / 180,
        threshold=65,
        minLineLength=40,
        maxLineGap=100
    )

    left_line, right_line = average_slope_intercept(frame, lines)
    line_img = display_lines(frame, [left_line, right_line])

    angle = compute_steering_angle(frame, left_line, right_line)

    combo = cv2.addWeighted(frame, 0.8, line_img, 1, 1)

    # =====================================================
    # 🔲 KHUNG THÔNG TIN GÓC LÁI – NHỎ GỌN, FONT ĐẸP
    # =====================================================
    text = f"Steering Angle: {angle:.1f} deg"
    font = cv2.FONT_HERSHEY_DUPLEX     # font “nhìn pro” hơn
    scale = 0.7                        # nhỏ hơn
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    x, y = 20, 40           # vị trí text (trên trái)
    pad_x = 6               # padding nhỏ
    pad_y = 6

    # nền tối (xám đậm)
    cv2.rectangle(
        combo,
        (x - pad_x, y - th - pad_y),
        (x + tw + pad_x, y + pad_y),
        (20, 20, 20),
        -1
    )

    # viền mảnh màu trắng
    cv2.rectangle(
        combo,
        (x - pad_x, y - th - pad_y),
        (x + tw + pad_x, y + pad_y),
        (255, 255, 255),
        1
    )

    # text màu xanh lá
    cv2.putText(
        combo,
        text,
        (x, y),
        font,
        scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA
    )

    cv2.imshow("result", combo)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# =========================================================
# 3) CANNY
# =========================================================
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)


# =========================================================
# 4) VẼ LINE
# =========================================================
def display_lines(image, lines):
    line_img = np.zeros_like(image)
    for line in lines:
        if line is None:
            continue
        x1, y1, x2, y2 = line
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 8)
    return line_img


# =========================================================
# 5) ROI
# =========================================================
def region_of_interest(image):
    h = image.shape[0]
    polygons = np.array([
        [(-100, h), (1100, h), (430, 230)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)


# =========================================================
# 6) TÍNH GÓC LÁI
# =========================================================
def compute_steering_angle(image, left_line, right_line):
    h, w, _ = image.shape
    img_center = w / 2

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
    return np.degrees(angle_rad)


# =========================================================
# 7) MAIN – VIDEO
# =========================================================
cap = cv2.VideoCapture("131125.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    canny_img = canny(frame)
    cropped = region_of_interest(canny_img)

    lines = cv2.HoughLinesP(
        cropped, 2, np.pi/180, 65,
        minLineLength=40,
        maxLineGap=100
    )

    left_line, right_line = average_slope_intercept(frame, lines)
    line_img = display_lines(frame, [left_line, right_line])

    angle = compute_steering_angle(frame, left_line, right_line)

    combo = cv2.addWeighted(frame, 0.8, line_img, 1, 1)

    # =====================================================
    # 🔲 VẼ KHUNG VUÔNG CHO STEERING ANGLE
    # =====================================================
    text = f"Steering Angle: {angle:.1f} deg"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = 30, 50

    # nền đen
    cv2.rectangle(combo,
                  (x - 10, y - th - 10),
                  (x + tw + 10, y + 10),
                  (0, 0, 0), -1)

    # viền trắng
    cv2.rectangle(combo,
                  (x - 10, y - th - 10),
                  (x + tw + 10, y + 10),
                  (255, 255, 255), 2)

    # text
    cv2.putText(combo, text,
                (x, y),
                font, scale,
                (0, 255, 0), thickness,
                cv2.LINE_AA)

    cv2.imshow("result", combo)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
