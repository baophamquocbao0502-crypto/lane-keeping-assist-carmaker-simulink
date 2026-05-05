import cv2
import numpy as np
import math
import csv
import time

# =========================
# INPUT / OUTPUT
# =========================
VIDEO_PATH = "test_281225_new_1.mp4"
LOG_CSV_PATH = "lane_raw_log.csv"

# =========================
# Lane detection tuning
# =========================
BOTTOM_Y_TOP_RATIO = 0.55     # process from 55% height to bottom
MIN_CONTOUR_AREA = 600        # remove small noise contours
EMA_ALPHA = 0.25              # smoothing for lane lines (0.15..0.35)
LINE_THICKNESS = 8

# Measurement reference height (more stable than bottom-most)
Y_REF_RATIO = 0.85            # measure offset/lane width at 85% of image height


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
    """Exponential moving average for line params (a,b)."""
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


def compute_raw_measurements(left_ab, right_ab, w, y1, y2, y_ref):
    """
    Compute RAW measurements (no controller):
    - heading_deg_raw: heading angle of lane centerline (deg)
    - offset_px_raw: lateral offset of lane center at y_ref (px)
    - lane_width_px: lane width at y_ref (px)
    Returns dict, valid flag.
    """
    if left_ab is None or right_ab is None:
        return None, 0

    # lane center at y_ref for offset
    xl_ref = x_at_y(left_ab, y_ref)
    xr_ref = x_at_y(right_ab, y_ref)
    xc_ref = 0.5 * (xl_ref + xr_ref)

    offset_px_raw = float(xc_ref - (w / 2.0))
    lane_width_px = float(xr_ref - xl_ref)

    # heading from centerline between y1 and y2
    xl1, xr1 = x_at_y(left_ab, y1), x_at_y(right_ab, y1)
    xl2, xr2 = x_at_y(left_ab, y2), x_at_y(right_ab, y2)
    xc1 = 0.5 * (xl1 + xr1)
    xc2 = 0.5 * (xl2 + xr2)

    dx = float(xc1 - xc2)
    dy = float(y1 - y2)
    heading_deg_raw = math.degrees(math.atan2(dx, dy))

    return {
        "heading_deg_raw": heading_deg_raw,
        "offset_px_raw": offset_px_raw,
        "lane_width_px": lane_width_px,
        "xc1": xc1,
        "xc2": xc2,
        "xl_ref": xl_ref,
        "xr_ref": xr_ref,
        "xc_ref": xc_ref,
    }, 1


def draw_hud_box_raw(img, meas, valid, fps=None):
    """Draw a HUD box for RAW measurements."""
    x0, y0 = 20, 20
    box_w, box_h = 420, 170

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (30, 30, 30), -1)
    alpha = 0.55
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    tx = x0 + 15
    ty = y0 + 35
    line_h = 30

    cv2.putText(img, "RAW LANE MEASUREMENTS", (tx, ty), font, 0.8, (0, 255, 0), 2)
    ty += line_h + 5

    if not valid or meas is None:
        cv2.putText(img, "valid: 0 (lane missing)", (tx, ty), font, 0.8, (0, 0, 255), 2)
        if fps is not None:
            cv2.putText(img, f"FPS: {fps:.1f}", (x0 + box_w - 120, y0 + box_h - 10),
                        font, 0.6, (200, 200, 200), 1)
        return

    heading = meas["heading_deg_raw"]
    offset = meas["offset_px_raw"]
    width = meas["lane_width_px"]

    cv2.putText(img, f"valid: 1", (tx, ty), font, 0.75, (255, 255, 255), 2)
    ty += line_h

    cv2.putText(img, f"heading_raw: {heading:+.2f} deg", (tx, ty), font, 0.75, (255, 255, 255), 2)
    ty += line_h

    cv2.putText(img, f"offset_raw:  {offset:+.1f} px", (tx, ty), font, 0.75, (255, 255, 255), 2)
    ty += line_h

    cv2.putText(img, f"lane_width:  {width:.1f} px", (tx, ty), font, 0.75, (255, 255, 255), 2)

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

    # CSV logger
    f = open(LOG_CSV_PATH, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "t_s", "frame", "fps",
        "valid",
        "heading_deg_raw", "offset_px_raw", "lane_width_px",
        "xc_ref", "xl_ref", "xr_ref"
    ])
    t0 = time.time()
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            h, w = frame.shape[:2]
            y1 = h
            y2 = int(BOTTOM_Y_TOP_RATIO * h)
            y_ref = int(Y_REF_RATIO * h)

            # 1) Preprocess
            frame_m = circular_mask(frame)
            mask = white_mask_hls(frame_m)
            mask_b, _ = keep_bottom(mask, BOTTOM_Y_TOP_RATIO)

            # 2) Contours
            contours, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            left_cnt = find_best_contour_by_side(contours, w, side="left")
            right_cnt = find_best_contour_by_side(contours, w, side="right")

            # 3) Fit + smooth (measurement smoothing, not control)
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
            draw_line_ab(out, left_ab_s, y1, y2, (0, 255, 0), LINE_THICKNESS)
            draw_line_ab(out, right_ab_s, y1, y2, (0, 255, 0), LINE_THICKNESS)

            # 5) RAW measurements (NO controller)
            meas, valid = compute_raw_measurements(left_ab_s, right_ab_s, w, y1, y2, y_ref)

            # Draw centerline if valid
            if valid and meas is not None:
                xc1 = int(meas["xc1"])
                xc2 = int(meas["xc2"])
                cv2.line(out, (xc1, y1), (xc2, y2), (255, 0, 0), 4)  # centerline
                # reference center point at y_ref
                cv2.circle(out, (int(meas["xc_ref"]), y_ref), 6, (255, 0, 0), -1)

            # FPS
            t_now = cv2.getTickCount()
            dt = (t_now - t_prev) / tick_freq
            t_prev = t_now
            fps = (1.0 / dt) if dt > 1e-6 else None

            # 6) HUD
            draw_hud_box_raw(out, meas, valid, fps=fps)

            # 7) LOG to CSV (MATLAB-ready)
            t_s = time.time() - t0
            if valid and meas is not None:
                heading = meas["heading_deg_raw"]
                offset = meas["offset_px_raw"]
                width = meas["lane_width_px"]
                xc_ref = meas["xc_ref"]
                xl_ref = meas["xl_ref"]
                xr_ref = meas["xr_ref"]
            else:
                heading = float("nan")
                offset = float("nan")
                width = float("nan")
                xc_ref = float("nan")
                xl_ref = float("nan")
                xr_ref = float("nan")

            writer.writerow([
                t_s, frame_idx, (fps if fps is not None else float("nan")),
                valid,
                heading, offset, width,
                xc_ref, xl_ref, xr_ref
            ])
            frame_idx += 1

            # Display
            cv2.imshow("mask_bottom", mask_b)
            cv2.imshow("result", out)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        f.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Saved CSV log: {LOG_CSV_PATH}")


if __name__ == "__main__":
    main()
