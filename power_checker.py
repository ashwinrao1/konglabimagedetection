import cv2
import numpy as np
import pandas as pd
from collections import deque
import os

# === CONFIGURATION ===
video_path = "/Users/ashwinrao/konglabimagedetection/konglabimagedetection/E652 - 20241202_181754.wmv"
output_csv = "sintering_quality_log.csv"
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

# --- ROI Definitions ---
# 1) Visualization ROI (for context on the full frame)
vis_x1, vis_y1, vis_x2, vis_y2 = 0, 330, 1280, 400

# 2) Processing ROI (a smaller strip *within* that visualization ROI)
#    Adjust these so that they tightly bracket just the sintering area under the probe.
proc_x1, proc_y1 = 150, vis_y1 + 10
proc_x2, proc_y2 = 350, vis_y2 - 10

# Make sure processing ROI is valid
if proc_x1 >= proc_x2 or proc_y1 >= proc_y2:
    mid_x = (vis_x1 + vis_x2) // 2
    proc_x1, proc_x2 = mid_x - 50, mid_x + 50
    proc_y1, proc_y2 = vis_y1 + 10, vis_y2 - 10

# === PARAMETERS ===
history_len           = 30
initial_chroma_thresh = 20
morph_iters           = 2
kernel                = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
clahe                 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# HSV bounds for that bright-flare background you wanted excluded
hsv_bg_lower = np.array([10, 40, 180])
hsv_bg_upper = np.array([20, 70, 255])

# === SETUP ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video at {video_path}")

history   = deque(maxlen=history_len)
frame_log = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    h, w = frame.shape[:2]

    # --- Clamp and extract visualization ROI ---
    vx1 = max(0, min(vis_x1, w-1))
    vy1 = max(0, min(vis_y1, h-1))
    vx2 = max(vx1+1,    min(vis_x2, w))
    vy2 = max(vy1+1,    min(vis_y2, h))
    # (We don't actually crop it, just draw the rectangle.)

    # --- Clamp and extract *processing* ROI ---
    px1 = max(0, min(proc_x1, w-1))
    py1 = max(0, min(proc_y1, h-1))
    px2 = max(px1+1,    min(proc_x2, w))
    py2 = max(py1+1,    min(proc_y2, h))
    proc_roi = frame[py1:py2, px1:px2]
    if proc_roi.size == 0:
        continue

    # --- Pre‐processing in Lab color space ---
    blur    = cv2.GaussianBlur(proc_roi, (3,3), 0)
    lab     = cv2.cvtColor(blur, cv2.COLOR_BGR2Lab)
    L, A, B = cv2.split(lab)
    L_eq    = clahe.apply(L)

    # 1) Otsu bright/dark split
    _, L_mask = cv2.threshold(
        L_eq, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # 2) neutral-color (achromatic) map
    chroma      = cv2.absdiff(A,128) + cv2.absdiff(B,128)
    _, C_mask   = cv2.threshold(
        chroma, initial_chroma_thresh, 255,
        cv2.THRESH_BINARY_INV
    )
    # 3) raw sintered candidate
    raw_sin = cv2.bitwise_and(L_mask, C_mask)

    # --- Exclude your bright-flare background via HSV ---
    hsv       = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    flare_mask = cv2.inRange(hsv, hsv_bg_lower, hsv_bg_upper)
    raw_sin    = cv2.bitwise_and(raw_sin, cv2.bitwise_not(flare_mask))

    # 4) Clean up mask
    sin_mask = cv2.morphologyEx(raw_sin, cv2.MORPH_CLOSE, kernel, iterations=morph_iters)
    sin_mask = cv2.morphologyEx(sin_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # --- Unsintered = ink area minus sintered area ---
    full_mask = np.full_like(sin_mask, 255)
    un_mask   = cv2.bitwise_and(full_mask, cv2.bitwise_not(sin_mask))
    un_mask   = cv2.morphologyEx(un_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iters)
    un_mask   = cv2.morphologyEx(un_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # --- Compute percent sintered on the small ROI ---
    proc_area = proc_roi.shape[0] * proc_roi.shape[1]
    sin_px    = cv2.countNonZero(sin_mask)
    pct       = 100.0 * sin_px / max(proc_area, 1)

    history.append(pct)
    avg_sp = sum(history) / len(history)

    # --- Decide quality ---
    if avg_sp > 80:
        quality, action = "Good",           "Keep power"
    elif avg_sp < 70:
        quality, action = "Under-sintered", "Increase power"
    else:
        quality, action = "Uncertain",      "Review manually"

    frame_log.append({
        "frame": frame_idx,
        "percent_sintered":        pct,
        "avg_percent_sintered":    avg_sp,
        "quality":                 quality,
        "action":                  action
    })

    # --- Visualization on the full frame ---
    out = frame.copy()

    # 1) draw the big (visualization) ROI
    cv2.rectangle(out, (vx1, vy1), (vx2, vy2), (255,255,0),  1)

    # 2) overlay colored regions on the small (processing) ROI
    overlay = proc_roi.copy()
    overlay[sin_mask > 0] = (0,255,0)   # green = sintered
    overlay[un_mask  > 0] = (0,0,255)   # red   = unsintered
    blended = cv2.addWeighted(proc_roi, 0.7, overlay, 0.3, 0)
    out[py1:py2, px1:px2] = blended

    # 3) outline the small ROI
    cv2.rectangle(out, (px1, py1), (px2, py2), (255,0,255), 2)

    # 4) status text
    cv2.putText(out, f"Frame {frame_idx}",               (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
    cv2.putText(out, f"{quality} (Avg: {avg_sp:.1f}%)",   (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(out, f"Action: {action}",                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Sintering Analysis (Full Frame)", out)

    # --- Intermediate debug windows (small ROI only) ---
    cv2.imshow("L_eq (Small ROI)",         L_eq)
    cv2.imshow("L_mask (Small ROI)",       L_mask)
    cv2.imshow("C_mask (Small ROI)",       C_mask)
    cv2.imshow("Final Sinter Mask (Small)", sin_mask)
    cv2.imshow("Final Unsinter Mask (Small)", un_mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# === CLEANUP & SAVE ===
cap.release()
cv2.destroyAllWindows()
pd.DataFrame(frame_log).to_csv(output_csv, index=False)
print(f"Saved log to {output_csv}")
