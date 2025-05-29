import cv2
import numpy as np
import pandas as pd
from collections import deque
import os

# === CONFIGURATION ===
video_path    = "/Users/ashwinrao/konglabimagedetection/konglabimagedetection/OriginalSintered.wmv"
output_csv    = "sintering_quality_log.csv"
output_dir    = "output_frames"
os.makedirs(output_dir, exist_ok=True)

# --- ROI Definitions ---
vis_x1, vis_y1, vis_x2, vis_y2 = 0, 330, 1280, 400
proc_x1, proc_y1 = 150, vis_y1 + 10
proc_x2, proc_y2 = 350, vis_y2 - 10
if proc_x1 >= proc_x2 or proc_y1 >= proc_y2:
    mid_x = (vis_x1 + vis_x2) // 2
    proc_x1, proc_x2 = mid_x - 50, mid_x + 50
    proc_y1, proc_y2 = vis_y1 + 10, vis_y2 - 10

# === PARAMETERS ===
history_len           = 30
initial_chroma_thresh = 20
morph_iters           = 2
kernel                = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
clahe                 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
hsv_bg_lower          = np.array([5, 40, 180])
hsv_bg_upper          = np.array([20, 70, 255])
min_area_big          = 500
min_area_small        = 100
temporal_M            = 5

# === SETUP ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video at {video_path}")

frame_log    = []
frame_idx    = 0
pct_history  = deque(maxlen=history_len)
mask_history = deque(maxlen=temporal_M)

# --- helper: area filtering ---
def area_filter(mask, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean

# --- compute masks with color refinement ---
def compute_masks(roi):
    blur   = cv2.GaussianBlur(roi, (5,5), 0)
    lab    = cv2.cvtColor(blur, cv2.COLOR_BGR2Lab)
    L, A, B = cv2.split(lab)

    # adaptive L channel threshold via Otsu
    L_eq  = clahe.apply(L)
    _, Lm = cv2.threshold(L_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # dynamic chroma threshold via Otsu
    chroma = cv2.absdiff(A,128) + cv2.absdiff(B,128)
    _, Cm  = cv2.threshold(chroma, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    raw = cv2.bitwise_and(Lm, Cm)

    # flare removal: HSV + brightness
    hsv   = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    flare_hsv = cv2.inRange(hsv, hsv_bg_lower, hsv_bg_upper)
    _, flare_v = cv2.threshold(hsv[:,:,2], 230, 255, cv2.THRESH_BINARY)
    flare = cv2.bitwise_or(flare_hsv, flare_v)
    raw &= cv2.bitwise_not(flare)

    # initial morphology for sintered mask
    sin = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=morph_iters)
    sin = cv2.morphologyEx(sin, cv2.MORPH_OPEN,  kernel, iterations=1)

    # unsintered mask is the inverse
    un  = cv2.bitwise_not(sin)
    un  = cv2.morphologyEx(un, cv2.MORPH_CLOSE, kernel, iterations=morph_iters)
    un  = cv2.morphologyEx(un, cv2.MORPH_OPEN,  kernel, iterations=1)

    # HSV-based color refinement: silver vs purple
    hsv2        = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    sin_hsv     = cv2.inRange(hsv2, np.array([0,0,200]), np.array([180,60,255]))
    purple_mask = cv2.inRange(hsv2, np.array([0,20,20]), np.array([150,255,150]))

    # combine: include silver regions, restrict unsintered to purple
    sin = cv2.bitwise_or(sin, sin_hsv)
    un  = cv2.bitwise_and(un, purple_mask)

    return sin, un

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    h, w = frame.shape[:2]
    bx1 = max(0, min(vis_x1, w-1)); bx2 = max(bx1+1, min(vis_x2, w))
    by1 = max(0, min(vis_y1, h-1)); by2 = max(by1+1, min(vis_y2, h))
    big_roi = frame[by1:by2, bx1:bx2]
    px1 = max(0, min(proc_x1, w-1)); px2 = max(px1+1, min(proc_x2, w))
    py1 = max(0, min(proc_y1, h-1)); py2 = max(py1+1, min(proc_y2, h))
    small_roi = frame[py1:py2, px1:px2]
    if big_roi.size==0 or small_roi.size==0:
        continue

    big_sin, big_un     = compute_masks(big_roi)
    small_sin, small_un = compute_masks(small_roi)

    # spatial filtering
    big_sin   = area_filter(big_sin,   min_area_big)
    small_sin = area_filter(small_sin, min_area_small)

    # temporal smoothing on big mask
    mask_history.append(big_sin)
    if len(mask_history) == temporal_M:
        avg_mask = np.mean(np.stack(mask_history, axis=0), axis=0)
        big_sin  = (avg_mask > 128).astype(np.uint8) * 255

    # decision logic on small ROI
    pct_sm   = 100.0 * cv2.countNonZero(small_sin) / max(small_sin.size, 1)
    pct_history.append(pct_sm)
    avg_pct  = sum(pct_history) / len(pct_history)
    if avg_pct>70:
        quality, action = ("Good","Keep power")
    elif avg_pct<40:
        quality, action = ("N/A","Keep power") 
    else:
        quality, action =("Under-sintered","Increase power")

    frame_log.append({
        "frame": frame_idx,
        "pct_small_roi": pct_sm,
        "avg_pct_small": avg_pct,
        "quality": quality,
        "action": action
    })

    # visualization
    out     = frame.copy()
    overlay = np.zeros_like(big_roi)
    overlay[:] = (100,100,100)
    overlay[big_sin>0] = (255,0,0)
    overlay[big_un>0] = (0,0,255)
    blended = cv2.addWeighted(big_roi, 0.7, overlay, 0.3, 0)
    out[by1:by2, bx1:bx2] = blended

    cv2.rectangle(out, (bx1,by1),(bx2,by2),(255,255,0),1)
    cv2.rectangle(out, (px1,py1),(px2,py2),(255,0,255),2)
    cv2.putText(out, f"Frame {frame_idx}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2)
    cv2.putText(out, f"{quality} (Avg: {avg_pct:.1f}%)",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    cv2.putText(out, f"Action: {action}",(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    cv2.imshow("Sintering Analysis (Full Frame)",out)
    cv2.imshow("Small ROI Sinter Mask", small_sin)
    cv2.imshow("Small ROI Unsinter Mask", small_un)

    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
pd.DataFrame(frame_log).to_csv(output_csv, index=False)
print(f"Saved to {output_csv}")
