import cv2
import numpy as np
import pandas as pd
from collections import deque
import os
import time

# …


# === CONFIGURATION ===
video_path    = "/Users/ashwinrao/konglabimagedetection/konglabimagedetection/EnhancedOriginalSintered.wmv"
output_csv    = "sintering_quality_log.csv"
output_dir    = "output_frames"
os.makedirs(output_dir, exist_ok=True)

# === PARAMETERS ===
history_len           = 30
morph_iters           = 2
kernel                = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
clahe                 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
hsv_bg_lower          = np.array([5, 40, 180])
hsv_bg_upper          = np.array([20, 70, 255])
min_area_big          = 500
min_area_small        = 100
temporal_M            = 5
presence_thresh_small = 50
desired_fps    = 20
display_delay  = int(1000 / desired_fps)   # in milliseconds

# === HELPERS ===
def area_filter(mask, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean

def compute_masks(roi):
    blur    = cv2.GaussianBlur(roi, (5,5), 0)
    lab     = cv2.cvtColor(blur, cv2.COLOR_BGR2Lab)
    L, A, B = cv2.split(lab)

    # Otsu thresholds
    L_eq   = clahe.apply(L)
    _, Lm  = cv2.threshold(L_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    chroma = cv2.absdiff(A,128) + cv2.absdiff(B,128)
    _, Cm  = cv2.threshold(chroma, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    raw    = cv2.bitwise_and(Lm, Cm)

    # remove flare
    hsv       = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    flare_hsv = cv2.inRange(hsv, hsv_bg_lower, hsv_bg_upper)
    _, fv     = cv2.threshold(hsv[:,:,2], 230, 255, cv2.THRESH_BINARY)
    raw      &= cv2.bitwise_not(cv2.bitwise_or(flare_hsv, fv))

    # morphology
    sin = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=morph_iters)
    sin = cv2.morphologyEx(sin, cv2.MORPH_OPEN,  kernel, iterations=1)
    un  = cv2.bitwise_not(sin)
    un  = cv2.morphologyEx(un, cv2.MORPH_CLOSE, kernel, iterations=morph_iters)
    un  = cv2.morphologyEx(un, cv2.MORPH_OPEN,  kernel, iterations=1)

    # color refinement
    hsv2        = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    sin_hsv     = cv2.inRange(hsv2, np.array([0,0,200]), np.array([180,60,255]))
    purple_mask = cv2.inRange(hsv2, np.array([0,20,20]), np.array([150,255,150]))
    sin = cv2.bitwise_or(sin, sin_hsv)
    un  = cv2.bitwise_and(un, purple_mask)

    return sin, un

# interactive ROI selection
points = []
clone = None
def on_mouse(event, x, y, flags, param):
    global clone, points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(clone, (x, y), 4, (0,255,0), -1)
        if len(points) in (2,4):
            tl = points[-2]
            br = points[-1]
            color = (255,0,0) if len(points)==2 else (0,0,255)
            cv2.rectangle(clone, tl, br, color, 2)
        cv2.imshow("Select ROIs", clone)

def select_rois(frame):
    global clone, points
    clone = frame.copy()
    points = []
    cv2.namedWindow("Select ROIs")
    cv2.setMouseCallback("Select ROIs", on_mouse)
    print("Click TL/BR for BIG ROI, then TL/BR for SMALL ROI. Press ESC to cancel.")
    cv2.imshow("Select ROIs", clone)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if len(points) == 4 or key == 27:
            break
    cv2.destroyWindow("Select ROIs")
    if len(points) < 4:
        print("ROI reset cancelled.")
        return None
    return (points[0], points[1], points[2], points[3])

# === INITIAL SETUP ===
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    raise IOError("Cannot read video.")
roi = select_rois(frame)
if roi is None:
    cap.release(); cv2.destroyAllWindows(); exit(0)
(vis_x1,vis_y1),(vis_x2,vis_y2),(proc_x1,proc_y1),(proc_x2,proc_y2) = roi

frame_log = []
pct_hist  = deque(maxlen=history_len)
mask_hist = deque(maxlen=temporal_M)
frame_idx = 0
# === PROCESS FRAMES ===
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # allow ROI reset
    disp = frame.copy()
    key = cv2.waitKey(display_delay) & 0xFF
    # optional extra sleep for more precise timing:
    # time.sleep(max(0, (1/desired_fps) - 0.001))

    if key in (ord('r'), ord('R')):
        new = select_rois(frame)
        if new:
            (vis_x1,vis_y1),(vis_x2,vis_y2),(proc_x1,proc_y1),(proc_x2,proc_y2) = new
            mask_hist.clear()    # ← clear old masks of the wrong shape
            pct_hist.clear()     # ← clear old pct history if you want fresh averages
        disp = frame.copy()

    big_roi   = frame[vis_y1:vis_y2, vis_x1:vis_x2]
    small_roi = frame[proc_y1:proc_y2, proc_x1:proc_x2]
    if big_roi.size == 0 or small_roi.size == 0:
        continue

    big_sin, big_un     = compute_masks(big_roi)
    small_sin, small_un = compute_masks(small_roi)
    big_sin   = area_filter(big_sin,   min_area_big)
    small_sin = area_filter(small_sin, min_area_small)

    mask_hist.append(big_sin)
    if len(mask_hist) == temporal_M:
        avg = np.mean(np.stack(mask_hist,0),0)
        big_sin = (avg > 128).astype(np.uint8) * 255

    # ─── new decision logic ───────────────────────────────────────────────
    presence = cv2.countNonZero(cv2.bitwise_or(small_sin, small_un))
    if presence < presence_thresh_small:
        quality, action = "Waiting", "Ink not in ROI yet"
        avg_pct = 0.0
        uns_pct = 0.0
    else:
        pct     = 100.0 * cv2.countNonZero(small_sin)   / small_sin.size
        uns_pct = 100.0 * cv2.countNonZero(small_un)    / small_un.size
        pct_hist.append(pct)
        avg_pct = sum(pct_hist) / len(pct_hist)

        if uns_pct > 60.0:
            quality, action = "Not sintered", "Increase power"
        elif avg_pct > 70.0:
            quality, action = "Good",          "Keep power"
        elif avg_pct < 60.0 and avg_pct >= 40.0:
            quality, action = "Under-sintered", "Increase power"
        elif avg_pct <40.0:
            quality, action = "N/A",           "Keep power"
    # ──────────────────────────────────────────────────────────────────────

    frame_log.append({
        "frame":    frame_idx,
        "quality":  quality,
        "action":   action,
        "presence": presence,
        "uns_pct":  uns_pct,
        "avg_pct":  avg_pct
    })

    # visualization (unchanged) …
    overlay = np.zeros_like(big_roi)
    overlay[:] = (100,100,100)
    overlay[big_sin>0] = (255,0,0)
    overlay[big_un>0]  = (0,0,255)
    blended = cv2.addWeighted(big_roi,0.7,overlay,0.3,0)
    disp[vis_y1:vis_y2,vis_x1:vis_x2] = blended

    cv2.rectangle(disp,(vis_x1,vis_y1),(vis_x2,vis_y2),(255,255,0),1)
    cv2.rectangle(disp,(proc_x1,proc_y1),(proc_x2,proc_y2),(255,0,255),2)
    cv2.putText(disp, f"Frame {frame_idx}",        (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2)
    cv2.putText(disp, f"{quality} (Avg: {avg_pct:.1f}%) (Un: {uns_pct:.1f}%)",(10,60), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    cv2.putText(disp, f"Action: {action}",         (10,90), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    cv2.imshow("Sintering Analysis", disp)
    if key == 27:
        break


# cleanup
cap.release()
cv2.destroyAllWindows()

# save log
pd.DataFrame(frame_log).to_csv(output_csv, index=False)
print(f"Saved to {output_csv}")
