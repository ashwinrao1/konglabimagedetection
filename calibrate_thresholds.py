# calibrate_thresholds.py

import cv2
import numpy as np

# === EDIT TO YOUR PATH & ROI ===
video_path = "/Users/ashwinrao/konglabimagedetection/konglabimagedetection/E652 - 20241202_181754.wmv"
proc_roi_x1, proc_roi_y1, proc_roi_x2, proc_roi_y2 = 80, 330, 400, 400

# Read one frame
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {video_path}")
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("Failed to read frame")

# Crop to your processing ROI
roi = frame[proc_roi_y1:proc_roi_y2, proc_roi_x1:proc_roi_x2]
lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
L_eq = clahe.apply(lab[:,:,0])
chroma = cv2.absdiff(lab[:,:,1], 128) + cv2.absdiff(lab[:,:,2], 128)

# Trackbar callback (does nothing)
def nothing(x): pass

cv2.namedWindow("Tune Masks", cv2.WINDOW_NORMAL)
cv2.createTrackbar("L ink",   "Tune Masks", 20,  255, nothing)
cv2.createTrackbar("L sint",  "Tune Masks", 120, 255, nothing)
cv2.createTrackbar("Chroma",  "Tune Masks", 20,  255, nothing)

while True:
    l_ink  = cv2.getTrackbarPos("L ink",  "Tune Masks")
    l_sint = cv2.getTrackbarPos("L sint", "Tune Masks")
    chrom  = cv2.getTrackbarPos("Chroma", "Tune Masks")

    # 1) ink mask: dark pixels
    _, ink_mask = cv2.threshold(L_eq, l_ink, 255, cv2.THRESH_BINARY_INV)
    # 2) bright spots = potential sintered
    _, bright    = cv2.threshold(L_eq, l_sint, 255, cv2.THRESH_BINARY)
    # 3) neutral chroma = low color
    _, neutral   = cv2.threshold(chroma, chrom, 255, cv2.THRESH_BINARY_INV)
    # 4) final sintered = bright & neutral & ink
    sinter_mask  = cv2.bitwise_and(bright, neutral)
    sinter_mask  = cv2.bitwise_and(sinter_mask, ink_mask)

    # stack for display
    disp = np.hstack([
        cv2.cvtColor(ink_mask,   cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(bright,     cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(neutral,    cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(sinter_mask,cv2.COLOR_GRAY2BGR),
    ])
    cv2.imshow("Tune Masks", disp)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()
print("Final thresholds:")
print(f"  L_threshold_for_ink_detection = {l_ink}")
print(f"  L_threshold_for_sintering     = {l_sint}")
print(f"  chroma_threshold              = {chrom}")
