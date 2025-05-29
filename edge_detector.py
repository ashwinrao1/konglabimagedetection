import cv2
import numpy as np

# Video path
video_path = "/Users/ashwinrao/konglabimagedetection/konglabimagedetection/E652 - 20241202_181754.wmv"
cap = cv2.VideoCapture(video_path)

# ROI coordinates (x1, y1, x2, y2)
x1, y1, x2, y2 = 0, 330, 1280, 400

# CLAHE and morphology setup
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
kernel = np.ones((3, 3), np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Show full original frame
    cv2.imshow("Full Frame", frame)

    # Crop ROI
    roi = frame[y1:y2, x1:x2]

    # Grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # CLAHE
    contrast = clahe.apply(gray)

    # Bilateral filter
    denoised = cv2.bilateralFilter(contrast, d=9, sigmaColor=75, sigmaSpace=75)

    # Sobel edges
    sobelx = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(np.clip(sobel, 0, 255))

    # Threshold and clean
    _, edge_mask = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)
    edge_clean = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
    edge_bgr = cv2.cvtColor(edge_clean, cv2.COLOR_GRAY2BGR)

    # Show the edge mask ROI in a separate window
    cv2.imshow("ROI Edge Mask", edge_bgr)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
