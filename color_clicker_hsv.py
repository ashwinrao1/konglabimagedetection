import cv2
import numpy as np

video_path = "/Users/ashwinrao/konglabimagedetection/konglabimagedetection/Unsintered_1.wmv"
cap = cv2.VideoCapture(video_path)

clicked_pos = None
paused = False

def mouse_callback(event, x, y, flags, param):
    global clicked_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pos = (x, y)

cv2.namedWindow("Video Frame")
cv2.setMouseCallback("Video Frame", mouse_callback)

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
    display_frame = frame.copy()

    if clicked_pos:
        x, y = clicked_pos
        bgr = frame[y, x].astype(int)
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
        lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2Lab)[0, 0]
        cv2.circle(display_frame, (x, y), 5, (0, 255, 255), 2)
        text = f"BGR: {tuple(bgr)} | HSV: {tuple(hsv)} | Lab: {tuple(lab)}"
        cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Video Frame", display_frame)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('p'):  # Pause/resume
        paused = not paused
    elif key == 27:  # Esc to quit
        break

cap.release()
cv2.destroyAllWindows()
