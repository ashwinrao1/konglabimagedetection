import cv2
import numpy as np
# Function to convert RGB to HSV
def rgb_to_hsv(r, g, b):
    # Convert RGB to BGR as OpenCV uses BGR
    bgr = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[0][0]

def detect_color(frame, lower_bound, upper_bound):
    """Detects a specific color in a frame."""
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the specified color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected objects
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame, mask


lower_silver = np.array([0, 0, 180])  # Light color (low saturation, high value)
upper_silver = np.array([179, 30, 255])  # Light color range

# Path to the video file
video_path = "/Users/ashwinrao/konglabimagedetection/konglabimagedetection/E652 - 20241202_181754.wmv"  # Replace with the path to your video file

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video.")
        break

    # Crop the frame (e.g., retain only the center region)
    height, width, _ = frame.shape
  # Define crop ranges for left 2/3 horizontally and middle 1/3 vertically
    vertical_start, vertical_end = (7*height)//21, (8*height)//20
    horizontal_start, horizontal_end = 0, (2 * width) // 3

   # vertical_start, vertical_end = 0, height
   # horizontal_start, horizontal_end = 0, width

    # Crop the frame
    cropped_frame = frame[vertical_start:vertical_end, horizontal_start:horizontal_end]

    # Detect blue objects and get the mask
    result_frame, mask = detect_color(cropped_frame, lower_silver, upper_silver)

     # Display the original video
    #cv2.imshow('Original Video', frame)

    # Display the cropped processed feed with bounding boxes
    cv2.imshow('Processed Feed (Color Detection)', result_frame)

    # Display the mask
    cv2.imshow('Mask', mask)


    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
