import cv2
import numpy as np

# Function to convert RGB to HSV
def rgb_to_hsv(r, g, b):
    # Convert RGB to BGR as OpenCV uses BGR
    bgr = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[0][0]

def detect_color(frame, lower_bound1, upper_bound1, lower_bound2, upper_bound2):
    """Detects a specific color in a frame."""
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for the specified color ranges
    mask1 = cv2.inRange(hsv, lower_bound1, upper_bound1)  # For H < 17
    mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)  # For H > 150

    # Combine the two masks
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours in the combined mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected objects
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame, mask



# Define HSV bounds for the silver material
lower_silver1 = np.array([0, 0, 200])    # H low, S low, V high
upper_silver1 = np.array([20, 50, 255]) # H high, S moderate, V high

lower_silver2 = np.array([160, 0, 200])  # Alternative range for H > 150
upper_silver2 = np.array([180, 50, 255])


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
    vertical_start, vertical_end = (7 * height) // 21, (8 * height) // 20
    horizontal_start, horizontal_end = 0, (2 * width) // 3
    
    # Crop the frame
    cropped_frame = frame[vertical_start:vertical_end, horizontal_start:horizontal_end]
    #cropped_frame = frame
    # Detect color objects and get the mask
    result_frame, mask = detect_color(cropped_frame, lower_silver1, upper_silver1, lower_silver2, upper_silver2)
    # Apply mask to the frame
    
    # Show the original image
    cv2.imshow("Original", frame)
    cv2.moveWindow("Original", 50, 50)  # Move "Original" to (50, 50)

# Show the mask
    cv2.imshow("Mask", mask)
    cv2.moveWindow("Mask", 700, 50)  # Move "Mask" to (500, 50)



    # Break the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
