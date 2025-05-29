import cv2
import numpy as np

# --- Configuration for Enhancements ---
# Set to True to enable, False to disable individual effects
APPLY_BRIGHTNESS_CONTRAST = True
APPLY_SATURATION = True
APPLY_SHARPENING = True
APPLY_NOISE_REDUCTION = True  # Choose a method below

# --- Enhancement Parameters (Adjust these values) ---
# Brightness and Contrast
brightness_value = 10      # Integer: Adjusts brightness (-255 to 255). Positive brighter, negative darker.
contrast_value = 1.2       # Float: Adjusts contrast (e.g., 1.0 for no change, 1.5 for higher contrast).

# Saturation
saturation_factor = 1.5    # Float: Adjusts color saturation (1.0 for no change, >1 for more saturation, <1 for less).

# Sharpening
# Kernel for sharpening. The sum of elements should ideally be 1 if you want to maintain overall brightness,
# or you can use kernels that might slightly brighten/darken. This one sums to 1.
sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
# Alternative stronger sharpening kernel (sums to 1):
# sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])


# Noise Reduction
noise_reduction_method = "bilateral"  # Options: "gaussian" or "bilateral"
# Gaussian Blur parameters (if method is "gaussian")
gaussian_ksize = (5, 5)          # Tuple: Kernel size (must be odd numbers, e.g., (3,3), (5,5)). Larger means more blur.
# Bilateral Filter parameters (if method is "bilateral") - better at preserving edges but slower
bilateral_d = 7                  # Integer: Diameter of each pixel neighborhood.
bilateral_sigma_color = 50       # Integer: Filter sigma in the color space.
bilateral_sigma_space = 50       # Integer: Filter sigma in the coordinate space.


def enhance_frame(frame):
    """Applies selected enhancements to a single video frame."""
    enhanced = frame.copy()

    if APPLY_BRIGHTNESS_CONTRAST:
        # Adjust brightness and contrast
        # The formula is: output_pixel = contrast_value * input_pixel + brightness_value
        enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast_value, beta=brightness_value)

    if APPLY_SATURATION:
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Multiply saturation channel by the factor
        # Ensure 's' is float for multiplication, then clip and convert back to uint8
        s = s.astype(np.float32)
        s = np.multiply(s, saturation_factor)
        s = np.clip(s, 0, 255)  # Values must be between 0 and 255
        s = s.astype(np.uint8)

        final_hsv = cv2.merge((h, s, v))
        enhanced = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    if APPLY_NOISE_REDUCTION:
        # It's generally better to apply noise reduction before sharpening
        if noise_reduction_method == "gaussian":
            enhanced = cv2.GaussianBlur(enhanced, gaussian_ksize, 0)
        elif noise_reduction_method == "bilateral":
            enhanced = cv2.bilateralFilter(enhanced, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)

    if APPLY_SHARPENING:
        enhanced = cv2.filter2D(enhanced, -1, sharpen_kernel)

    return enhanced

def main():
    # --- Video Source ---
    # To use a webcam, use 0 (default), 1, 2, etc.
    # To use a video file, replace 0 with the file path, e.g., "my_video.mp4"
    video_source = '/Users/ashwinrao/konglabimagedetection/konglabimagedetection/OriginalSintered.wmv'
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'.")
        print("If using a webcam, ensure it's connected and not in use by another application.")
        print("If using a file, ensure the path is correct.")
        return

    # Set window properties
    cv2.namedWindow("Original Frame", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Enhanced Frame", cv2.WINDOW_NORMAL)
    # You can resize these windows as needed

    print("\nStarting Real-Time Video Enhancer...")
    print("Press 'q' in a video window to quit.")
    print("You can enable/disable effects and change parameters at the top of this script.")
    if isinstance(video_source, str):
        print(f"Processing video file: {video_source}")
    else:
        print(f"Using webcam index: {video_source}")


    while True:
        ret, frame = cap.read()

        if not ret:
            if isinstance(video_source, str):  # End of video file
                print("Reached end of video file.")
                # Optional: Loop the video
                # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # continue # Uncomment to loop
                break # Exit if not looping
            else:  # Error reading from webcam
                print("Error: Could not read frame from webcam. Exiting.")
                break

        # --- Apply Enhancements ---
        enhanced_frame = enhance_frame(frame)

        # --- Display Frames ---
        # Display original and enhanced frames in separate windows
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Enhanced Frame", enhanced_frame)

        # To display side-by-side in one window (frames must have same height):
        # combined_display = np.hstack((frame, enhanced_frame))
        # cv2.imshow("Original vs Enhanced", combined_display)


        # --- User Input ---
        key = cv2.waitKey(1) & 0xFF  # Wait for 1ms, get key press

        if key == ord('q'):  # If 'q' is pressed, quit
            print("Quitting...")
            break
        # You could add more key bindings here to toggle effects dynamically
        # For example:
        # elif key == ord('s'):
        #     APPLY_SHARPENING = not APPLY_SHARPENING
        #     print(f"Sharpening {'enabled' if APPLY_SHARPENING else 'disabled'}")


    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing finished and windows closed.")

if __name__ == "__main__":
    # Before running, make sure you have OpenCV and NumPy installed:
    # pip install opencv-python numpy
    main()