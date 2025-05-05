import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from collections import deque

# Parameters
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
TEMPORAL_WINDOW = 3
PERSISTENCE_THRESHOLD = 1  # How many times a pixel must be bright in a row to count as cured
ROI = (330, 0, 400, 1280)

frame_history = deque(maxlen=TEMPORAL_WINDOW)
persistence_map = None
previous_frame = None

def compute_adaptive_threshold(gray):
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    return mean_val + 2 * std_val

def get_bright_regions(gray_frame, threshold):
    _, mask = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    # Morphological filtering to remove noise
    kernel = np.ones((2, 2), np.uint8)
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return clean_mask

def compute_lbp(gray_frame):
    lbp = local_binary_pattern(gray_frame, LBP_POINTS, LBP_RADIUS, method='uniform')
    lbp_normalized = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return lbp_normalized

def update_persistence_map(bright_mask):
    global persistence_map
    if persistence_map is None:
        persistence_map = np.zeros_like(bright_mask, dtype=np.uint8)
    
    persistence_map = cv2.add(persistence_map, bright_mask // 255)
    persistence_map[persistence_map > TEMPORAL_WINDOW] = TEMPORAL_WINDOW  # Cap values

def filter_by_persistence():
    return (persistence_map >= PERSISTENCE_THRESHOLD).astype(np.uint8) * 255

def process_frame(frame):
    global previous_frame

    # Crop to ROI (adjust these indices as needed)
    y1, x1, y2, x2 = ROI
    roi_frame = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold
    adaptive_thresh = compute_adaptive_threshold(gray)
    bright_mask = get_bright_regions(gray, adaptive_thresh)

    # Frame differencing to suppress global flashes
    if previous_frame is not None:
        frame_diff = cv2.absdiff(gray, previous_frame)
        _, diff_mask = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)
        bright_mask = cv2.bitwise_and(bright_mask, diff_mask)

    previous_frame = gray.copy()

    # LBP texture filtering
    lbp = compute_lbp(gray)
    texture_mask = cv2.inRange(lbp, 10, 150)  # heuristic for roughness
    combined_mask = cv2.bitwise_and(bright_mask, texture_mask)

    # Temporal persistence
    frame_history.append(combined_mask)
    if len(frame_history) == TEMPORAL_WINDOW:
        update_persistence_map(combined_mask)
        cured_mask = filter_by_persistence()
    else:
        cured_mask = np.zeros_like(bright_mask)

    # Overlay result
    result = roi_frame.copy()
    result[cured_mask == 255] = [0, 255, 0]  # Green overlay for cured

    # Place ROI result back into original frame
    output = frame.copy()
    output[y1:y2, x1:x2] = result
    return output

def run_video_detection(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = process_frame(frame)
        cv2.imshow('Cured Detection', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_file= "/Users/ashwinrao/konglabimagedetection/konglabimagedetection/E652 - 20241202_181754.wmv"  # Replace with the path to your video file
    run_video_detection(video_file)
