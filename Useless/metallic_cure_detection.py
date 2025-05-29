import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from collections import deque

# Parameters
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
TEMPORAL_WINDOW = 3
PERSISTENCE_THRESHOLD = 1
DENSITY_THRESHOLD = 0.15
GRID_ROWS = 5
GRID_COLS = 5
ROI = (330, 0, 400, 450)  # Adjust depending on your video resolution

frame_history = deque(maxlen=TEMPORAL_WINDOW)
persistence_map = None
previous_frame = None

def compute_adaptive_threshold(gray):
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    return mean_val + 2 * std_val

def get_bright_regions(gray_frame, threshold):
    _, mask = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
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
    persistence_map[persistence_map > TEMPORAL_WINDOW] = TEMPORAL_WINDOW

def filter_by_persistence():
    return (persistence_map >= PERSISTENCE_THRESHOLD).astype(np.uint8) * 255

def draw_density_overlay(cured_mask, roi_frame):
    tile_height = cured_mask.shape[0] // GRID_ROWS
    tile_width = cured_mask.shape[1] // GRID_COLS
    overlay = roi_frame.copy()

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            y1 = row * tile_height
            y2 = (row + 1) * tile_height
            x1 = col * tile_width
            x2 = (col + 1) * tile_width

            tile = cured_mask[y1:y2, x1:x2]
            density = np.count_nonzero(tile) / (tile_height * tile_width)

            color = (0, 255, 0) if density > DENSITY_THRESHOLD else (0, 0, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, f"{density:.2f}", (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return overlay

def process_frame(frame):
    global previous_frame

    y1, x1, y2, x2 = ROI
    roi_frame = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

    adaptive_thresh = compute_adaptive_threshold(gray)
    bright_mask = get_bright_regions(gray, adaptive_thresh)

    if previous_frame is not None:
        frame_diff = cv2.absdiff(gray, previous_frame)
        _, diff_mask = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)
        bright_mask = cv2.bitwise_and(bright_mask, diff_mask)

    previous_frame = gray.copy()

    lbp = compute_lbp(gray)
    texture_mask = cv2.inRange(lbp, 10, 150)
    combined_mask = cv2.bitwise_and(bright_mask, texture_mask)

    frame_history.append(combined_mask)
    if len(frame_history) == TEMPORAL_WINDOW:
        update_persistence_map(combined_mask)
        cured_mask = filter_by_persistence()
    else:
        cured_mask = np.zeros_like(bright_mask)

    # Output 1: Green mask on ROI
    cured_overlay = roi_frame.copy()
    cured_overlay[cured_mask == 255] = [0, 255, 0]

    # Output 2: Tile-based density overlay
    density_overlay = draw_density_overlay(cured_mask, roi_frame)

    return cured_overlay, density_overlay

def run_video_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cured_overlay, density_overlay = process_frame(frame)

        #
        cv2.imshow('Cured Mask', cured_overlay)
        cv2.imshow('Density Overlay', density_overlay)
        #cv2.imshow('Density Overlay', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_file = "/Users/ashwinrao/konglabimagedetection/konglabimagedetection/E652 - 20241202_181754.wmv"  # Replace with the path to your video file
    run_video_detection(video_file)

