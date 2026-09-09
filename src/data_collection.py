import os
import sys
import csv
import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np  # type: ignore

# Ensure parent root directory is in sys.path for direct script execution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from src.utils import mp_hands, draw_styled_landmarks, extract_landmarks, LABEL_LIST  # type: ignore
except Exception:
    from utils import mp_hands, draw_styled_landmarks, extract_landmarks, LABEL_LIST  # type: ignore

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "hand_landmarks.csv")


def ensure_csv_header():
    """Ensures the dataset CSV file exists with proper headers."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        headers = ["label"] + [f"feat_{i}" for i in range(126)]
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def get_sample_counts():
    """Reads CSV file and returns sample count dictionary for each label."""
    counts = {label: 0 for label in LABEL_LIST}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row and row[0] in counts:
                    counts[row[0]] += 1
    return counts


def save_landmark_sample(label: str, landmark_vector: np.ndarray):
    """Appends a single landmark sample vector to the dataset CSV file."""
    ensure_csv_header()
    row = [label] + landmark_vector.tolist()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def run_data_collection(cam_index=0):
    """
    Launches interactive webcam loop for collecting sign language landmark data.
    
    Controls:
    - 'n' or 'p': Next / Previous gesture label
    - 'SPACE': Record single frame landmark sample
    - 'c': Toggle continuous recording mode (10 samples with slight delay)
    - 'q' or 'ESC': Quit data collection module
    """
    ensure_csv_header()
    counts = get_sample_counts()
    
    current_label_idx = 0
    cap = cv2.VideoCapture(cam_index)
    
    if not cap.isOpened():
        print(f"[Error] Camera device index {cam_index} could not be opened.")
        return

    print("\n" + "="*60)
    print("      REAL-TIME SIGN LANGUAGE DATA COLLECTION MODULE")
    print("="*60)
    print("Controls:")
    print("  'n' / 'p' : Switch to Next / Previous sign label")
    print("  SPACE     : Save current hand frame sample")
    print("  'c'       : Capture burst of 15 continuous samples")
    print("  'q' / ESC : Exit data collection\n")

    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=2
    ) as hands:
        
        continuous_burst = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[Warning] Blank frame grabbed from webcam.")
                continue

            # Flip image horizontally for intuitive mirror view
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Convert BGR to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Draw skeletal hand landmarks on display frame
            draw_styled_landmarks(frame, results)
            
            # Extract normalized feature vector
            landmark_vector = extract_landmarks(results)
            hand_detected = np.any(landmark_vector != 0)
            
            target_label = LABEL_LIST[current_label_idx]
            
            # Handle continuous burst recording
            if continuous_burst > 0 and hand_detected:
                save_landmark_sample(target_label, landmark_vector)
                counts[target_label] += 1
                continuous_burst -= 1
                cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 6)

            # Draw HUD Overlay
            cv2.rectangle(frame, (0, 0), (w, 90), (25, 25, 25), -1)
            
            label_text = f"Target Label [{current_label_idx + 1}/{len(LABEL_LIST)}]: '{target_label}'"
            count_text = f"Collected Samples: {counts[target_label]}"
            status_text = "HAND DETECTED" if hand_detected else "NO HAND DETECTED"
            status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            
            cv2.putText(frame, label_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, count_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(frame, status_text, (w - 220, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

            cv2.imshow("Sign Language Data Collector", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('n'):
                current_label_idx = (current_label_idx + 1) % len(LABEL_LIST)
            elif key == ord('p'):
                current_label_idx = (current_label_idx - 1) % len(LABEL_LIST)
            elif key == 32:  # SPACE bar
                if hand_detected:
                    save_landmark_sample(target_label, landmark_vector)
                    counts[target_label] += 1
                    print(f"Recorded sample #{counts[target_label]} for sign '{target_label}'")
                else:
                    print("[Warning] Cannot save sample: No hand detected in frame!")
            elif key == ord('c'):
                if hand_detected:
                    continuous_burst = 15
                    print(f"Recording burst of 15 samples for sign '{target_label}'...")
                else:
                    print("[Warning] Cannot start burst: No hand detected!")

    cap.release()
    cv2.destroyAllWindows()
    print("\nData collection finished.")


if __name__ == "__main__":
    run_data_collection()
