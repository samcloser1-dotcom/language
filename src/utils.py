import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import threading
import queue
import numpy as np  # type: ignore
import cv2  # type: ignore
try:
    import pyttsx3  # type: ignore
except Exception:
    pyttsx3 = None
import mediapipe as mp  # type: ignore

# MediaPipe Hand landmarks visualization solution
mp_hands = mp.solutions.hands  # type: ignore[attr-defined]
mp_drawing = mp.solutions.drawing_utils  # type: ignore[attr-defined]
mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore[attr-defined]

# Standard ASL Alphabet + Common Words & Special Gestures
LABEL_LIST = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'hello', 'thank_you', 'yes', 'no', 'please',
    'space', 'delete'
]


def extract_landmarks(results):
    """
    Extracts scale, rotation, and camera-invariant anatomical hand features from MediaPipe results.
    Includes relative coordinates, finger extension ratios, tip distances, and bone angles.
    
    Args:
        results: MediaPipe Hands process result object.
        
    Returns:
        np.ndarray: Flattened array of length 126 (2 hands * 63 features).
    """
    landmark_vector = np.zeros(126, dtype=np.float32)
    
    if not results or not results.multi_hand_landmarks:
        return landmark_vector

    # Process up to 2 hands
    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        if hand_idx >= 2:
            break
            
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
        
        # 1. Wrist Origin Normalization (Landmark 0)
        wrist = coords[0]
        coords_rel = coords - wrist
        
        # 2. Anatomical Palm Size Scaling (Wrist to Index MCP - Landmark 5)
        palm_size = np.linalg.norm(coords[5] - coords[0])
        if palm_size < 1e-5:
            palm_size = np.max(np.linalg.norm(coords_rel, axis=1))
            
        if palm_size > 1e-6:
            coords_norm = coords_rel / palm_size
        else:
            coords_norm = coords_rel

        # 3. 42 Relative 2D (x, y) coordinates
        norm_xy = coords_norm[:, :2].flatten()  # 42 values

        # 4. 5 Finger Extension Ratios (Tip distance to Wrist / MCP distance to Wrist)
        tips = [4, 8, 12, 16, 20]
        mcps = [2, 5, 9, 13, 17]
        ext_ratios = []
        for tip_i, mcp_i in zip(tips, mcps):
            tip_dist = np.linalg.norm(coords_norm[tip_i])
            mcp_dist = np.linalg.norm(coords_norm[mcp_i])
            ratio = tip_dist / (mcp_dist + 1e-5)
            ext_ratios.append(ratio)

        # 5. 5 Inter-finger Tip Distances
        tip_dists = [
            np.linalg.norm(coords_norm[8] - coords_norm[4]),   # Index - Thumb
            np.linalg.norm(coords_norm[8] - coords_norm[12]),  # Index - Middle
            np.linalg.norm(coords_norm[12] - coords_norm[16]), # Middle - Ring
            np.linalg.norm(coords_norm[16] - coords_norm[20]), # Ring - Pinky
            np.linalg.norm(coords_norm[4] - coords_norm[20]),  # Thumb - Pinky
        ]

        # 6. 4 Vector Angles (between adjacent finger vectors)
        v_thumb = coords_norm[4]
        v_index = coords_norm[8]
        v_middle = coords_norm[12]
        v_ring = coords_norm[16]
        v_pinky = coords_norm[20]

        def get_angle(v1, v2):
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-5 or n2 < 1e-5:
                return 0.0
            cos_a = np.dot(v1, v2) / (n1 * n2)
            return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))

        angles = [
            get_angle(v_thumb, v_index),
            get_angle(v_index, v_middle),
            get_angle(v_middle, v_ring),
            get_angle(v_ring, v_pinky),
        ]

        # Combine into 56 anatomical feature vector per hand, zero-pad to 63
        hand_feat = np.zeros(63, dtype=np.float32)
        combined = np.concatenate([norm_xy, ext_ratios, tip_dists, angles])
        hand_feat[:len(combined)] = combined

        start_idx = hand_idx * 63
        landmark_vector[start_idx:start_idx + 63] = hand_feat

    return landmark_vector


def draw_styled_landmarks(image, results):
    """
    Draws custom styled hand landmarks and skeletal connections on an OpenCV image frame.
    
    Args:
        image (np.ndarray): BGR image frame from OpenCV.
        results: MediaPipe Hands process result object.
    """
    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=2, circle_radius=2)
            )


class TextToSpeechManager:
    """
    Thread-safe Text-to-Speech manager using pyttsx3.
    Executes speech requests asynchronously in a worker thread.
    """
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._speech_loop, daemon=True)
        self.worker_thread.start()

    def _speech_loop(self):
        """Worker loop executing speech requests sequentially."""
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass

        if pyttsx3 is not None:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)    # Speaking speed
                engine.setProperty('volume', 1.0)  # Volume level
            except Exception as e:
                print(f"[TTS Warning] Could not initialize pyttsx3 engine: {e}")
                engine = None
        else:
            engine = None

        while True:
            text = self.speech_queue.get()
            if text is None:
                break
            if engine and text.strip():
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as err:
                    print(f"[TTS Error] Speech execution failed: {err}")
            self.speech_queue.task_done()

    def speak(self, text: str):
        """Enqueue text for spoken output without blocking UI."""
        if text and text.strip():
            self.speech_queue.put(text)

