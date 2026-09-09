import os
import sys
import pickle
import time
from collections import deque, Counter
import numpy as np  # type: ignore
import cv2  # type: ignore
import mediapipe as mp  # type: ignore

# Ensure parent root directory is in sys.path for direct script execution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from src.utils import mp_hands, draw_styled_landmarks, extract_landmarks  # type: ignore
except Exception:
    from utils import mp_hands, draw_styled_landmarks, extract_landmarks  # type: ignore

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sign_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


class SignLanguagePredictor:
    """
    Real-time Sign Language Prediction & Sentence Assembly Engine.
    Employs temporal prediction buffering and confidence thresholding.
    """
    def __init__(self, confidence_threshold=0.65, buffer_size=12, cooldown_sec=1.0):
        self.confidence_threshold = confidence_threshold
        self.buffer = deque(maxlen=buffer_size)
        self.cooldown_sec = cooldown_sec
        self.last_prediction_time = 0
        self.last_confirmed_sign = None

        self.model = None
        self.label_encoder = None
        self.is_loaded = False
        self.load_model()

        # Sentence State
        self.current_sentence = ""
        self.active_sign = "Waiting..."
        self.active_confidence = 0.0

    def load_model(self):
        """Loads trained classification model and label encoder."""
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
                with open(ENCODER_PATH, "rb") as f:
                    self.label_encoder = pickle.load(f)
                self.is_loaded = True
                print("[Sign Predictor] Successfully loaded model and label encoder.")
            except Exception as e:
                print(f"[Sign Predictor Error] Failed to load model: {e}")
                self.is_loaded = False
        else:
            print("[Sign Predictor Warning] Model files not found. Please train model first.")
            self.is_loaded = False

    def process_frame(self, results):
        """
        Processes hand landmarks from MediaPipe results and updates sentence assembly.
        
        Args:
            results: MediaPipe process results object.
            
        Returns:
            tuple: (active_sign, active_confidence, current_sentence)
        """
        if not self.is_loaded:
            return "Model Not Loaded", 0.0, self.current_sentence

        # Extract normalized feature vector
        features = extract_landmarks(results)
        hand_present = np.any(features != 0)

        if not hand_present:
            self.buffer.clear()
            self.active_sign = "No Hand Detected"
            self.active_confidence = 0.0
            return self.active_sign, self.active_confidence, self.current_sentence

        # Run Model Inference
        if self.model is None or self.label_encoder is None:
            return "Model Not Loaded", 0.0, self.current_sentence

        features = features.reshape(1, -1)
        probabilities = self.model.predict_proba(features)[0]
        max_idx = np.argmax(probabilities)
        confidence = probabilities[max_idx]
        predicted_sign = self.label_encoder.inverse_transform([max_idx])[0]

        # Add to rolling buffer
        if confidence >= self.confidence_threshold:
            self.buffer.append(predicted_sign)
        else:
            self.buffer.append("Uncertain")

        self.active_sign = predicted_sign
        self.active_confidence = float(confidence)

        # Confirm Gesture using temporal voting
        buffer_maxlen = self.buffer.maxlen or len(self.buffer)
        if len(self.buffer) == buffer_maxlen:
            counts = Counter(self.buffer)
            most_common_sign, frequency = counts.most_common(1)[0]
            
            # Requires at least 70% agreement in buffer window
            if frequency >= int(0.7 * buffer_maxlen) and most_common_sign != "Uncertain":
                now = time.time()
                # Apply cooldown to prevent duplicate auto-repeat
                if (most_common_sign != self.last_confirmed_sign) or (now - self.last_prediction_time > self.cooldown_sec):
                    self.append_to_sentence(most_common_sign)
                    self.last_confirmed_sign = most_common_sign
                    self.last_prediction_time = now

        return self.active_sign, self.active_confidence, self.current_sentence

    def process_features(self, features: np.ndarray):
        """
        Processes a raw 126-length anatomical feature vector and updates sentence assembly.
        """
        if not self.is_loaded or self.model is None or self.label_encoder is None:
            return "Model Not Loaded", 0.0, self.current_sentence

        hand_present = np.any(features != 0)
        if not hand_present:
            self.buffer.clear()
            self.active_sign = "No Hand Detected"
            self.active_confidence = 0.0
            return self.active_sign, self.active_confidence, self.current_sentence

        features = features.reshape(1, -1)
        probabilities = self.model.predict_proba(features)[0]
        max_idx = np.argmax(probabilities)
        confidence = probabilities[max_idx]
        predicted_sign = self.label_encoder.inverse_transform([max_idx])[0]

        if confidence >= self.confidence_threshold:
            self.buffer.append(predicted_sign)
        else:
            self.buffer.append("Uncertain")

        self.active_sign = predicted_sign
        self.active_confidence = float(confidence)

        buffer_maxlen = self.buffer.maxlen or len(self.buffer)
        if len(self.buffer) == buffer_maxlen:
            counts = Counter(self.buffer)
            most_common_sign, frequency = counts.most_common(1)[0]
            
            if frequency >= int(0.7 * buffer_maxlen) and most_common_sign != "Uncertain":
                now = time.time()
                if (most_common_sign != self.last_confirmed_sign) or (now - self.last_prediction_time > self.cooldown_sec):
                    self.append_to_sentence(most_common_sign)
                    self.last_confirmed_sign = most_common_sign
                    self.last_prediction_time = now

        return self.active_sign, self.active_confidence, self.current_sentence

    def append_to_sentence(self, sign: str):
        """Appends confirmed sign gesture to live constructed sentence."""
        sign_clean = sign.strip().lower()
        
        if sign_clean == "space":
            if not self.current_sentence.endswith(" "):
                self.current_sentence += " "
        elif sign_clean == "delete":
            if len(self.current_sentence) > 0:
                self.current_sentence = self.current_sentence[:-1]
        elif sign_clean in ['hello', 'thank_you', 'yes', 'no', 'please']:
            # Word gesture: insert word directly formatted nicely
            formatted_word = sign_clean.replace("_", " ").capitalize()
            if self.current_sentence and not self.current_sentence.endswith(" "):
                self.current_sentence += " "
            self.current_sentence += formatted_word + " "
        else:
            # Alphabet character
            self.current_sentence += sign.upper()

    def clear_sentence(self):
        """Resets the current sentence."""
        self.current_sentence = ""

    def delete_last_char(self):
        """Deletes the last character from current sentence."""
        if len(self.current_sentence) > 0:
            self.current_sentence = self.current_sentence[:-1]
