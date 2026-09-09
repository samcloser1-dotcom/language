import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import cv2  # type: ignore
from PIL import Image, ImageTk  # type: ignore
import mediapipe as mp  # type: ignore

# Add parent directory to sys.path for clean package imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from src.utils import mp_hands, draw_styled_landmarks, TextToSpeechManager  # type: ignore
    from src.real_time_detection import SignLanguagePredictor  # type: ignore
    from src.train_model import train_sign_model  # type: ignore
    from create_sample_data import generate_sample_dataset  # type: ignore
except Exception:
    from utils import mp_hands, draw_styled_landmarks, TextToSpeechManager  # type: ignore
    from real_time_detection import SignLanguagePredictor  # type: ignore
    from train_model import train_sign_model  # type: ignore
    from create_sample_data import generate_sample_dataset  # type: ignore

# Safe Pillow Resampling Enum for all Pillow versions
try:
    LANCZOS_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS_FILTER = Image.LANCZOS  # type: ignore[attr-defined]

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Sign Language Detection & Translation System")
        self.root.geometry("1200x820")
        self.root.configure(bg="#121824")
        
        # Application State
        self.cap = None
        self.is_running = False
        self.predictor = SignLanguagePredictor()
        self.tts = TextToSpeechManager()
        self.hands = None
        
        # Build Design & UI Widgets
        self._setup_styles()
        self._build_header()
        self._build_main_layout()
        
        # Initialize MediaPipe Hands
        try:
            self.hands = mp_hands.Hands(
                model_complexity=1,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.65,
                max_num_hands=2
            )
        except Exception as e:
            print(f"[Warning] MediaPipe Hands init fallback: {e}")
            self.hands = mp_hands.Hands(max_num_hands=2)

    def _setup_styles(self):
        """Configures custom dark theme colors and styles."""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Dark Theme Colors
        self.bg_color = "#121824"
        self.card_bg = "#1e2638"
        self.accent_color = "#00e5ff"
        self.btn_bg = "#2a354d"
        self.text_color = "#ffffff"

        self.style.configure(".", background=self.bg_color, foreground=self.text_color)
        self.style.configure("Card.TFrame", background=self.card_bg, relief="flat")
        self.style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"), background=self.bg_color, foreground=self.accent_color)
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 11), background=self.bg_color, foreground="#a0aec0")
        
        # Buttons
        self.style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), background="#00c853", foreground="#ffffff")
        self.style.map("Primary.TButton", background=[("active", "#00e676")])

        self.style.configure("Danger.TButton", font=("Segoe UI", 10, "bold"), background="#d50000", foreground="#ffffff")
        self.style.map("Danger.TButton", background=[("active", "#ff1744")])

        self.style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), background="#2979ff", foreground="#ffffff")
        self.style.map("Action.TButton", background=[("active", "#448aff")])

        self.style.configure("Secondary.TButton", font=("Segoe UI", 10), background="#37474f", foreground="#ffffff")

    def _build_header(self):
        header_frame = tk.Frame(self.root, bg=self.bg_color, pady=12, padx=20)
        header_frame.pack(fill="x")
        
        title_label = ttk.Label(header_frame, text="🤟 Real-Time Sign Language Translator", style="Header.TLabel")
        title_label.pack(side="left")
        
        subtitle_label = ttk.Label(header_frame, text="MediaPipe + Deep Neural Network Classifier", style="SubHeader.TLabel")
        subtitle_label.pack(side="right")

    def _build_main_layout(self):
        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(fill="both", expand=True, padx=15, pady=10)

        # Left Column: Video Viewport
        left_frame = tk.Frame(main_container, bg=self.card_bg, highlightbackground="#2a354d", highlightthickness=1)
        left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        video_header = tk.Label(left_frame, text="Live Camera Feed & Landmark Mesh", font=("Segoe UI", 12, "bold"), bg=self.card_bg, fg="#00e5ff")
        video_header.pack(anchor="w", padx=15, pady=10)

        self.video_label = tk.Label(left_frame, bg="#0a0d14")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=5)

        # Right Column: Dashboard Controls & Translation Panel
        right_frame = tk.Frame(main_container, bg=self.bg_color, width=420)
        right_frame.pack(side="right", fill="both", padx=10)
        right_frame.pack_propagate(False)

        # Active Sign Display Card
        sign_card = tk.Frame(right_frame, bg=self.card_bg, highlightbackground="#2a354d", highlightthickness=1, pady=15, padx=15)
        sign_card.pack(fill="x", pady=(0, 10))

        tk.Label(sign_card, text="CURRENT DETECTED SIGN", font=("Segoe UI", 9, "bold"), bg=self.card_bg, fg="#a0aec0").pack(anchor="w")
        self.sign_display = tk.Label(sign_card, text="Waiting...", font=("Segoe UI", 26, "bold"), bg=self.card_bg, fg="#00e5ff")
        self.sign_display.pack(anchor="w", pady=5)

        # Confidence Bar
        conf_frame = tk.Frame(sign_card, bg=self.card_bg)
        conf_frame.pack(fill="x", pady=5)
        
        self.conf_label = tk.Label(conf_frame, text="Confidence: 0.0%", font=("Segoe UI", 9), bg=self.card_bg, fg="#ffffff")
        self.conf_label.pack(side="left")
        
        self.conf_progress = ttk.Progressbar(conf_frame, orient="horizontal", mode="determinate")
        self.conf_progress.pack(side="right", fill="x", expand=True, padx=(10, 0))

        # Sentence Translation Panel
        sentence_card = tk.Frame(right_frame, bg=self.card_bg, highlightbackground="#2a354d", highlightthickness=1, pady=15, padx=15)
        sentence_card.pack(fill="both", expand=True, pady=10)

        tk.Label(sentence_card, text="CONSTRUCTED SENTENCE TRANSCRIPT", font=("Segoe UI", 9, "bold"), bg=self.card_bg, fg="#a0aec0").pack(anchor="w")
        
        self.sentence_text = tk.Text(
            sentence_card, font=("Segoe UI", 14), bg="#0a0d14", fg="#ffffff",
            insertbackground="white", wrap="word", relief="flat", height=6
        )
        self.sentence_text.pack(fill="both", expand=True, pady=10)

        # Action Buttons Grid
        btn_frame1 = tk.Frame(sentence_card, bg=self.card_bg)
        btn_frame1.pack(fill="x", pady=5)

        self.btn_speak = ttk.Button(btn_frame1, text="🔊 Speak Text", style="Action.TButton", command=self._on_speak)
        self.btn_speak.pack(side="left", fill="x", expand=True, padx=2)

        self.btn_delete = ttk.Button(btn_frame1, text="⌫ Delete Last", style="Secondary.TButton", command=self._on_delete)
        self.btn_delete.pack(side="left", fill="x", expand=True, padx=2)

        self.btn_clear = ttk.Button(btn_frame1, text="🗑 Clear", style="Secondary.TButton", command=self._on_clear)
        self.btn_clear.pack(side="left", fill="x", expand=True, padx=2)

        # Camera & Model Control Section
        ctrl_card = tk.Frame(right_frame, bg=self.card_bg, highlightbackground="#2a354d", highlightthickness=1, pady=15, padx=15)
        ctrl_card.pack(fill="x", pady=(10, 0))

        tk.Label(ctrl_card, text="SYSTEM CONTROLS", font=("Segoe UI", 9, "bold"), bg=self.card_bg, fg="#a0aec0").pack(anchor="w")

        btn_frame2 = tk.Frame(ctrl_card, bg=self.card_bg)
        btn_frame2.pack(fill="x", pady=8)

        self.btn_start = ttk.Button(btn_frame2, text="▶ Start Camera", style="Primary.TButton", command=self.start_camera)
        self.btn_start.pack(side="left", fill="x", expand=True, padx=2)

        self.btn_stop = ttk.Button(btn_frame2, text="⏹ Stop Camera", style="Danger.TButton", command=self.stop_camera, state="disabled")
        self.btn_stop.pack(side="left", fill="x", expand=True, padx=2)

        btn_frame3 = tk.Frame(ctrl_card, bg=self.card_bg)
        btn_frame3.pack(fill="x", pady=4)

        self.btn_sample = ttk.Button(btn_frame3, text="⚡ Generate Sample Dataset", style="Secondary.TButton", command=self._on_generate_samples)
        self.btn_sample.pack(fill="x", pady=2)

        self.btn_train = ttk.Button(btn_frame3, text="🧠 Train Model", style="Secondary.TButton", command=self._on_train_model)
        self.btn_train.pack(fill="x", pady=2)

        # Status Footer
        self.status_label = tk.Label(self.root, text="System Ready", font=("Segoe UI", 9), bg="#0a0d14", fg="#a0aec0", anchor="w", padx=15, pady=4)
        self.status_label.pack(fill="x", side="bottom")

    def log_status(self, message: str):
        """Updates the status footer bar."""
        self.status_label.config(text=f"Status: {message}")
        print(f"[GUI Log] {message}")

    def start_camera(self):
        """Starts OpenCV camera capture thread."""
        if self.is_running:
            return
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam (Device Index 0). Please verify camera connection.")
            return

        self.is_running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.log_status("Camera stream active.")

        # Start Video Thread
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

    def stop_camera(self):
        """Stops webcam streaming loop."""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.video_label.config(image="")
        self.log_status("Camera stream stopped.")

    def _video_loop(self):
        """Continuous video frame processing and UI update loop."""
        # Initialize thread-local MediaPipe Hands detector
        try:
            thread_hands = mp_hands.Hands(
                model_complexity=1,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.65,
                max_num_hands=2
            )
        except Exception:
            thread_hands = mp_hands.Hands(max_num_hands=2)

        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process MediaPipe Hand Landmarks
            results = thread_hands.process(rgb_frame)

            # Draw Skeletal Hand Mesh
            draw_styled_landmarks(frame, results)

            # Process Real-Time Prediction & Sentence Assembly
            sign, conf, sentence = self.predictor.process_frame(results)

            # Update Frame Visual Overlay
            cv2.rectangle(frame, (10, 10), (320, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"Sign: {sign}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 229, 255), 2)

            # Convert BGR frame to PIL Image for main thread rendering
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.resize((640, 480), LANCZOS_FILTER)

            # Pass PIL Image object to main thread callback (PhotoImage MUST be built on main thread)
            self.root.after(0, self._update_ui_elements, img, sign, conf, sentence)
            
            time.sleep(0.01)

        thread_hands.close()

    def _update_ui_elements(self, img, sign, conf, sentence):
        """Thread-safe callback to update GUI controls on main thread."""
        if not self.is_running:
            return

        # Instantiate PhotoImage on main thread to prevent Tcl/Tk thread crashes
        img_tk = ImageTk.PhotoImage(image=img)
        self.video_label.img_tk = img_tk  # type: ignore[attr-defined]
        self.video_label.config(image=img_tk)

        self.sign_display.config(text=str(sign))
        self.conf_label.config(text=f"Confidence: {conf * 100:.1f}%")
        self.conf_progress["value"] = conf * 100

        # Update Sentence Text Widget if content changed
        current_text = self.sentence_text.get("1.0", tk.END).strip()
        if current_text != sentence.strip():
            self.sentence_text.delete("1.0", tk.END)
            self.sentence_text.insert(tk.END, sentence)

    def _on_speak(self):
        """Triggers text-to-speech narration of current transcript."""
        text = self.sentence_text.get("1.0", tk.END).strip()
        if text:
            self.log_status(f"Speaking: '{text}'")
            self.tts.speak(text)
        else:
            messagebox.showinfo("TTS Info", "Sentence transcript is empty!")

    def _on_delete(self):
        """Deletes last character from transcript."""
        self.predictor.delete_last_char()
        self.sentence_text.delete("1.0", tk.END)
        self.sentence_text.insert(tk.END, self.predictor.current_sentence)

    def _on_clear(self):
        """Clears full transcript."""
        self.predictor.clear_sentence()
        self.sentence_text.delete("1.0", tk.END)

    def _on_generate_samples(self):
        """Triggers synthetic dataset generator."""
        self.log_status("Generating sample dataset...")
        try:
            generate_sample_dataset()
            messagebox.showinfo("Dataset Generator", "Sample landmark dataset successfully generated in 'data/hand_landmarks.csv'!")
            self.log_status("Sample dataset generated.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate dataset: {e}")

    def _on_train_model(self):
        """Triggers model training in background thread."""
        self.log_status("Training model... Please wait.")
        
        def train_worker():
            try:
                acc, model, encoder = train_sign_model()
                # Reload predictor model
                self.predictor.load_model()
                self.root.after(0, lambda: messagebox.showinfo("Training Complete", f"Model trained successfully!\nAccuracy: {acc * 100:.2f}%"))
                self.root.after(0, lambda: self.log_status(f"Model reloaded. Accuracy: {acc * 100:.2f}%"))
            except Exception as err:
                self.root.after(0, lambda: messagebox.showerror("Training Error", f"Model training failed: {err}"))
                self.root.after(0, lambda: self.log_status("Model training failed."))

        threading.Thread(target=train_worker, daemon=True).start()

    def on_closing(self):
        """Graceful application shutdown."""
        self.stop_camera()
        if self.hands:
            self.hands.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
