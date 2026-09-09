# Real-Time Sign Language Detection and Translation System 🤟

A complete, production-grade Python application for real-time American Sign Language (ASL) detection, gesture classification, sentence translation, text-to-speech output, and interactive desktop GUI.

Powered by **OpenCV**, **MediaPipe Hands**, **Scikit-Learn Neural Networks (MLPClassifier)**, **PyTTSx3**, and **Tkinter**.

---

## 🌟 Key Features

- **MediaPipe Hand Landmark Extraction**: Extracts 21 3D hand keypoints $(x, y, z)$ per hand (up to 2 hands, 126 features total).
- **Scale & Origin Invariance**: Hand landmarks are translated relative to the wrist origin and normalized by hand span, ensuring accurate gesture detection regardless of hand size or camera distance.
- **Deep Neural Network Classifier**: Multi-Layer Perceptron (MLP) architecture trained to classify 26 ASL alphabet letters ($A-Z$), common phrase gestures (`hello`, `thank_you`, `yes`, `no`, `please`), and control actions (`space`, `delete`).
- **Temporal Prediction Smoothing**: Uses rolling prediction buffers (`deque`) and confidence thresholding to prevent frame flickering and eliminate false positives.
- **Sentence Builder State Machine**: Assembles characters into words, inserts spaces, handles deletion gestures, and converts full sentences to speech via `pyttsx3`.
- **Modern Desktop GUI**: Sleek dark-mode Tkinter dashboard featuring live camera preview with skeletal hand mesh, confidence meter, transcript text box, text-to-speech narration, and one-click training.
- **Out-of-the-Box Execution**: Includes an automated synthetic sample dataset generator (`create_sample_data.py`) so you can train and run the system immediately without manual data recording.

---

## 📁 Project Directory Structure

```
sign-language-detection/
├── data/                      # Dataset storage
│   └── hand_landmarks.csv     # Extracted 126-feature landmark vectors with labels
├── models/                    # Serialized model artifacts
│   ├── sign_model.pkl         # Trained MLPClassifier neural network
│   └── label_encoder.pkl      # Label encoder mapping
├── src/                       # Core application packages
│   ├── __init__.py
│   ├── utils.py               # MediaPipe landmarks, normalization & TTS manager
│   ├── data_collection.py     # Interactive webcam dataset collector module
│   ├── train_model.py         # Neural network training & evaluation script
│   └── real_time_detection.py # Real-time prediction engine & sentence buffer
├── app/                       # User Interface
│   └── gui.py                 # Tkinter Desktop GUI application
├── create_sample_data.py      # Synthetic sample dataset generator
├── main.py                    # Multi-purpose CLI launcher entry point
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup

Ensure you have Python 3.10 or higher installed.

```bash
# Clone or navigate into the project directory
cd c:\Users\moham\OneDrive\Desktop\LANGUAGE

# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Immediate End-to-End Execution (Sample Data Mode)

You can run the entire pipeline out-of-the-box in 3 simple commands:

```bash
# Step 1: Generate sample landmark dataset
python main.py --sample-data

# Step 2: Train the neural network model
python main.py --train

# Step 3: Launch the Desktop GUI Application
python main.py
```

---

## 📽️ Interactive Usage & Modules

### A. Collecting Custom Webcam Sign Data (`src/data_collection.py`)

To collect real custom sign language gestures using your webcam:

```bash
python main.py --collect
```

**Webcam Collector Keyboard Controls:**
- `n` / `p`: Switch to Next / Previous sign label (e.g. 'A' through 'Z', 'hello', 'thank_you', etc.).
- `SPACE`: Save current webcam hand frame as a single landmark sample in `data/hand_landmarks.csv`.
- `c`: Capture a burst of 15 continuous samples with slight delay.
- `q` or `ESC`: Quit data collection.

---

### B. Training & Evaluating the Model (`src/train_model.py`)

Train a fresh classification model on all samples stored in `data/hand_landmarks.csv`:

```bash
python main.py --train
```

**Model Evaluation Output:**
- Accuracy Score percentage
- Classification Report (Precision, Recall, F1-Score per sign label)
- Confusion Matrix Overview
- Automatic saving of `models/sign_model.pkl` and `models/label_encoder.pkl`.

---

### C. Running the Live Translation Desktop App (`app/gui.py`)

Launch the main GUI dashboard:

```bash
python main.py
# or
python app/gui.py
```

**GUI Features:**
- **Start / Stop Camera**: Toggles live video feed with drawn green/cyan hand skeletal mesh.
- **Current Detected Sign**: Large overlay displaying real-time prediction and confidence progress bar.
- **Constructed Sentence**: Live editable transcript box accumulating detected signs.
- **🔊 Speak Text**: Reads sentence aloud using `pyttsx3` without freezing video playback.
- **⌫ Delete Last**: Deletes last character.
- **🗑 Clear**: Resets sentence transcript.
- **⚡ Generate Sample Dataset** & **🧠 Train Model**: Train or re-train model weights directly inside the GUI!

---

## 🖐️ Supported Signs & Gestures

| Category | Supported Labels |
| :--- | :--- |
| **ASL Alphabet** | `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`, `Z` |
| **Common Words** | `hello`, `thank_you`, `yes`, `no`, `please` |
| **Control Actions** | `space` (Appends word space), `delete` (Deletes last character) |

---

## 🛠️ Technical Details

1. **Feature Extraction (`src/utils.py`)**:
   - MediaPipe Hands processes frames at up to 60+ FPS.
   - Extracts $(x, y, z)$ coordinates for 21 key points per hand.
   - Normalization: Coordinates are shifted relative to Landmark 0 (wrist) and divided by the maximum hand span distance:
     $$\mathbf{x}_{\text{norm}} = \frac{\mathbf{x} - \mathbf{x}_{\text{wrist}}}{\max_{i} \|\mathbf{x}_i - \mathbf{x}_{\text{wrist}}\|}$$
   - Zero-padded for up to 2 hands (total feature length = $21 \times 3 \times 2 = 126$).

2. **Inference & Smoothing (`src/real_time_detection.py`)**:
   - Predictions are pushed into a rolling `deque(maxlen=12)` buffer.
   - A gesture is committed to the transcript only if it achieves $\ge 70\%$ agreement across buffered frames with probability $> 0.65$.
   - Includes a 1.0s cooldown timer to avoid duplicate auto-repeats while holding a sign steady.

3. **Non-Blocking Audio (`src/utils.py`)**:
   - `pyttsx3` is managed inside a daemon thread worker with a queue to ensure audio playback never blocks the Tkinter GUI thread or OpenCV video rendering.

---

## ❓ Troubleshooting

- **Camera Not Opening**: Ensure your webcam is connected and not locked by another application (e.g. Zoom/Teams). If index 0 fails, change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `app/gui.py` and `src/data_collection.py`.
- **PyTTSx3 Audio Issue on Linux**: Install `espeak` via `sudo apt-get install espeak` or `ffmpeg`.
- **Low Prediction Accuracy**: Ensure consistent lighting and record at least 50–100 samples per sign using `python main.py --collect`.

---

## 📜 License

Distributed under the MIT License.
