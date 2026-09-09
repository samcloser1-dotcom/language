import os
import sys
import time
import csv
import numpy as np  # type: ignore
import cv2  # type: ignore
import streamlit as st  # type: ignore

# Force Pure Python Protobuf Implementation
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Add project root directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from src.utils import mp_hands, draw_styled_landmarks, TextToSpeechManager, LABEL_LIST, extract_landmarks  # type: ignore
    from src.real_time_detection import SignLanguagePredictor  # type: ignore
    from src.train_model import train_sign_model  # type: ignore
    from create_sample_data import generate_sample_dataset  # type: ignore
except Exception:
    from utils import mp_hands, draw_styled_landmarks, TextToSpeechManager, LABEL_LIST, extract_landmarks  # type: ignore
    from real_time_detection import SignLanguagePredictor  # type: ignore
    from train_model import train_sign_model  # type: ignore
    from create_sample_data import generate_sample_dataset  # type: ignore

CSV_PATH = os.path.join(BASE_DIR, "data", "hand_landmarks.csv")

# Streamlit Page Config
st.set_page_config(
    page_title="Real-Time Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00e5ff;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #a0aec0;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1e2638;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2a354d;
    }
    .sign-banner {
        font-size: 2rem;
        font-weight: bold;
        color: #00e5ff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "predictor" not in st.session_state:
    st.session_state.predictor = SignLanguagePredictor()
else:
    st.session_state.predictor.load_model()

if "tts" not in st.session_state:
    st.session_state.tts = TextToSpeechManager()
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

# Sidebar Controls
st.sidebar.title("🎛️ System Controls")
st.sidebar.markdown("---")

run_camera = st.sidebar.toggle("▶ Start Webcam Feed", value=st.session_state.camera_running)
st.session_state.camera_running = run_camera

conf_thresh = st.sidebar.slider("Confidence Threshold", min_value=0.40, max_value=0.95, value=0.60, step=0.05)
st.session_state.predictor.confidence_threshold = conf_thresh

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 1-Click Calibration Wizard")
st.sidebar.caption("Record 30 real webcam frames of your hand to calibrate any gesture for 100% precision!")

target_sign_calib = st.sidebar.selectbox("Select Sign to Calibrate:", LABEL_LIST, index=LABEL_LIST.index("L") if "L" in LABEL_LIST else 0)

if st.sidebar.button(f"📸 Calibrate '{target_sign_calib}' from Webcam"):
    with st.spinner(f"Capturing 30 real webcam samples for '{target_sign_calib}'... Hold sign in front of camera!"):
        cap_calib = cv2.VideoCapture(0)
        samples_recorded = 0
        
        try:
            calib_hands = mp_hands.Hands(min_detection_confidence=0.6, max_num_hands=2)
        except Exception:
            calib_hands = mp_hands.Hands(max_num_hands=2)

        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        file_exists = os.path.exists(CSV_PATH)

        recorded_rows = []
        start_time = time.time()

        while samples_recorded < 30 and (time.time() - start_time) < 10.0:
            ret, frame = cap_calib.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = calib_hands.process(rgb)

            if res and res.multi_hand_landmarks:
                feats = extract_landmarks(res)
                recorded_rows.append([target_sign_calib] + feats.tolist())
                samples_recorded += 1
                time.sleep(0.08)

        cap_calib.release()
        calib_hands.close()

        if samples_recorded > 0:
            with open(CSV_PATH, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["label"] + [f"feat_{i}" for i in range(126)])
                writer.writerows(recorded_rows)

            acc, _, _ = train_sign_model()
            st.session_state.predictor.load_model()
            st.sidebar.success(f"Successfully recorded {samples_recorded} real frames for '{target_sign_calib}'! Model accuracy: {acc*100:.1f}%")
        else:
            st.sidebar.error("Could not detect hand in camera feed during calibration. Please try again!")

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Model & Dataset Actions")

if st.sidebar.button("⚡ Generate Baseline Dataset"):
    with st.spinner("Generating MediaPipe skeleton baseline dataset..."):
        generate_sample_dataset()
        acc, _, _ = train_sign_model()
        st.session_state.predictor.load_model()
        st.sidebar.success(f"Dataset generated and trained! Accuracy: {acc*100:.1f}%")

if st.sidebar.button("🧠 Retrain Model"):
    with st.spinner("Training model on landmark dataset..."):
        try:
            acc, _, _ = train_sign_model()
            st.session_state.predictor.load_model()
            st.sidebar.success(f"Model trained successfully! Accuracy: {acc*100:.1f}%")
        except Exception as e:
            st.sidebar.error(f"Training failed: {e}")

if st.sidebar.button("🗑 Clear Sentence Transcript"):
    st.session_state.predictor.clear_sentence()
    st.sidebar.info("Sentence transcript cleared.")

# Header
st.markdown("<p class='main-header'>🤟 Real-Time Sign Language Detection & Translation</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Powered by MediaPipe Hand Keypoint Mesh & Deep MLP Classifier</p>", unsafe_allow_html=True)

# Main 2-Column Layout
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📹 Live Camera Feed & Landmark Tracking")
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

with col2:
    st.subheader("📊 Live Translation Dashboard")
    
    # Active Sign Card
    with st.container():
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("##### CURRENT DETECTED SIGN")
        sign_display = st.empty()
        conf_bar = st.progress(0)
        conf_text = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # Sentence Transcript
    st.markdown("##### 📝 CONSTRUCTED SENTENCE TRANSCRIPT")
    sentence_box = st.empty()
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔊 Speak Sentence", use_container_width=True):
            text = st.session_state.predictor.current_sentence.strip()
            if text:
                st.session_state.tts.speak(text)
                st.success(f"Speaking: '{text}'")
            else:
                st.warning("Sentence transcript is empty!")
    with col_btn2:
        if st.button("⌫ Delete Last Char", use_container_width=True):
            st.session_state.predictor.delete_last_char()

# Video Stream Processing Loop
if st.session_state.camera_running:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        status_placeholder.info("☁️ **Cloud Mode Active:** Hardware camera unavailable on cloud server. Use your browser's camera snapshot widget below to translate ASL hand signs!")
        camera_img = st.camera_input("📷 Capture Hand Sign Snapshot")
        if camera_img is not None:
            bytes_data = camera_img.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            if cv2_img is not None:
                rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                try:
                    hands = mp_hands.Hands(min_detection_confidence=0.55, max_num_hands=2) if mp_hands else None
                except Exception:
                    hands = mp_hands.Hands(max_num_hands=2) if mp_hands else None

                results = hands.process(rgb_img) if hands else None
                draw_styled_landmarks(cv2_img, results)
                sign, conf, sentence = st.session_state.predictor.process_frame(results)

                cv2.rectangle(cv2_img, (10, 10), (320, 60), (0, 0, 0), -1)
                cv2.putText(cv2_img, f"Sign: {sign}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 229, 255), 2)

                frame_placeholder.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), channels="RGB")
                sign_display.markdown(f"<p class='sign-banner'>{sign}</p>", unsafe_allow_html=True)
                conf_bar.progress(int(conf * 100))
                conf_text.write(f"Confidence: **{conf * 100:.1f}%**")
                sentence_box.info(sentence if sentence else "_Start signing to build a sentence..._")
                if hands:
                    hands.close()
    else:
        status_placeholder.info("Camera active. Show ASL hand gestures to start translating!")
        
        try:
            thread_hands = mp_hands.Hands(
                model_complexity=1,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55,
                max_num_hands=2
            ) if mp_hands else None
        except Exception:
            thread_hands = mp_hands.Hands(max_num_hands=2) if mp_hands else None

        while st.session_state.camera_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process Hand Landmarks
            results = thread_hands.process(rgb_frame) if thread_hands else None

            # Draw Skeletal Hand Mesh
            draw_styled_landmarks(frame, results)

            # Predict Sign Gesture
            sign, conf, sentence = st.session_state.predictor.process_frame(results)

            # Visual Banner Overlay on Frame
            cv2.rectangle(frame, (10, 10), (320, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"Sign: {sign}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 229, 255), 2)

            # Render Stream to Web Viewport
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            # Render Live Dashboard Updates
            sign_display.markdown(f"<p class='sign-banner'>{sign}</p>", unsafe_allow_html=True)
            conf_bar.progress(int(conf * 100))
            conf_text.write(f"Confidence: **{conf * 100:.1f}%**")
            sentence_box.info(sentence if sentence else "_Start signing to build a sentence..._")

            time.sleep(0.01)

        cap.release()
        if thread_hands:
            thread_hands.close()
        status_placeholder.warning("Camera stream stopped.")

else:
    frame_placeholder.info("Camera is currently stopped. Toggle '▶ Start Webcam Feed' in the sidebar to activate video translation!")
    sign_display.markdown("<p class='sign-banner'>Stopped</p>", unsafe_allow_html=True)
    conf_bar.progress(0)
    conf_text.write("Confidence: 0.0%")
    sentence_box.info(st.session_state.predictor.current_sentence if st.session_state.predictor.current_sentence else "_Start signing to build a sentence..._")
