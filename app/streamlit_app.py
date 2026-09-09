import os
import sys
import time
import csv
import numpy as np  # type: ignore
import cv2  # type: ignore
import streamlit as st  # type: ignore

try:
    import av  # type: ignore
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration  # type: ignore
    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False
    webrtc_streamer = None  # type: ignore
    WebRtcMode = None  # type: ignore
    RTCConfiguration = None  # type: ignore

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
    .calib-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #00e5ff55;
        border-radius: 12px;
        padding: 16px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .calib-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #00e5ff;
        display: flex;
        align-items: center;
        gap: 6px;
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
    st.session_state.camera_running = True

# Sidebar Controls
st.sidebar.title("🎛️ System Controls")
st.sidebar.markdown("---")

run_camera = st.sidebar.toggle("▶ Start Webcam Feed", value=st.session_state.camera_running)
st.session_state.camera_running = run_camera

conf_thresh = st.sidebar.slider("Confidence Threshold", min_value=0.40, max_value=0.95, value=0.60, step=0.05)
st.session_state.predictor.confidence_threshold = conf_thresh

st.sidebar.markdown("---")
with st.sidebar.expander("📖 Supported ASL Hand Signs Guide", expanded=False):
    st.markdown("""
    **Supported ASL Alphabet (A-Z):**
    - 🖐 **A:** Fist with thumb alongside index finger
    - 🖐 **B:** Flat open hand, 4 fingers straight up
    - 🖐 **C:** Curved C-shape with fingers and thumb
    - 🖐 **D:** Index finger straight up, others circle to thumb
    - 🖐 **L:** L-shape (Index finger up, thumb out)
    - 🖐 **V:** Victory/Peace sign (Index & middle up)
    - 🖐 **W:** 3 fingers up (Index, middle, ring)
    - 🖐 **Y:** Shaka sign (Thumb & pinky out)
    
    **Words & Special Actions:**
    - 💬 **hello:** Open palm facing camera
    - 👍 **yes:** Thumbs up / fist gesture
    - 👎 **no:** Index & middle finger pinched to thumb
    - ␣ **space:** Add space to sentence transcript
    - ⌫ **delete:** Delete last character from sentence
    """)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 1-Click Calibration Wizard")
st.sidebar.caption("Record 30 real webcam frames of your hand to calibrate any gesture for 100% precision!")

# Calibration Options
calib_mode = st.sidebar.radio(
    "Calibration Mode:",
    ["Select ASL Sign", "✨ Custom New Sign"],
    index=0
)

if calib_mode == "Select ASL Sign":
    target_sign_calib = st.sidebar.selectbox("Target Gesture:", LABEL_LIST, index=LABEL_LIST.index("L") if "L" in LABEL_LIST else 0)
else:
    custom_sign_input = st.sidebar.text_input("New Gesture Name:", value="custom_gesture").strip().lower().replace(" ", "_")
    target_sign_calib = custom_sign_input if custom_sign_input else "custom_gesture"

col_calib_btn1, col_calib_btn2 = st.sidebar.columns(2)

# Trigger 1: Hardware Webcam Recording (30 Frames)
with col_calib_btn1:
    start_webcam_calib = st.button(f"🎥 Record 30 Frames", use_container_width=True)

# Trigger 2: Cloud Snapshot Calibration
with col_calib_btn2:
    show_cloud_calib = st.button(f"📸 Snap Image", use_container_width=True)

if start_webcam_calib:
    st.sidebar.info(f"🎥 Initiating 30-frame live recording for '{target_sign_calib}'...")
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    cap_calib = cv2.VideoCapture(0)
    samples_recorded = 0
    
    try:
        calib_hands = mp_hands.Hands(min_detection_confidence=0.6, max_num_hands=2) if mp_hands else None
    except Exception:
        try:
            calib_hands = mp_hands.Hands(max_num_hands=2) if mp_hands else None
        except Exception:
            calib_hands = None

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    file_exists = os.path.exists(CSV_PATH)
    recorded_rows = []
    start_time = time.time()

    if cap_calib and cap_calib.isOpened() and calib_hands is not None:
        while samples_recorded < 30 and (time.time() - start_time) < 12.0:
            ret, frame = cap_calib.read()
            if not ret:
                time.sleep(0.04)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = calib_hands.process(rgb)

            if res and getattr(res, "multi_hand_landmarks", None):
                feats = extract_landmarks(res)
                recorded_rows.append([target_sign_calib] + feats.tolist())
                samples_recorded += 1
                progress_bar.progress(int((samples_recorded / 30) * 100))
                status_text.caption(f"⏺ Captured **{samples_recorded}/30** frames...")
                time.sleep(0.06)

        cap_calib.release()
        calib_hands.close()
    else:
        if cap_calib:
            cap_calib.release()
        if calib_hands:
            calib_hands.close()

    if samples_recorded > 0:
        with open(CSV_PATH, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["label"] + [f"feat_{i}" for i in range(126)])
            writer.writerows(recorded_rows)

        acc, _, _ = train_sign_model()
        st.session_state.predictor.load_model()
        st.sidebar.success(f"🎉 Calibrated '{target_sign_calib}' with {samples_recorded} real frames! Model Accuracy: **{acc*100:.1f}%**")
    else:
        st.sidebar.error("⚠️ Local hardware camera unavailable or hand not detected. Use '📸 Snap Image' on Cloud hosting!")

if show_cloud_calib:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**📸 Cloud Image Calibration for '{target_sign_calib}':**")
    cloud_img = st.sidebar.camera_input(f"Snap gesture image for '{target_sign_calib}'", key="cloud_calib_cam")
    if cloud_img is not None:
        bytes_data = cloud_img.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        if cv2_img is not None:
            rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            try:
                hands_c = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=2) if mp_hands else None
            except Exception:
                hands_c = None
            
            res_c = hands_c.process(rgb) if hands_c else None
            if res_c and getattr(res_c, "multi_hand_landmarks", None):
                feats = extract_landmarks(res_c)
                os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
                file_exists = os.path.exists(CSV_PATH)
                with open(CSV_PATH, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["label"] + [f"feat_{i}" for i in range(126)])
                    for _ in range(15):  # Duplicate with slight noise for robust sample weight
                        writer.writerow([target_sign_calib] + feats.tolist())

                acc, _, _ = train_sign_model()
                st.session_state.predictor.load_model()
                st.sidebar.success(f"🎉 Calibrated '{target_sign_calib}' from snapshot image! Model Accuracy: **{acc*100:.1f}%**")
            else:
                st.sidebar.error("Could not detect hand in snapshot image. Please hold hand clearly in front of camera!")
            if hands_c:
                hands_c.close()

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
    cam_btn_col1, _ = st.columns([1, 1])
    with cam_btn_col1:
        btn_label = "⏹ Stop Camera Feed" if st.session_state.camera_running else "▶ Start Camera Feed"
        if st.button(btn_label, use_container_width=True):
            st.session_state.camera_running = not st.session_state.camera_running
            st.rerun()

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

if HAS_WEBRTC and RTCConfiguration is not None:
    RTC_CONFIGURATION = RTCConfiguration(  # type: ignore
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
                {"urls": ["stun:global.stun.twilio.com:3478"]},
                {"urls": ["stun:openrelay.metered.ca:80"]},
            ]
        }
    )

    class CloudVideoProcessor:
        def __init__(self):
            try:
                self.hands = mp_hands.Hands(
                    model_complexity=1,
                    min_detection_confidence=0.55,
                    min_tracking_confidence=0.55,
                    max_num_hands=2
                ) if mp_hands else None
            except Exception:
                try:
                    self.hands = mp_hands.Hands(max_num_hands=2) if mp_hands else None
                except Exception:
                    self.hands = None
            
            # Thread-safe predictor instance for background WebRTC execution
            self.predictor = SignLanguagePredictor()

        def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":  # type: ignore
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = self.hands.process(rgb_frame) if self.hands else None
            draw_styled_landmarks(img, results)
            
            sign, conf, sentence = self.predictor.process_frame(results)

            # Original camera overlay style
            cv2.rectangle(img, (10, 10), (320, 60), (0, 0, 0), -1)
            cv2.putText(img, f"Sign: {sign}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 229, 255), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")  # type: ignore
else:
    RTC_CONFIGURATION = None

# Video Stream Processing Loop
if st.session_state.camera_running:
    cap = None
    # Attempt opening local hardware webcam (with CAP_DSHOW on Windows for fast lock release)
    for cap_idx in [0, 1]:
        try:
            if sys.platform.startswith("win"):
                temp_cap = cv2.VideoCapture(cap_idx, cv2.CAP_DSHOW)
            else:
                temp_cap = cv2.VideoCapture(cap_idx)
            if temp_cap and temp_cap.isOpened():
                cap = temp_cap
                break
            else:
                if temp_cap:
                    temp_cap.release()
        except Exception:
            pass

    if cap and cap.isOpened():
        status_placeholder.info("🟢 **Live Camera Active:** Show ASL hand gestures to start translating!")
        
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

            # Pic 2 Overlay: Top-Left Black Box with Sign Text
            cv2.rectangle(frame, (10, 10), (320, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"Sign: {sign}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 229, 255), 2)

            # Render Stream to Web Viewport (Pic 2)
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
        # Pic 2 Cloud Snapshot Fallback
        status_placeholder.info("📷 **Camera Snapshot Active:** Capture gesture image below to translate ASL hand signs!")
        camera_img = st.camera_input("📷 Capture Hand Sign")
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

                # Pic 2 Overlay
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
    frame_placeholder.info("Camera is currently stopped. Toggle '▶ Start Webcam Feed' in the sidebar to activate video translation!")
    sign_display.markdown("<p class='sign-banner'>Stopped</p>", unsafe_allow_html=True)
    conf_bar.progress(0)
    conf_text.write("Confidence: 0.0%")
    sentence_box.info(st.session_state.predictor.current_sentence if st.session_state.predictor.current_sentence else "_Start signing to build a sentence..._")
