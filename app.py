import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import pandas as pd
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import threading
import streamlit as st
if not hasattr(st, "experimental_rerun"):
    st.experimental_rerun = st.rerun

# ===========================
# 1. 页面配置 (Page Configuration)
# ===========================
st.set_page_config(
    page_title="AI Gesture Recognition System",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {background-color: #F5F7F9;}
    </style>
    """, unsafe_allow_html=True)

# ===========================
# 2. 模型定义 (Model Architecture)
# ===========================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    def forward(self, lstm_output):
        energy = self.attention(lstm_output)
        weights = torch.softmax(energy, dim=1)
        context_vector = torch.sum(lstm_output * weights, dim=1)
        return context_vector, weights

class BiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
        super(BiLSTMAttention, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=0.4, bidirectional=True
        )
        self.attention = Attention(hidden_size * 2)
        self.bn = nn.BatchNorm1d(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.output_layer = nn.Linear(128, num_classes)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context_vector, _ = self.attention(lstm_out)
        out = self.bn(context_vector)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.output_layer(out)
        return out

def extract_keypoints(results):
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# --- 加载模型 (Load Model) ---
@st.cache_resource
def load_model():
    gestures = [f"Gesture {i}" for i in range(1, 16)] 
    
    device = torch.device("cpu")
    model = BiLSTMAttention(input_size=258, hidden_size=128, num_classes=len(gestures))
    try:
        model.load_state_dict(torch.load("trained_model.pth", map_location=device))
        model.eval()
        st.sidebar.success("Model Loaded Successfully!", icon="✅")
    except FileNotFoundError:
        st.sidebar.error("Critical Error: 'trained_model.pth' not found.")
        return None, None
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        return None, None
    return model, gestures

# 初始化全局模型
global_model, global_gestures = load_model()

# ===========================
# 3. 摄像头实时处理逻辑 (Real-time Webcam Processor)
# ===========================
class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.sequence = [] # 缓冲区
        self.predicted_gesture = "Waiting..."
        self.confidence_str = ""
        self.lock = threading.Lock() # 线程锁

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 获取图像并转为 OpenCV 格式
        image = frame.to_ndarray(format="bgr24")
        
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image_rgb)
        image.flags.writeable = True

        # 绘制骨骼点
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # 提取关键点并推理
        keypoints = extract_keypoints(results)
        
        with self.lock:
            self.sequence.append(keypoints)
            self.sequence = self.sequence[-30:] # 保持最新的30帧

            if len(self.sequence) == 30 and global_model is not None:
                input_tensor = torch.tensor(np.array([self.sequence]), dtype=torch.float32)
                with torch.no_grad():
                    output = global_model(input_tensor)
                    probs = torch.softmax(output, dim=1)[0]
                    conf, idx = torch.max(probs, 0)
                    
                    current_conf = conf.item()
                    if current_conf > 0.6: 
                        self.predicted_gesture = global_gestures[idx.item()]
                        self.confidence_str = f"({current_conf*100:.1f}%)"
                    else:
                        self.predicted_gesture = "Analyzing..."
                        self.confidence_str = ""

        # 在视频上绘制结果
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, f"Result: {self.predicted_gesture} {self.confidence_str}", 
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# ===========================
# 4. 界面布局 (UI Layout)
# ===========================

# --- 侧边栏 (Sidebar) ---
with st.sidebar:
    st.title("Control Panel ⚙️")
    st.markdown("---")
    app_mode = st.radio("Select Mode:", ["📸 Real-time Webcam", "📂 Upload Video File"])
    st.markdown("---")
    st.info("""
    **About System:**
    This system utilizes a BiLSTM + Attention deep learning model combined with MediaPipe to perform real-time recognition of dynamic gesture sequences.
    """)

# --- 主标题 (Main Title) ---
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🖐️ AI Gesture Recognition System</h1>", unsafe_allow_html=True)
st.markdown("---")

# 检查模型是否加载
if global_model is None:
    st.error("Model not loaded. Please ensure 'trained_model.pth' is uploaded to the repository.")
    st.stop()

# ===========================
# 模式一: 实时摄像头 (Real-time Webcam)
# ===========================
if app_mode == "📸 Real-time Webcam":
    st.header("📸 Real-time Inference")
    st.write("Please allow browser camera access. The system analyzes your movement in real-time.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # WebRTC 配置 (STUN Server)
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        webrtc_ctx = webrtc_streamer(
            key="gesture-recognition",
            mode=WebRtcMode.SENDRECV, # 已经修正为 WebRtcMode.SENDRECV
            rtc_configuration=rtc_configuration,
            video_processor_factory=GestureProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
    with col2:
        st.subheader("💡 Instructions")
        st.markdown("""
        1. Click **"START"** to enable the camera.
        2. Ensure your upper body and hands are visible.
        3. Perform the gesture.
        4. The system needs about **30 frames** (1 sec) to start predicting.
        5. Results are shown at the top of the video feed.
        """)
        if webrtc_ctx.state.playing:
             st.success("Camera is running...", icon="🤖")

# ===========================
# 模式二: 视频文件上传 (Video File Upload)
# ===========================
elif app_mode == "📂 Upload Video File":
    st.header("📂 Offline Analysis")
    
    uploaded_file = st.file_uploader("Upload Video (mp4/mov/avi)", type=['mp4', 'mov', 'avi'])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.video(uploaded_file)
        
        with col2:
            st.subheader("Analysis Results")
            if st.button("Start Analysis", type="primary"):
                with st.spinner("Processing video frame by frame..."):
                    # --- 处理视频 (Process Video) ---
                    cap = cv2.VideoCapture(tfile.name)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    skip = max(int(total_frames / 30), 1)
                    sequence = []
                    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                        for i in range(30):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
                            ret, frame = cap.read()
                            if not ret: break
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            res = holistic.process(frame)
                            sequence.append(extract_keypoints(res))
                    cap.release()
                    while len(sequence) < 30: sequence.append(np.zeros(258))
                    
                    # 推理 (Inference)
                    input_tensor = torch.tensor(np.array([sequence]), dtype=torch.float32)
                    with torch.no_grad():
                        output = global_model(input_tensor)
                        probs = torch.softmax(output, dim=1)[0]
                    
                    # 显示结果 (Display)
                    conf, idx = torch.max(probs, 0)
                    st.success(f"Prediction: **{global_gestures[idx.item()]}**")
                    st.metric("Confidence", f"{conf.item() * 100:.2f}%")
                    
                    st.markdown("---")
                    st.write("### 📊 Probability Distribution")
                    chart_data = pd.DataFrame({
                        "Gesture": global_gestures,
                        "Probability (%)": (probs.numpy() * 100).round(2)
                    }).sort_values(by="Probability (%)", ascending=False)
                    
                    st.bar_chart(chart_data.set_index("Gesture"), color="#FF4B4B")

