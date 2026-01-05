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

# ===========================
# 1. 页面配置
# ===========================
st.set_page_config(
    page_title="AI Gesture Recognition System",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 兼容性补丁
if not hasattr(st, "experimental_rerun"):
    st.experimental_rerun = st.rerun

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {background-color: #F5F7F9;}
    </style>
    """, unsafe_allow_html=True)

# ===========================
# 2. 模型定义
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

# --- 加载模型 ---
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

global_model, global_gestures = load_model()

# ===========================
# 3. 摄像头处理逻辑
# ===========================
class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.sequence = [] 
        self.predicted_gesture = "Waiting..."
        self.confidence_str = ""
        self.lock = threading.Lock() 

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            image = frame.to_ndarray(format="bgr24")
            
            # 降低处理频率：如果图片过大，可以稍微缩小一点以提高速度
            # image = cv2.resize(image, (640, 480))

            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image_rgb)
            image.flags.writeable = True

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            keypoints = extract_keypoints(results)
            
            with self.lock:
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:] 

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

            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, f"Result: {self.predicted_gesture} {self.confidence_str}", 
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(image, format="bgr24")
        except Exception as e:
            # 捕获任何处理错误，防止断开连接
            print(f"Error in processing: {e}")
            return frame

# ===========================
# 4. 界面布局
# ===========================
with st.sidebar:
    st.title("Control Panel ⚙️")
    st.markdown("---")
    app_mode = st.radio("Select Mode:", ["📸 Real-time Webcam", "📂 Upload Video File"])
    st.markdown("---")
    st.info("System Ready.")

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🖐️ AI Gesture Recognition System</h1>", unsafe_allow_html=True)
st.markdown("---")

if global_model is None:
    st.error("Model not loaded.")
    st.stop()

if app_mode == "📸 Real-time Webcam":
    st.header("📸 Real-time Inference")
    st.write("Please allow browser camera access.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        rtc_configuration = RTCConfiguration(
            {"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
            ]}
        )
        
        webrtc_ctx = webrtc_streamer(
            key="gesture-recognition",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_processor_factory=GestureProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )
        
    with col2:
        st.subheader("💡 Instructions")
        st.write("Click START to begin.")
        st.caption("Note: Initialization may take 10-20 seconds due to cloud latency.")
        if webrtc_ctx.state.playing:
             st.success("Camera Running", icon="🤖")

elif app_mode == "📂 Upload Video File":
    st.header("📂 Offline Analysis")
    uploaded_file = st.file_uploader("Upload Video (mp4/mov/avi)", type=['mp4', 'mov', 'avi'])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        col1, col2 = st.columns([3, 2])
        with col1: st.video(uploaded_file)
        with col2:
            if st.button("Start Analysis", type="primary"):
                with st.spinner("Processing..."):
                    cap = cv2.VideoCapture(tfile.name)
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    skip = max(int(total/30), 1)
                    seq = []
                    with mp_holistic.Holistic() as holistic:
                        for i in range(30):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, i*skip)
                            ret, f = cap.read()
                            if not ret: break
                            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                            res = holistic.process(f)
                            seq.append(extract_keypoints(res))
                    cap.release()
                    while len(seq)<30: seq.append(np.zeros(258))
                    
                    inp = torch.tensor(np.array([seq]), dtype=torch.float32)
                    with torch.no_grad():
                        out = global_model(inp)
                        probs = torch.softmax(out, dim=1)[0]
                    
                    conf, idx = torch.max(probs, 0)
                    st.success(f"Prediction: **{global_gestures[idx.item()]}**")
                    st.metric("Confidence", f"{conf.item()*100:.2f}%")
                    
                    chart_data = pd.DataFrame({
                        "Gesture": global_gestures,
                        "Prob": probs.numpy()
                    }).sort_values("Prob", ascending=False)
                    st.bar_chart(chart_data.set_index("Gesture"))
