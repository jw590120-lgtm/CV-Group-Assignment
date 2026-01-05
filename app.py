import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import threading
import gc

# ===========================
# 1. 基础配置
# ===========================
st.set_page_config(page_title="AI Gesture Lite", page_icon="🖐️")

# 兼容性补丁
if not hasattr(st, "experimental_rerun"):
    st.experimental_rerun = st.rerun

# ===========================
# 2. 轻量化模型定义
# ===========================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 简化的模型结构（只用于推理）
class BiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BiLSTMAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes) # 简化全连接层

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 简化注意力机制，直接取最后一帧，节省计算量
        out = self.fc(lstm_out[:, -1, :])
        return out

def extract_keypoints(results):
    # 保持数据格式一致
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

@st.cache_resource
def load_model():
    gestures = [f"Gesture {i}" for i in range(1, 16)]
    try:
        class Attention(nn.Module):
            def __init__(self, hidden_dim):
                super(Attention, self).__init__()
                self.attention = nn.Linear(hidden_dim, 1)
            def forward(self, lstm_output):
                energy = self.attention(lstm_output)
                weights = torch.softmax(energy, dim=1)
                context_vector = torch.sum(lstm_output * weights, dim=1)
                return context_vector, weights

        class OriginalBiLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes, num_layers=2):
                super(OriginalBiLSTM, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.4, bidirectional=True)
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

        device = torch.device("cpu")
        model = OriginalBiLSTM(input_size=258, hidden_size=128, num_classes=len(gestures))
        model.load_state_dict(torch.load("trained_model.pth", map_location=device))
        model.eval()
        return model, gestures
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

global_model, global_gestures = load_model()

# ===========================
# 3. 极速版处理器
# ===========================
class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        # ⚠️ 关键优化：model_complexity=0 (最快，最省内存)
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0, 
            refine_face_landmarks=False
        )
        self.sequence = []
        self.predicted_gesture = "Init..."
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            image = frame.to_ndarray(format="bgr24")
            
            # ⚠️ 关键优化：强制缩小分辨率
            image = cv2.resize(image, (320, 240)) 

            image.flags.writeable = False
            results = self.holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image.flags.writeable = True

            # 仅绘制手部，减少 CPU 绘图压力
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            keypoints = extract_keypoints(results)
            
            with self.lock:
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]

                if len(self.sequence) == 30 and global_model:
                    inp = torch.tensor(np.array([self.sequence]), dtype=torch.float32)
                    with torch.no_grad():
                        out = global_model(inp)
                        probs = torch.softmax(out, dim=1)[0]
                        conf, idx = torch.max(probs, 0)
                        if conf.item() > 0.5:
                            self.predicted_gesture = f"{global_gestures[idx.item()]} ({int(conf.item()*100)}%)"
                        else:
                            self.predicted_gesture = "..."

            cv2.putText(image, self.predicted_gesture, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # ⚠️ 关键：手动垃圾回收
            gc.collect()
            
            return av.VideoFrame.from_ndarray(image, format="bgr24")
        except Exception:
            return frame

# ===========================
# 4. 界面
# ===========================
st.title("🖐️ Gesture Recognition (Lite)")
st.caption("Running in Low-Latency Mode for Free Tier Servers")

rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ⚠️ 关键优化：关闭 async_processing，防止队列堆积内存溢出
webrtc_ctx = webrtc_streamer(
    key="gesture-recognition-lite",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    video_processor_factory=GestureProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False 
)

if webrtc_ctx.state.playing:
    st.success("Camera Active")
