import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import pandas as pd
import tempfile
import time
import os

# ===========================
# 1. 页面配置与美化
# ===========================
st.set_page_config(
    page_title="AI Gesture Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%; border-radius: 10px; height: 3em;
        background-color: #FF4B4B; color: white; font-weight: bold;
    }
    .stButton>button:hover { background-color: #D93F3F; border-color: #D93F3F; }
    </style>
    """, unsafe_allow_html=True)

# ===========================
# 2. 核心模型定义
# ===========================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- 人脸模糊工具函数 (新增) ---
def blur_face_region(image, results):
    """
    检测人脸坐标并应用高斯模糊
    使用 pose_landmarks (0-10点) 来快速定位人脸，比 face_landmarks 更快
    """
    if not results.pose_landmarks:
        return image
        
    h, w, _ = image.shape
    
    # 提取面部关键点 (鼻子0, 眼睛1-6, 耳朵7-8, 嘴巴9-10)
    face_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x_coords = []
    y_coords = []
    
    for idx in face_indices:
        lm = results.pose_landmarks.landmark[idx]
        x_coords.append(int(lm.x * w))
        y_coords.append(int(lm.y * h))
    
    if not x_coords or not y_coords:
        return image
        
    # 计算边界框
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # 添加一些边距 (Padding) 让模糊范围更大一点
    padding_w = int((x_max - x_min) * 0.5)
    padding_h = int((y_max - y_min) * 0.5)
    
    x_min = max(0, x_min - padding_w)
    x_max = min(w, x_max + padding_w)
    y_min = max(0, y_min - padding_h)
    y_max = min(h, y_max + padding_h)
    
    # 截取人脸区域
    face_roi = image[y_min:y_max, x_min:x_max]
    
    if face_roi.size > 0:
        # 应用强力高斯模糊
        # (99, 99) 是模糊核大小，必须是奇数，越大越模糊
        blurred_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)
        # 将模糊后的区域放回原图
        image[y_min:y_max, x_min:x_max] = blurred_roi
        
    return image

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
    gestures = [
        "abang", "apa", "ayah", "beli", "bila",
        "bomba", "buat", "emak", "hi", "lelaki",
        "main", "polis", "saudara", "siapa", "tandas"
    ]
    device = torch.device("cpu")
    model = BiLSTMAttention(input_size=258, hidden_size=128, num_classes=len(gestures))
    
    try:
        model.load_state_dict(torch.load("trained_model.pth", map_location=device))
        model.eval()
        return model, gestures, "Loaded"
    except FileNotFoundError:
        return None, None, "Missing File"
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# ===========================
# 3. 侧边栏设计
# ===========================
with st.sidebar:
    st.title("🧩 System Dashboard")
    st.markdown("---")
    
    # --- 新增：隐私保护开关 ---
    st.write("### 🛡️ Privacy Settings")
    enable_blur = st.checkbox("🙈 Blur Faces", value=False, help="Automatically detect and blur faces in the video.")
    
    if enable_blur:
        st.info("Privacy Mode Active: Faces will be blurred in processing.")
    
    st.markdown("---")
    
    # 模型状态
    model, gestures, status = load_model()
    if status == "Loaded":
        st.success("Model Status: **Active** ✅")
        st.warning("⚠️ **Model Limitation**")
        st.caption("Current model supports ONLY these 15 gestures:")
        st.code("\n".join(gestures), language="text")
    else:
        st.error(f"Model Status: **{status}** ❌")
        st.warning("Please upload 'trained_model.pth'.")
    
    st.markdown("---")
    st.caption("CV Group Assignment 2025")

# ===========================
# 4. 主界面设计
# ===========================

st.markdown("# 🎬 AI Gesture Analysis Studio")
st.markdown("#### Upload a video to identify dynamic gestures using Deep Learning.")
st.markdown("---")

uploaded_file = st.file_uploader("", type=['mp4', 'mov', 'avi'], help="Supported formats: MP4, MOV, AVI")

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    col_video, col_results = st.columns([1.5, 1])
    
    with col_video:
        st.subheader("📺 Video Preview")
        
        # --- 隐私保护逻辑：如果开启模糊，则不显示原视频 ---
        if enable_blur:
            st.warning("🔒 **Raw Video Hidden** (Privacy Mode On)")
            st.image("https://placehold.co/600x400/333/FFF?text=Privacy+Mode+Active", use_column_width=True)
        else:
            st.video(uploaded_file)
        
        process_btn = st.button("🚀 Start Deep Analysis", type="primary")

    if process_btn:
        if model is None:
            st.error("Cannot proceed: Model not loaded.")
        else:
            with col_results:
                st.subheader("📊 Processing Status")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_window = st.empty() # 用于显示处理画面的占位符
                
                cap = cv2.VideoCapture(tfile.name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames == 0: total_frames = 100
                skip = max(int(total_frames / 30), 1)
                sequence = []
                
                with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    
                    for i in range(30):
                        progress = int((i / 30) * 100)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {i+1}/30...")
                        
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
                        ret, frame = cap.read()
                        if not ret: break
                        
                        # 转换颜色
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # MediaPipe 处理
                        res = holistic.process(frame)
                        
                        # --- 核心修改：如果是模糊模式，处理人脸 ---
                        display_frame = frame.copy()
                        if enable_blur:
                            display_frame = blur_face_region(display_frame, res)
                            
                        # 在显示的画面上画骨骼点 (可选，增加科技感)
                        mp_drawing.draw_landmarks(display_frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                        mp_drawing.draw_landmarks(display_frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                        
                        # 实时更新左侧视频区域的画面，让用户看到处理过程
                        frame_window.image(display_frame, channels="RGB", caption=f"Frame {i+1} Analysis", use_column_width=True)
                        
                        # 收集数据用于推理
                        sequence.append(extract_keypoints(res))
                
                cap.release()
                progress_bar.progress(100)
                status_text.success("✅ Extraction Complete!")
                
                # 补齐数据
                while len(sequence) < 30:
                    sequence.append(np.zeros(258))
                
                # --- 推理逻辑 ---
                with st.spinner("🧠 Analyzing Pattern..."):
                    input_tensor = torch.tensor(np.array([sequence]), dtype=torch.float32)
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, dim=1)[0]
                    
                    conf, idx = torch.max(probs, 0)
                    prediction = gestures[idx.item()]
                    confidence_val = conf.item() * 100
                    time.sleep(0.5)

                # --- 结果展示 ---
                st.divider()
                st.metric(label="🏆 Top Prediction", value=prediction, delta=f"{confidence_val:.2f}% Confidence")
                
                if confidence_val > 80: st.balloons()
                
                st.write("### 📈 Full Probability Distribution")
                chart_data = pd.DataFrame({
                    "Gesture": gestures,
                    "Probability": probs.numpy()
                }).sort_values(by="Probability", ascending=False)
                
                st.bar_chart(chart_data, x="Gesture", y="Probability", color="#FF4B4B")
                
                with st.expander("📄 View Raw Data Table"):
                    st.dataframe(chart_data.style.format({"Probability": "{:.4%}"}))

else:
    st.info("👈 Please upload a video file from the sidebar or main area to begin.")
