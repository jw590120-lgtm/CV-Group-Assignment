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
# 1. 页面配置与美化 (UI Configuration)
# ===========================
st.set_page_config(
    page_title="AI Gesture Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #D93F3F;
        border-color: #D93F3F;
    }
    h1 {
        color: #1E1E1E;
    }
    .css-1aumxhk {
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ===========================
# 2. 核心模型定义 (Model Core)
# ===========================
mp_holistic = mp.solutions.holistic

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
    # 【重要】请修改为你真实的15个英文手势名称
    gestures = [f"Gesture {i}" for i in range(1, 16)] 
    
    device = torch.device("cpu")
    model = BiLSTMAttention(input_size=258, hidden_size=128, num_classes=len(gestures))
    
    status_text = "Checking model file..."
    try:
        model.load_state_dict(torch.load("trained_model.pth", map_location=device))
        model.eval()
        return model, gestures, "Loaded"
    except FileNotFoundError:
        return None, None, "Missing File"
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# ===========================
# 3. 侧边栏设计 (Sidebar)
# ===========================
with st.sidebar:
    st.title("🧩 System Dashboard")
    st.markdown("---")
    
    # 模型状态指示器
    model, gestures, status = load_model()
    if status == "Loaded":
        st.success("Model Status: **Active** ✅")
        st.caption(f"Architecture: BiLSTM + Attention\nClasses: {len(gestures)}")
    else:
        st.error(f"Model Status: **{status}** ❌")
        st.warning("Please upload 'trained_model.pth' to your GitHub repository.")
    
    st.markdown("---")
    st.info("""
    **How to use:**
    1. Upload a video file.
    2. Click 'Start Analysis'.
    3. View frame-by-frame processing.
    4. Check the prediction report.
    """)
    st.markdown("---")
    st.caption("CV Group Assignment 2025")

# ===========================
# 4. 主界面设计 (Main Interface)
# ===========================

# 标题区
st.markdown("# 🎬 AI Gesture Analysis Studio")
st.markdown("#### Upload a video to identify dynamic gestures using Deep Learning.")
st.markdown("---")

# 文件上传区
uploaded_file = st.file_uploader("", type=['mp4', 'mov', 'avi'], help="Supported formats: MP4, MOV, AVI")

if uploaded_file is not None:
    # 保存临时文件
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # 布局：左侧视频，右侧结果占位
    col_video, col_results = st.columns([1.5, 1])
    
    with col_video:
        st.subheader("📺 Video Preview")
        st.video(uploaded_file)
        
        # 启动按钮
        process_btn = st.button("🚀 Start Deep Analysis", type="primary")

    if process_btn:
        if model is None:
            st.error("Cannot proceed: Model not loaded.")
        else:
            with col_results:
                st.subheader("📊 Analysis Report")
                
                # 进度条和状态文本
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # --- 视频处理逻辑 ---
                cap = cv2.VideoCapture(tfile.name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if total_frames == 0: total_frames = 100
                
                # 采样策略：均匀提取30帧
                skip = max(int(total_frames / 30), 1)
                sequence = []
                
                status_text.markdown("**🔄 Initializing MediaPipe...**")
                
                # 使用 MediaPipe 处理
                with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    frames_processed = 0
                    
                    for i in range(30):
                        # 更新进度条
                        progress = int((i / 30) * 100)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {i+1}/30...")
                        
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
                        ret, frame = cap.read()
                        if not ret: break
                        
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        res = holistic.process(frame)
                        sequence.append(extract_keypoints(res))
                        frames_processed += 1
                
                cap.release()
                progress_bar.progress(100)
                status_text.success("✅ Feature Extraction Complete!")
                
                # 补齐数据 (Padding)
                while len(sequence) < 30:
                    sequence.append(np.zeros(258))
                
                # --- 推理逻辑 ---
                with st.spinner("🧠 Running Neural Network Inference..."):
                    input_tensor = torch.tensor(np.array([sequence]), dtype=torch.float32)
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, dim=1)[0]
                    
                    # 获取结果
                    conf, idx = torch.max(probs, 0)
                    prediction = gestures[idx.item()]
                    confidence_val = conf.item() * 100
                    
                    time.sleep(0.5)

                # --- 结果展示 (Result Dashboard) ---
                st.divider()
                
                # 1. 核心指标卡片
                st.metric(
                    label="🏆 Top Prediction",
                    value=prediction,
                    delta=f"{confidence_val:.2f}% Confidence"
                )
                
                if confidence_val > 80:
                    st.balloons() 
                
                # 2. 概率分布图 (显示所有权重)
                st.write("### 📈 Full Probability Distribution")
                
                # 整理数据
                chart_data = pd.DataFrame({
                    "Gesture": gestures,
                    "Probability": probs.numpy()
                }).sort_values(by="Probability", ascending=False)
                
                # 直接展示所有数据
                st.bar_chart(
                    chart_data, 
                    x="Gesture", 
                    y="Probability",
                    color="#FF4B4B"
                )
                
                # 3. 详细数据展开
                with st.expander("📄 View Raw Data Table"):
                    st.dataframe(chart_data.style.format({"Probability": "{:.4%}"}))

else:
    # 空状态提示
    st.info("👈 Please upload a video file from the sidebar or main area to begin.")
