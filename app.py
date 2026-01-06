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
    /* Start 按钮样式 (Primary - 红色) */
    div.stButton > button:first-child {
        width: 100%; border-radius: 10px; height: 3em; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ===========================
# 2. 核心模型定义
# ===========================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def blur_face_region(image, results):
    if not results.pose_landmarks: return image
    h, w, _ = image.shape
    face_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x_coords = [int(results.pose_landmarks.landmark[i].x * w) for i in face_indices]
    y_coords = [int(results.pose_landmarks.landmark[i].y * h) for i in face_indices]
    
    if not x_coords or not y_coords: return image
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    padding_w = int((x_max - x_min) * 0.5)
    padding_h = int((y_max - y_min) * 0.5)
    
    x_min = max(0, x_min - padding_w)
    x_max = min(w, x_max + padding_w)
    y_min = max(0, y_min - padding_h)
    y_max = min(h, y_max + padding_h)
    
    face_roi = image[y_min:y_max, x_min:x_max]
    if face_roi.size > 0:
        blurred_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)
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
    
    # --- 功能区 1: 验证模式 ---
    st.write("### 🎯 Validation Mode")
    st.caption("Select the actual gesture to verify prediction correctness.")
    
    # 加载词汇表用于下拉菜单
    _, gestures_list, _ = load_model()
    if gestures_list:
        ground_truth_options = ["❓ Select Ground Truth..."] + gestures_list
        ground_truth = st.selectbox("Actual Gesture (Truth):", ground_truth_options)
    else:
        ground_truth = "❓ Select Ground Truth..."

    # 【修改点 1】添加柱状图颜色说明
    if ground_truth != "❓ Select Ground Truth...":
        st.info(f"""
        **Chart Legend for '{ground_truth}':**
        
        🟢 **Green Bar:** Prediction matches **{ground_truth}**.
        
        🔴 **Red Bar:** Prediction does NOT match.
        """)
    else:
        st.caption("ℹ️ Select a gesture above to enable validation color coding (Green/Red).")

    st.markdown("---")
    
    # --- 功能区 2: 隐私保护 ---
    st.write("### 🛡️ Privacy Settings")
    enable_blur = st.checkbox("🙈 Blur Faces", value=False)
    
    st.markdown("---")
    
    # 模型状态
    model, gestures, status = load_model()
    if status == "Loaded":
        st.success("Model Status: **Active** ✅")
    else:
        st.error(f"Model Status: **{status}** ❌")
        st.warning("Please upload 'trained_model.pth'.")
    
    st.markdown("---")
    st.caption("CV Group Assignment 2025")

# ===========================
# 4. 主界面设计
# ===========================

st.markdown("# 🎬 AI Gesture Analysis Studio")
st.markdown("#### Upload a video (Max 3s) to identify dynamic gestures using Deep Learning.")
st.markdown("---")

uploaded_file = st.file_uploader("", type=['mp4', 'mov', 'avi'], help="Limit: 3 seconds max.")

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap_check = cv2.VideoCapture(tfile.name)
    fps = cap_check.get(cv2.CAP_PROP_FPS)
    frame_count = cap_check.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    cap_check.release()
    
    if duration > 3.5:
        st.error(f"⛔ **Video too long!** ({duration:.2f}s)")
        st.warning("Please upload a video shorter than **3 seconds**.")
    else:
        col_video, col_results = st.columns([1.5, 1])
        
        with col_video:
            st.subheader("📺 Video Preview")
            if enable_blur:
                st.warning("🔒 **Raw Video Hidden** (Privacy Mode On)")
                st.image("https://placehold.co/600x400/333/FFF?text=Privacy+Mode+Active", use_column_width=True)
            else:
                st.video(uploaded_file)
            
            # 【修改点 2】按钮布局：Start Analysis (左/主) + Feedback (右/灰)
            btn_col1, btn_col2 = st.columns([3, 1])
            with btn_col1:
                process_btn = st.button("🚀 Start Deep Analysis", type="primary")
            with btn_col2:
                # 这是一个灰色的按钮，点击无反应，纯装饰，后续pre用到
                st.button("💬 Feedback")

        if process_btn:
            if model is None:
                st.error("Cannot proceed: Model not loaded.")
            else:
                with col_results:
                    st.subheader("📊 Analysis Report")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    frame_window = st.empty()
                    
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
                            
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            res = holistic.process(frame)
                            
                            display_frame = frame.copy()
                            if enable_blur:
                                display_frame = blur_face_region(display_frame, res)
                                
                            mp_drawing.draw_landmarks(display_frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                            mp_drawing.draw_landmarks(display_frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                            
                            frame_window.image(display_frame, channels="RGB", caption=f"Processing...", use_column_width=True)
                            
                            sequence.append(extract_keypoints(res))
                    
                    cap.release()
                    progress_bar.progress(100)
                    status_text.success("✅ Complete!")
                    
                    while len(sequence) < 30:
                        sequence.append(np.zeros(258))
                    
                    with st.spinner("🧠 Analyzing Pattern..."):
                        input_tensor = torch.tensor(np.array([sequence]), dtype=torch.float32)
                        with torch.no_grad():
                            output = model(input_tensor)
                            probs = torch.softmax(output, dim=1)[0]
                        
                        conf, idx = torch.max(probs, 0)
                        prediction = gestures[idx.item()]
                        confidence_val = conf.item() * 100
                        time.sleep(0.5)

                    st.divider()
                    
                    # --- 颜色与结果逻辑 ---
                    bar_color = "#FF4B4B" # 默认红
                    
                    # 【修改点 3】根据 Ground Truth 显示红/绿背景框
                    if ground_truth != "❓ Select Ground Truth...":
                        if prediction.lower() == ground_truth.lower():
                            # 预测正确：绿色条 + 绿色Success框
                            bar_color = "#2ECC71" 
                            st.success(f"✅ **Correct!** The model predicted **{prediction}** ({confidence_val:.1f}%) which matches your selection.")
                        else:
                            # 预测错误：红色条 + 红色Error框 (背景是红色的)
                            bar_color = "#E74C3C" 
                            st.error(f"❌ **Incorrect.** Expected **{ground_truth}**, but model predicted **{prediction}**.")
                    else:
                        # 没选 Ground Truth，显示中性结果
                        st.metric(label="Prediction Result", value=prediction, delta=f"{confidence_val:.2f}% Confidence")
                    
                    st.write("### 📈 Probability Distribution")
                    chart_data = pd.DataFrame({
                        "Gesture": gestures,
                        "Probability": probs.numpy()
                    }).sort_values(by="Probability", ascending=False)
                    
                    # 动态设置图表颜色
                    st.bar_chart(chart_data, x="Gesture", y="Probability", color=bar_color)
                    
                    with st.expander("📄 View Raw Data"):
                        st.dataframe(chart_data.style.format({"Probability": "{:.4%}"}))

else:
    st.info("👈 Please upload a video file (Max 3s).")
