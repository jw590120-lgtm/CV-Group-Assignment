import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import pandas as pd
import tempfile
import os

# --- 1. 配置页面 ---
st.set_page_config(page_title="Gesture Recognition Demonstration", page_icon="🖐️")
st.title("🖐️ Intelligent gesture recognition system")
st.write("Upload a video, and AI will recognize which type of gesture it is and display the proportion of all possibilities.")

# --- 2. 定义模型架构 ---
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

# --- 3. 工具函数 ---
mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# --- 4. 加载模型 ---
@st.cache_resource
def load_model():
    # 这里定义你的15个手势标签
    # 这里我暂时用 User 1 到 15 代替，需要根据 BIM 文件夹修改
    gestures = [f"Gesture {i}" for i in range(1, 16)] 
    
    device = torch.device("cpu")
    model = BiLSTMAttention(input_size=258, hidden_size=128, num_classes=len(gestures))
    
    try:
        model.load_state_dict(torch.load("trained_model.pth", map_location=device))
        model.eval()
    except FileNotFoundError:
        st.error("The model file trained_model.pth cannot be found. Please ensure that it has been uploaded.")
        return None, None
        
    return model, gestures

model, gestures = load_model()

# --- 5. 视频处理逻辑 ---
uploaded_file = st.file_uploader("Please upload the video file (mp4, mov, avi)", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None and model is not None:
    # 保存临时文件供OpenCV读取
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    # 显示视频
    st.video(uploaded_file)
    
    if st.button("Start recognition"):
        with st.spinner("Analyzing gestures frame by frame..."):
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            skip = max(int(total_frames / 30), 1)
            
            sequence = []
            
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                for i in range(30):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = holistic.process(frame)
                    sequence.append(extract_keypoints(res))
            
            cap.release()
            
            # 补齐帧数
            while len(sequence) < 30:
                sequence.append(np.zeros(258))
                
            # 推理
            input_tensor = torch.tensor(np.array([sequence]), dtype=torch.float32)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)[0] # 获取概率分布
                
            # --- 6. 结果展示 ---
            # 获取最高置信度的结果
            conf, idx = torch.max(probs, 0)
            predicted_gesture = gestures[idx.item()]
            confidence_score = conf.item() * 100
            
            st.success(f"Recognition result: **{predicted_gesture}**")
            st.info(f"Confidence level: **{confidence_score:.2f}%**")
            
            # 展示所有15个动作的比重（柱状图）
            st.write("### Action probability distribution")
            
            # 构建DataFrame用于图表
            probs_np = probs.numpy()
            chart_data = pd.DataFrame({
                "Gesture": gestures,
                "Probability": probs_np
            }).sort_values(by="Probability", ascending=False)
            
            # 交互式柱状图
            st.bar_chart(chart_data, x="Gesture", y="Probability", color="#FF4B4B")