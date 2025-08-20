import streamlit as st
import torch
import numpy as np
import os
import cv2
import tempfile
import pandas as pd
import altair as alt
import re
from transformers import AutoFeatureExtractor, AutoModelForVideoClassification
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Action Recognition")

# Sidebar: Model path selection (for display only)

st.sidebar.write("## Model Path Selection")
available_models = {
    "HuggingFace Pretrained (Default)": "facebook/timesformer-base-finetuned-k400",
    "SGD LR=0.001": "Results/lr - 0.001/SGD/timesformer_lr000100_f8_s32_20250805_134346.pth",
    "SGD LR=0.0005": "Results/lr - 0.0005/SGD/timesformer_lr000050_f8_s32_20250804_202022.pth"
}
model_choice = st.sidebar.selectbox("Select Model (for reference):", list(available_models.keys()), index=0)
model_path = available_models[model_choice]
st.sidebar.write(f"Selected model path: {model_path}")

# Add Load Model button
if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False

load_model_btn = st.sidebar.button("Load Model")
if load_model_btn:
    with st.sidebar:
        with st.spinner("Loading model, please wait..."):
            model, extractor = AutoModelForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400"), AutoFeatureExtractor.from_pretrained("facebook/timesformer-base-finetuned-k400")
            st.session_state['model_loaded'] = True




# Only show video upload after model is loaded
if st.session_state.get('model_loaded', False):
    st.write("## Upload and Process Video 🎥")
    uploaded_file = st.file_uploader("Upload a video file:", type=["mp4", "avi", "mov"])
else:
    st.info("Please load the model using the button in the sidebar before uploading a video.")



@st.cache_resource
def load_model():
    model = AutoModelForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
    extractor = AutoFeatureExtractor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    return model, extractor

model, extractor = load_model()
model.eval()

def extract_frames_from_video(video_path, output_folder, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // num_frames, 1)
    frame_count = 0
    saved_frames = 0
    while cap.isOpened() and saved_frames < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved_frames + 1:04d}.jpg")
            frame = cv2.resize(frame, (224, 224))
            cv2.imwrite(frame_path, frame)
            saved_frames += 1
        frame_count += 1
    cap.release()

st.write("## Action Recognition App")
st.write("Upload a video to predict the action using a pre-trained model.")

st.write("""
This app allows you to upload a video, converts it into frames, and predicts the action using a pre-trained model.
We use **TimeSformer**, a state-of-the-art video transformer model, which processes video frames as a sequence of images and captures temporal relationships to predict actions effectively.
Experience seamless action recognition with visualizations and confidence scores.
""")

col1, col2 = st.columns(2)

if uploaded_file:
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        col1.write("### Uploaded Video")
        col1.video(video_path)
        st.info("Extracting frames from the video...")
        extract_frames_from_video(video_path, temp_dir, num_frames=8)
        folder_path = temp_dir
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])[:8]
        frames = []
        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)
            frame = cv2.imread(img_path)
            frames.append(frame)
        if len(frames) < 8:
            st.warning("The video must contain enough frames to extract 8 frames.")
        else:
            inputs = extractor([frames], return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                top_prob, top_index = torch.max(probs, dim=-1)
                # Get top-5 predictions
                top5_probs, top5_indices = torch.topk(probs[0], 5)
                top5_labels = [model.config.id2label[idx.item()] for idx in top5_indices]
                top5_confidences = [prob.item() * 100 for prob in top5_probs]
            col2.write("### Predicted Action")
            action_label = model.config.id2label[top_index.item()]
            confidence = top_prob.item() * 100
            col2.markdown(
                f"""
                <div style="background-color: #f9f9f9; padding: 10px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 10px;">
                    <h2 style="font-size: 24px; color: #4CAF50;">{action_label}</h2>
                    <p style="font-size: 16px; color: #777;">Confidence: {confidence:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Top-5 prediction bar chart (professional style)
            col2.write("### Top-5 Predicted Classes")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            # Dark background and vibrant bar colors
            fig.patch.set_facecolor('#181818')
            ax.set_facecolor('#222831')
            palette = ["#00ADB5", "#FBC02D", "#D32F2F", "#7B1FA2", "#388E3C"]
            bars = ax.barh(top5_labels[::-1], top5_confidences[::-1], color=palette[::-1], edgecolor='none', height=0.6)
            ax.set_xlabel("Confidence (%)", fontsize=13, color="#EEEEEE")
            ax.set_xlim(0, 100)
            ax.set_title("Top-5 Predicted Classes", fontsize=16, color="#00ADB5", fontweight='semibold', pad=15)
            ax.tick_params(axis='y', labelsize=12, colors="#EEEEEE")
            ax.tick_params(axis='x', labelsize=11, colors="#EEEEEE")
            # Add value labels inside bars for clarity
            for i, (bar, v) in enumerate(zip(bars, top5_confidences[::-1])):
                ax.text(v - 5 if v > 10 else v + 2, bar.get_y() + bar.get_height()/2,
                        f"{v:.2f}%", va='center', ha='right' if v > 10 else 'left',
                        fontsize=12, color='white', fontweight='bold')
            # Remove spines for a clean look
            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_color('#00ADB5')
            ax.grid(axis='x', linestyle='--', alpha=0.2, color='#393E46')
            plt.tight_layout()
            col2.pyplot(fig)
        # Show the 8 extracted frames below the video
        st.write("### Extracted Frames")
        frame_cols = st.columns(8)
        for i, frame in enumerate(frames):
            # Convert BGR (OpenCV) to RGB for correct color display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_cols[i].image(rgb_frame, caption=f"Frame {i+1}", use_container_width=True)

