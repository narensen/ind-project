import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from transformers import pipeline
import os

# --------------------- Page Config ---------------------
st.set_page_config(
    page_title="Face Detection & Classification with Recognition (PyTorch)",
    page_icon="👤",
    layout="centered"
)

# --------------------- Custom CSS ---------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        text-align: center;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# --------------------- Initialize Session State ---------------------
if 'known_encodings' not in st.session_state:
    st.session_state.known_encodings = []  # List of (name, embedding)

# --------------------- Load PyTorch Models (once) ---------------------
@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.info(f"Using device: {device}")
    
    # Face detection
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    
    # Face recognition embeddings
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # Separate HuggingFace pipelines for classification (using models with safetensors)
    age_pipe = pipeline(
        "image-classification",
        model="nateraw/vit-age-classifier",
        device=0 if torch.cuda.is_available() else -1
    )
    
    gender_pipe = pipeline(
        "image-classification",
        model="dima806/fairface_gender_image_detection",
        device=0 if torch.cuda.is_available() else -1
    )
    
    emotion_pipe = pipeline(
        "image-classification",
        model="mo-thecreator/vit-Facial-Expression-Recognition",
        device=0 if torch.cuda.is_available() else -1
    )
    
    return mtcnn, resnet, age_pipe, gender_pipe, emotion_pipe, device

# --------------------- Title ---------------------
st.markdown("<h1 class='main-header'>👤 PyTorch Face Analysis & Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:1.2rem;'>Upload or capture a photo to detect faces, predict Age • Gender • Emotion, and recognize known people</p>", unsafe_allow_html=True)

# --------------------- Sidebar Options ---------------------
with st.sidebar:
    st.header("⚙️ Settings")
    enable_age = st.checkbox("Predict Age", value=True)
    enable_gender = st.checkbox("Predict Gender", value=True)
    enable_emotion = st.checkbox("Predict Emotion", value=True)
    enable_recognition = st.checkbox("Enable Face Recognition", value=True)
    
    # --------------------- Known Faces Management ---------------------
    st.header("👥 Known Faces Database")
    if st.session_state.known_encodings:
        st.subheader("Current Database")
        known_names = {}
        for name, _ in st.session_state.known_encodings:
            known_names[name] = known_names.get(name, 0) + 1
        for name, count in known_names.items():
            st.write(f"• {name} ({count} images)")
    
    with st.expander("Add New Known Face"):
        name = st.text_input("Person's Name")
        uploaded_file = st.file_uploader("Upload Face Image", type=['jpg', 'jpeg', 'png'], key="upload_known")
        if st.button("Add to Database") and uploaded_file and name:
            try:
                image = Image.open(uploaded_file).convert('RGB')
                mtcnn, resnet, _, _, _, device = load_models()
                img_tensor = mtcnn(image).unsqueeze(0).to(device)
                embedding = resnet(img_tensor).detach().cpu().numpy().flatten()
                st.session_state.known_encodings.append((name, embedding))
                st.success(f"Added encoding for {name}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if st.button("Clear Database"):
        st.session_state.known_encodings = []
        st.success("Database cleared!")

# --------------------- Load Models ---------------------
mtcnn, resnet, age_pipe, gender_pipe, emotion_pipe, device = load_models()

# --------------------- Main App: Snapshot-based Analysis ---------------------
st.markdown("### 📸 Capture or Upload Photo")
uploaded_image = st.camera_input("Take a photo") or st.file_uploader("Or upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_image:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption="Original Image", use_column_width=True)
    
    # Detect faces
    boxes, probs = mtcnn.detect(image)
    if boxes is not None:
        with torch.no_grad():
            # Align and get embeddings
            aligned_faces = mtcnn(image)
            if aligned_faces is not None:
                aligned_faces = aligned_faces.unsqueeze(0).to(device)
                embeddings = resnet(aligned_faces).detach().cpu().numpy()
                
                # Process each face
                results = []
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    if prob < 0.9:  # Confidence threshold
                        continue
                    
                    # Crop face from original
                    x1, y1, x2, y2 = [int(b) for b in box]
                    face_crop = np.array(image)[y1:y2, x1:x2]
                    face_pil = Image.fromarray(face_crop).resize((224, 224))  # Resize for HF models
                    
                    # Recognition
                    face_name = "Unknown"
                    if enable_recognition and st.session_state.known_encodings:
                        distances = np.linalg.norm(embeddings[i] - np.array([enc for _, enc in st.session_state.known_encodings]), axis=1)
                        min_dist = np.min(distances)
                        if min_dist < 0.8:  # Tolerance for VGGFace2
                            match_idx = np.argmin(distances)
                            face_name = st.session_state.known_encodings[match_idx][0]
                    
                    # Classifications
                    age_group = ""
                    gender = ""
                    emotion = ""
                    
                    if enable_age:
                        age_results = age_pipe(face_pil)
                        age_group = age_results[0]['label']  # e.g., '0-2', '4-6', etc.
                    
                    if enable_gender:
                        gender_results = gender_pipe(face_pil)
                        gender = gender_results[0]['label'].title()  # 'Male' or 'Female'
                    
                    if enable_emotion:
                        emo_results = emotion_pipe(face_pil)
                        emotion = emo_results[0]['label'].replace('_', ' ').title()  # e.g., 'Happy', 'Sad'
                    
                    results.append({
                        'name': face_name,
                        'age': age_group,
                        'gender': gender,
                        'emotion': emotion,
                        'box': box,
                        'prob': prob
                    })
                
                if results:
                    # Annotate image
                    draw_image = np.array(image)
                    for res in results:
                        x1, y1, x2, y2 = [int(b) for b in res['box']]
                        cv2.rectangle(draw_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        text = f"{res['name']} ({res['prob']:.2f})"
                        if enable_age: text += f" | Age: {res['age']}"
                        if enable_gender: text += f" | {res['gender']}"
                        if enable_emotion: text += f" | {res['emotion']}"
                        
                        cv2.rectangle(draw_image, (x1, y1-35), (x2, y1), (0, 255, 0), -1)
                        cv2.putText(draw_image, text, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    annotated_pil = Image.fromarray(cv2.cvtColor(draw_image, cv2.COLOR_RGB2BGR))
                    st.image(annotated_pil, caption="Analyzed Image", use_column_width=True)
                    
                    # Results box
                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    for i, res in enumerate(results, 1):
                        st.write(f"**Face {i}:** {res['name']} | Confidence: {res['prob']:.2f}")
                        if enable_age: st.write(f"Age Group: {res['age']}")
                        if enable_gender: st.write(f"Gender: {res['gender']}")
                        if enable_emotion: st.write(f"Emotion: {res['emotion']}")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("No faces detected with sufficient confidence.")
    else:
        st.warning("No faces detected.")

# --------------------- Installation Note ---------------------
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🚀 Installation
```bash
pip install streamlit torch torchvision facenet-pytorch transformers pillow opencv-python
                    """)