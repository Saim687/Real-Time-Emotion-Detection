

import numpy as np
import cv2
import streamlit as st
import av
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
from pathlib import Path
import logging
import datetime
import os

st.set_page_config(page_title="Emotion Detection App", page_icon="😊", layout="centered", initial_sidebar_state="expanded")

# Create a unique log file with the current timestamp
log_filename = f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Application started.")

# Emotion Detection Configuration and Initialization
emotion_labels = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad', 4: 'Surprise'}

@st.cache_resource
def load_detection_resources():
    with open('models/face_emotion_model1.json', 'r') as model_file:
        model_structure = model_file.read()
    model = model_from_json(model_structure)
    model.load_weights("models/face_emotion_model1.h5")

    cascade = cv2.CascadeClassifier('models/face_haarcascade_frontalface_default.xml')
    if cascade.empty():
        raise RuntimeError("Failed to load Haar Cascade for face detection.")

    return model, cascade


try:
    emotion_model, face_detector = load_detection_resources()
    logging.info("Model and Haar Cascade loaded successfully.")
except Exception as e:
    logging.error(f"Resource initialization failed: {e}")
    st.error(f"Initialization failed: {e}")


frame_counter = 0
last_detections = []

# WebRTC Configuration for Streaming
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}
    ]
})


# Custom CSS for Styling
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background: #1a1a2e;
        color: #eaeaea;
        padding-bottom: 100px; /* Space for footer */
    }
    .stButton button {
        background: linear-gradient(to right, #6a11cb, #2575fc);
        color: white;
        padding: 10px 20px;
        border-radius: 20px;
        transition: transform 0.3s;
    }
    .stButton button:hover {
        transform: scale(1.1);
    }
    img {
        display: block;
        margin: 20px auto;
        border-radius: 10px;
        width: 300px;
        height: 300px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #24252a;
        color: white;
        text-align: center;
        padding: 15px 0;
        font-size: 14px;
        font-family: 'Arial', sans-serif;
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.3);
    }
    .footer a {
        color: #f9c74f;
        text-decoration: none;
        font-weight: bold;
    }
    .footer a:hover {
        color: #f9844a;
    }
    .icons {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 10px;
    }
    .icon {
        text-align: center;
        color: white;
    }
    .icon img {
        width: 60px;
        height: 60px;
        display: block;
        margin: 0 auto 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def process_video_frame(frame: av.VideoFrame) -> av.VideoFrame:
    global frame_counter, last_detections
    frame_bgr = frame.to_ndarray(format="bgr24")
    
    try:
        frame_counter += 1

        # Run heavier detection/prediction every 3rd frame for smoother streaming on CPU.
        if frame_counter % 3 == 0:
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            updated = []
            for (x, y, w, h) in faces:
                face_roi = frame_gray[y:y + h, x:x + w]
                face_roi = cv2.resize(face_roi, (48, 48))
                face_roi = face_roi.astype('float32') / 255.0
                face_roi = img_to_array(face_roi)
                face_roi = np.expand_dims(face_roi, axis=0)

                # THE FIX: Call the model directly instead of using .predict()
                # Setting training=False ensures it runs in pure inference mode
                predictions_tensor = emotion_model(face_roi, training=False)
                
                # Convert the tensor back to a numpy array and get the first result
                predictions = predictions_tensor.numpy()[0] 
                
                max_index = np.argmax(predictions)
                emotion = emotion_labels[max_index]
                updated.append((x, y, w, h, emotion))

            last_detections = updated

        # Draw the boxes and text
        for (x, y, w, h, emotion) in last_detections:
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame_bgr, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    except Exception as e:
        logging.error(f"Error during frame processing: {e}")

    return av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
    
img ="images/emotion_image.jpg"

# App Functions
def display_home_page():
    st.write("Welcome to the **Real-Time Emotion Detection Application**! This application uses cutting-edge deep learning technology to recognize facial emotions directly from a webcam feed.")

    st.image(img, caption="Emotion Detection in Real-Time", use_column_width=False)
    
    st.markdown("### 💡 **Features**:")
    st.markdown("""
    - Detect emotions like **Angry**, **Happy**, **Neutral**, **Sad**, and **Surprise** in real time.
    - Powered by a **pre-trained Convolutional Neural Network (CNN)** model.
    """)

    st.markdown("### 🔥 **Why Use This?**")
    st.markdown("""
    - Understand non-verbal cues in real-time.
    - Enhance interaction-based applications, such as:
        - Mental health analysis 🧠
        - Gaming 🎮
        - Customer interaction insights 📈
    """)

    st.markdown("---")
    
    st.markdown("### 💡 **How to Navigate**:")
    st.markdown("""
    - **Open the Sidebar**:
        - Look at the left side of the screen.
        - If the sidebar is not visible, click the **`>` arrow** in the top-left corner to open it.
    - **Select the Page**:
        - In the sidebar, locate the dropdown menu labeled **"Choose a page"**.
        - Click the dropdown and select **"Web Emotion Detection"**.
    """)
    
    st.write("🎥 Once you follow these steps, you'll be redirected to the Web Emotion Detection page, where you can start detecting emotions in real time using your webcam.")

    st.markdown("---")
    



def run_emotion_detection():
    st.header("📷 Webcam Emotion Detection")
    st.write("Click **Start** to begin real-time emotion detection.")
    
    webrtc_streamer(
        key="emotion_recognition",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=process_video_frame,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,

    )

def explore_projects():
    st.header("🔍 Explore Projects")
    st.write("We develop cutting-edge projects in AI, ML, and more. Visit our [project page] to learn more.")

def about_section():
    st.subheader("📚 About This Application")
    st.write("This real-time emotion detection application allows users to identify emotions based on facial expressions using a pre-trained CNN model.")
    

    # Add icons horizontally on top of the page (Repo, Stars, LinkedIn)
    st.markdown("""
    <div class="icons" style="display: flex; justify-content: center; gap: 40px;">
        <div class="icon">
            <a href="https://github.com/Saim687/Real-Time-Emotion-Detection.git" target="_blank">
                <img src="https://i.ibb.co/fYynxNn/git.png" alt="GitHub" width="100px" height="100px">
                GitHub
            </a>
        </div>
        <div class="icon">
                <img src="https://i.ibb.co/RBsm6TR/image.png" alt="LinkedIn" width="40px" height="40px">
                LinkedIn
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Project functionalities and description
    st.markdown("""
    ### Functionalities:
    - **Real-time emotion recognition:** Detect emotions like Angry, Happy, Neutral, Sad, and Surprise.
    - **Pre-trained model:** Utilizes a CNN trained on facial expression datasets for accurate emotion detection.
    - **Webcam Integration:** Directly analyzes webcam feeds for on-the-spot emotion recognition.
    - **Multiplatform Support:** Runs seamlessly on any modern web browser with Streamlit.

    ### Libraries Used:
    - **OpenCV:** Used for real-time video capture and face detection.
    - **TensorFlow/Keras:** The underlying deep learning framework used for emotion classification.
    - **Streamlit:** Powers the web interface for easy deployment and interactivity.
    - **NumPy:** Essential for data processing and matrix manipulations.
    - **Streamlit WebRTC:** Enables real-time video streaming and processing.

    """)

    st.markdown("""
    ### How It Works:
    - The system captures video from the webcam, processes it frame by frame, and detects faces using Haar cascades.
    - Detected faces are passed through the emotion classification model, which identifies the predominant emotion.
    - The predicted emotion is displayed in real-time on the video feed.

    ### Why Use This?
    - **Mental Health Monitoring:** Helps in analyzing emotional trends for better mental health management.
    - **Interactive Gaming:** Emotion-driven responses in games for immersive experiences.
    - **Customer Insights:** Analyze emotions in customer service environments for improved interaction.

    """)


def footer():
    st.markdown("""
        <div class="footer">
            Crafted by Saim & Abdullah | <a href="https://github.com/Saim687/Real-Time-Emotion-Detection.git" target="_blank">GitHub</a>
        </div>
    """, unsafe_allow_html=True)

# Main Application Runner
def run_app():

    st.title("🌟 Real-Time Emotion Detection 🌟")

    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = "Home"

    st.sidebar.title("Emotion Detection App")
    app_pages = ["Home", "Webcam Detection", "About Project", "Explore Other Projects"]
    default_index = app_pages.index(st.session_state["selected_page"])
    choice = st.sidebar.selectbox("Choose a page", app_pages, index=default_index, key="page_selector")
    st.session_state["selected_page"] = choice

    if st.session_state["selected_page"] == "Home":
        display_home_page()
    elif st.session_state["selected_page"] == "Webcam Detection":
        run_emotion_detection()
    elif st.session_state["selected_page"] == "Explore Other Projects":
        explore_projects()
    elif st.session_state["selected_page"] == "About Project":
        about_section()
    
    footer()

if __name__ == "__main__":
    try:
        run_app()
        logging.info("Application running.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        st.error(f"An unexpected error occurred: {e}")