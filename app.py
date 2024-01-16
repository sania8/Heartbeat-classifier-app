import ultralytics
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
from scipy.io import wavfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
    # Remove whitespace from the top of the page and sidebar
st.markdown("""
    <style>
        .reportview-container {
            background-color: pink;
        }
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem; 
            padding-left: 0rem; 
            padding-right: 0rem;
        }

        .st-emotion-cache-ocqkz7 {
            display: flex;
            flex-wrap: wrap;
            gap: 1.5rem; /* Increase the gap between columns */
        }

        .stApp {
            margin-left: 0;
        }

        .column {
            flex: 1;
            margin: 0.5rem; /* Increase the margin for each column */
        }

        .column img {
            width: 100%;
            height: auto;
        }
       
    </style>
""", unsafe_allow_html=True)


st.title('**Heart Sound Classifier**')

# Load the YOLO model
model = YOLO('best.pt')  # Use the correct path to your pre-trained model



col1, col2 = st.columns([2, 4])  # Adjust column widths as needed

with col1:
    st.markdown(
    """
    <div style="margin-top: 0rem;">
        <h3>Introduction</h3>
        <p style="margin-top:-1rem;">Welcome! to the Heartbeat Sound Classification App . This application is designed to classify heartbeat sounds as Normal/Abnormal</p>
    </div>
    """, 
    unsafe_allow_html=True
)
    st.markdown(
    """
    <div style="margin-top: -2rem;">
        <h4>How to Use:</h4>
        <p style="margin-top:-1rem;">1. Upload a WAV audio file using the file uploader below.</p>
        <p style="margin-top:-1rem;">2. Click the 'Predict' button to analyze the uploaded audio.</p>
        <p style="margin-top:-1rem;">3. The predictions include the classification (Normal/Abnormal) and the confidence level.</p>
    </div>
    """, 
    unsafe_allow_html=True
)
with col2:
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
    if uploaded_file is not None:
        if st.button("Predict"):
            # Convert audio to spectrogram
            rate, data = wavfile.read(uploaded_file)
            plt.specgram(data, Fs=rate, aspect='auto')
            # Set axis limits to match the desired extent
            plt.xlim(0, 7)
            plt.ylim(0, 1000)

            # Remove ticks
            plt.xticks([])
            plt.yticks([])

            plt.savefig('temp.png')
            spectrogram = Image.open('temp.png')

            # Make predictions
            predictions = model.predict(spectrogram)[0].probs

            if predictions.data[0] > predictions.data[1]:
                class__ = 'Abnormal'
                conf__ = predictions.data[0]
            else:
                class__ = "Normal"
                conf__ = predictions.data[1]

            st.write("Class Probabilities:", class__, "(confidence:", round(float(conf__) * 100, 2), ")")
            # Save predictions to a text file
            with open("predictions.txt", "w") as f:
                f.write(f"Predicted class: {class__}\n")
                f.write(f"Confidence: {round(float(conf__) * 100, 2)}%\n")

                # Add class and confidence to the content of the file
                content = f"Predicted class: {class__}\nConfidence: {round(float(conf__) * 100, 2)}%"
                f.write(content)

            # Download predictions file
            st.download_button(
                label="Download Predictions",
                data=content,
                file_name="heartbeat_predictions.txt",
                mime="text/plain"
            )


st.markdown('<div style="display: none;">Invisible Markdown</div>', unsafe_allow_html=True)

