import ultralytics
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
from scipy.io import wavfile
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO('best.pt')  # Use the correct path to your pre-trained model

# Center-align text using HTML styling
st.markdown(
    """
    <div style="text-align:center;">
        <h1>Heart Beat Sound Classifier</h1>
    </div>
    """,
    unsafe_allow_html=True
)
# Example with a file path
image_path = "images.png"
original_image = Image.open(image_path)

# Resize the image
width, height = 200 , 100
resized_image = original_image.resize((width, height))

# Display the resized image with reduced size

st.markdown(
    """
    <style>
    body {
        background-color: #F0F0F0; /* replace with your desired color code */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Display the resized image
st.image(resized_image, use_column_width=True)


st.write("Welcome to the Heartbeat Sound Classification App! This application is designed to classify heartbeat sounds "
         "from uploaded audio files. Upload a WAV file, and the app will generate a spectrogram, analyze the audio, and "
         "predict whether the heartbeat is normal or abnormal.")

st.write("### How to Use:")
st.write("1. Upload a WAV audio file using the file uploader below.")
st.write("2. Click the 'Predict' button to analyze the uploaded audio.")
st.write("3. The predictions include the classification (Normal/Abnormal) and the confidence level.")

st.write("### Note:")
st.write("This app uses a YOLO (You Only Look Once) model trained on heartbeat sound data for classification. "
         "Please upload audio files with clear heartbeat sounds for accurate predictions.")


# Streamlit app
st.title("Audio Classifier ")

# File upload
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

# Trigger button
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
