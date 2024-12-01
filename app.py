import streamlit as st
import torch
from PIL import Image
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

st.title("🔍 YOLOv5 Object Detection App")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Run YOLOv5 model on the image
    st.write("⏳ Detecting objects...")
    results = model(image)

    # Display results
    st.write("✅ Detection Complete!")
    st.image(np.squeeze(results.render()), caption='Detected Objects', use_column_width=True)

    # Display detected labels and confidence scores
    st.write("📋 **Detection Results**:")
    for result in results.xyxy[0]:  # xyxy format: xmin, ymin, xmax, ymax, confidence, class
        st.write(f"- {model.names[int(result[5])]}: {result[4]:.2f} confidence")

st.sidebar.write("💡 **Tip:** Upload an image and YOLOv5 will detect objects in it!")
