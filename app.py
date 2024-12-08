import streamlit as st
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import cv2 as cv
import tempfile

st.header("Fashion _YOLO_ Classification")

st.write("""This model can classify the following fashion products:""")
st.write("""Cardigans, Denims, Shirts and many more...""")

uploaded_file = st.file_uploader("Choose a file to upload:", type=["jpg", "jpeg", "png", "mp4"])

model_trained = YOLO("newbest.pt")

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(uploaded_file, caption="Uploaded Image") 
#     results = model_trained(image)

#     for result in results:
#         top1 = result.probs.top1
#         classes = result.names
#         top1_class_name = classes[top1]
#         print(top1_class_name)
#         st.write(f"Result is: {top1_class_name}")

if uploaded_file is not None:
    preds = set()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.read())

    st.video(temp_file)
    video = cv.VideoCapture(temp_file)

    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        resized_frames = cv.resize(frame, (224, 224))

        results = model_trained(resized_frames)

        for result in results:
            top1 = result.probs.top1
            classes = result.names
            top1_class_name = classes[top1]
            if top1 > 0.5:
                preds.add(top1_class_name)

    st.write(f"Tags are:{preds}")