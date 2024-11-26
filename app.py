import streamlit as st
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image

st.header("Fashion _YOLO_ Classification")

st.write("""This model can classify the following fashion products:""")
st.write("""Shirt, Jeans, Tshirt, Tops, Socks, Briefs and many more...""")

uploaded_file = st.file_uploader("Choose a file to upload:", type=["jpg", "jpeg", "png"])



model_trained = YOLO("results/runs/classify/train/weights/best.pt")



if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(uploaded_file, caption="Uploaded Image") 
    results = model_trained(image)

    for result in results:
        top1 = result.probs.top1
        classes = result.names
        top1_class_name = classes[top1]
        print(top1_class_name)
        st.write(f"Result is: {top1_class_name}")


