import numpy as np
import pickle
import streamlit as st

model = "Package_prediction_model.pkl"
with open(model, "rb") as f:
    data = pickle.load(f)
    print(data)

image=st.image("https://presidencyuniversity.in/uploads/cms_images/6829baa7694631747565223.webp",use_container_width=True)


st.subheader("Package Prediction based on CGPA")

cgpa = st.text_input("Enter your Cgpa")


if st.button("Predict"):
    features = np.array([[cgpa]],dtype=np.float64)
    results = data.predict(features)
    st.write(f"Predicted Package ---> {results[0]:.3f} LPA")
