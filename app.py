import streamlit as st
from functions import show_graph, show_classification_result
from transformers import pipeline
import torch

@st.cache_resource
def load_model():
    return pipeline(
        task="image-classification",
        model = "google/vit-base-patch16-224")

classifier = load_model()

st.set_page_config(layout="wide", page_title="📷이미지 분류")
st.title("이미지 분류하기")
st.markdown("---")

option = st.radio(label="넣을 이미지 방법을 선택하세요.", options=["촬영하기","사진 업로드"])

if option == "촬영하기":
    st.header("촬영하기")
    img_f = st.camera_input(label="👀여기 보세요")
    if st.button("**분류하기**"):
        if img_f is None:
            st.error("Take Photo 누른 후에 분류해주세요.")
        else:
            st.subheader("결과")
            preds = show_classification_result(img_f, classifier)
            show_graph(preds)


elif option == "사진 업로드":
    st.header("사진 업로드")
    imgs = st.file_uploader(
        label="이미지를 넣어주세요",
        label_visibility="hidden",
        type=["png", "jpg","jpeg"],
        accept_multiple_files=True
        )
    if st.button("**분류하기**"):
        if len(imgs) == 0:
            st.error("이미지를 먼저 업로드 해주세요.")
        else:
            st.subheader("결과")
            for img in imgs:
                st.image(img, width="content")
                preds = show_classification_result(img, classifier)
                show_graph(preds)














