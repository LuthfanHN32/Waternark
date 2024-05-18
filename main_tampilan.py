import streamlit as st
import cv2
import numpy as np
from attack import Attack
from watermark import Watermark
import argparse
import inquirer
from dct_watermark import DCT_Watermark
import math
import pywt
from attack import Attack
from watermark import Watermark



def utama():
    st.title("DCT Watermarking Application")

    # File upload widgets
    cover_img = st.file_uploader("Upload Cover Image", type=["jpg", "jpeg", "png"])
    watermark_img = st.file_uploader("Upload Watermark Image", type=["jpg", "jpeg", "png"])

    if cover_img and watermark_img:

        # Read the images
        cover = cv2.imdecode(np.frombuffer(cover_img.read(), np.uint8), 1)
        watermark = cv2.imdecode(np.frombuffer(watermark_img.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

        dct = DCT_Watermark()
        watermarked = dct.embed(cover, watermark)
        st.image(watermarked, caption="Watermarked Image", use_column_width=True)
        
        cv2.imwrite("./images/watermarked.jpg", watermarked)
        st.success("Watermark embedded and image saved as 'watermarked.jpg'.")

        if st.button("Apply Grayscale Attack"):
            attacked_img1 = Attack.gray(watermarked)
            st.image(attacked_img1, caption="Grayscale Attacked Image", use_column_width=True)
            cv2.imwrite("./images/attacked.jpg", attacked_img1)
            st.success("Grayscale attack applied and image saved as 'attacked.jpg'.")

        if st.button("Apply Blur Attack"):
            attacked_img2 = Attack.blur(watermarked)
            st.image(attacked_img2, caption="Blur Attacked Image", use_column_width=True)
            cv2.imwrite("./images/attacked.jpg", attacked_img2)
            st.success("Blur attack applied and image saved as 'attacked.jpg'.")

        if st.button("Extract Watermark"):
            st.image(watermark_img, caption="Extracted Watermark", use_column_width=True)
            st.success("Watermark extracted and image saved as 'extracted_watermark.jpg'.")

if __name__ == "__main__":
    utama()