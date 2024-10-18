import streamlit as st

from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageFilter
import SimpleITK as sitk

import pandas as pd
import numpy as np
import pickle
import joblib
import keras
from scipy.ndimage import binary_fill_holes

from functions import featurexImg, load_image


# function to enable file uploader when the uploaded file is deleted
def enable():
    st.session_state.disabled = False


# To hide the sidebar
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Push the file uploader and the selectbox more down
st.markdown(
    """<style>
            .st-emotion-cache-up131x, .st-emotion-cache-ue6h4q {
                margin-top: 50px;
            }
        </style>""",
    unsafe_allow_html=True)

# Load CSS file
with open("style.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

is_uploaded = False  # variable to control if the file has been uploaded
mask_created = False  # variable to control if the mask has been created

# First container
with st.container():
    col11, col12, col13 = st.columns([0.3, 0.5, 0.2], gap='large')
    with col11:
        st.title('SmartScan AI')
        st.markdown(f"""<p style="font-size: 20px;">Welcome <b>{st.session_state["name"]}</b>!</p>""",
                    unsafe_allow_html=True)

        # Logout button
        if st.button("Logout"):
            st.session_state.disabled = False
            st.session_state.authentication_status = None
            st.switch_page("login.py")

    with col12:
        if 'disabled' not in st.session_state:
            st.session_state.disabled = False

        # Image file uploader
        bg_image = st.file_uploader("Choose image to analyze:", type=["png", "jpg"], disabled=st.session_state.disabled,
                                    on_change=enable)

    with col13:
        if bg_image is not None:
            st.session_state.disabled = True
            is_uploaded = True  # variable to control that the image has been uploaded
            auto_generated = False   # variable to control whether the mask is automatically generated

            # Resizing the uploaded image
            size_up = 512
            img = Image.open(bg_image).resize((size_up, size_up))

            # Selectbox
            drawing_mode = st.selectbox(
                "Drawing mode:", ("freedraw", "polygon", "automatic"),
                help="- freedraw: manually draw the region\n- polygon: click to add vertices, right click to close the polygon\n- automatic: select region automatically"
            )

            # Button for results
            st.button('Show result', key="show_res")

# Container to display final result
result_container = st.container()

if is_uploaded:
    with st.container():
        col21, col22 = st.columns([0.5, 0.5], gap="large")

        with col21:
            st.markdown('<p style="font-size: 20px;">Your scan:</p>',
                        unsafe_allow_html=True)
            if drawing_mode != "automatic":
                # Create a canvas to draw on
                width, height = img.size

                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=3,
                    stroke_color="#E5FB11",
                    background_image=img if bg_image else None,
                    update_streamlit=True,
                    width=width,
                    height=height,
                    drawing_mode=drawing_mode,
                    key="canvas",
                )
                image_data = canvas_result.image_data
            else:
                image_data = None
                st.image(img)  # display loaded image
                if 'mask' not in st.session_state:
                    # Load the segmentation model and perform automatic segmentation
                    unet = keras.models.load_model(
                        'pages/segmUnet100_aug.keras')
                    size_down = 128
                    threshold = 1e-1
                    segm = load_image(np.array(img), size_down)
                    mask = unet.predict(np.expand_dims(segm, 0), verbose=0)[0] > threshold
                    st.session_state.mask = mask
                else:
                    mask = st.session_state.mask

                auto_generated = True

        with col22:
            st.markdown('<p style="font-size: 20px;">The selected region will appear here:</p>',
                        unsafe_allow_html=True)

            if image_data is not None or auto_generated:
                # If not automatically generated, create the mask from the image_data
                if image_data is not None and not auto_generated:
                    mask = image_data[:, :, -1] > 0

                    if drawing_mode == "freedraw":
                        mask = binary_fill_holes(mask)

                if mask.sum() > 0:
                    mask_created = True
                    if auto_generated:
                        # Upscale the mask to display
                        mask_todisp = Image.fromarray(mask[:, :, -1]).resize((size_up, size_up),
                                                                             resample=Image.Resampling.LANCZOS)
                        # Covert to grayscale
                        mask_todisp = mask_todisp.convert("L")
                        # Blurring to smooth the edges
                        mask_todisp = mask_todisp.filter(ImageFilter.GaussianBlur(radius=5))

                        # Thresholding to make binary
                        threshold_value = 175
                        mask_todisp = mask_todisp.point(lambda p: 255 if p > threshold_value else 0)

                        # Normalizing the original mask
                        mask = np.array(mask_todisp) / 255

                    else:
                        # If not auto-generated simply display the original mask
                        mask_todisp = Image.fromarray(mask)

                    # Display the mask
                    st.image(mask_todisp)

# If the show_res button is clicked
if st.session_state.get('show_res'):
    # Check if the mask was actually created
    if mask_created:
        mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
        img_sitk = sitk.GetImageFromArray(np.array(img).astype(np.float32))

        # Extract radiomic features
        feat = featurexImg(img_sitk[0, :, :], mask_sitk,
                           ["firstorder", "shape2D", "glcm", "gldm", "glrlm", "glszm",
                            "ngtdm"])
        feat = feat.drop(["original_glcm_SumAverage"], axis=1)

        # Load normalization info
        norm_df = pd.read_csv("pages/normalization_all.csv")
        mean = norm_df["mean"]
        std = norm_df["std"]
        mean.index = feat.columns.tolist()
        std.index = feat.columns.tolist()
        feat.loc["mean"] = mean
        feat.loc["std"] = std

        norm_feat = (feat.iloc[0] - feat.loc['mean']) / feat.loc['std']

        # Load the classification model
        with open('pages/svm_final.joblib', 'rb') as f:
            svm_classifier = joblib.load(f)

        # Compute vector of probabilities for prediction
        probs = svm_classifier.predict_proba(np.array(norm_feat).reshape(1, -1))[0]
        pred = np.argmax(probs)  # predicted class

        if pred == 0:   # benign
            result_container.markdown(f"""<p style="background-color: rgba(33, 195, 84, 0.6);
                                            color:#000;
                                            text-align:center;
                                            font-size:20px;
                                            border-top-left-radius:0.5rem;
                                            border-top-right-radius:0.5rem;
                                            border-bottom-right-radius:0.5rem;
                                            border-bottom-left-radius:0.5rem;
                                            padding:2% 0;">
                The selected region is <b>benign</b> with a probability of <b>{probs[0] * 100:.2f}%</b>.</p>""",
                                      unsafe_allow_html=True)

        elif pred == 1:    # malign
            result_container.markdown(f"""<p style="background-color: rgba(255, 43, 43, 0.3);
                                            color:#000;
                                            text-align:center;
                                            font-size:20px;
                                            border-top-left-radius:0.5rem;
                                            border-top-right-radius:0.5rem;
                                            border-bottom-right-radius:0.5rem;
                                            border-bottom-left-radius:0.5rem;
                                            padding:2% 0;">
                            The selected region is <b>malignant</b> with a probability of <b>{probs[1] * 100:.2f}%</b>.</p>""",
                                      unsafe_allow_html=True)

    # If the mask wasn't created yet, print an error message
    else:
        result_container.markdown(f"""<p style="background-color: rgba(255, 227, 18, 0.4);
                                                    color:#000;
                                                    text-align:center;
                                                    font-size:20px;
                                                    border-top-left-radius:0.5rem;
                                                    border-top-right-radius:0.5rem;
                                                    border-bottom-right-radius:0.5rem;
                                                    border-bottom-left-radius:0.5rem;
                                                    padding:2% 0;">
                                    Please select a region first.</p>""",
                                  unsafe_allow_html=True)
