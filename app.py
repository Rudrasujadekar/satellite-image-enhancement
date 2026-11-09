
import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="Satellite Image Enhancement",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Satellite Image Enhancement")
st.write("Enhance aerial images using OpenCV.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

technique = st.selectbox(
    "Choose an enhancement technique",
    [
        "Brightness & Contrast",
        "Histogram Equalization",
        "Gaussian Blur",
        "Median Blur",
        "Bilateral Filter",
        "Sharpening",
        "Canny Edge Detection"
    ]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    original = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if original is None:
        st.error("Could not read this image.")
        st.stop()

    if technique == "Brightness & Contrast":
        alpha = st.slider("Contrast", 0.5, 3.0, 1.15)
        beta = st.slider("Brightness", -100, 100, 15)
        result = cv2.convertScaleAbs(
            original, alpha=alpha, beta=beta
        )

    elif technique == "Histogram Equalization":
        gray = cv2.cvtColor(
            original, cv2.COLOR_BGR2GRAY
        )
        result = cv2.equalizeHist(gray)

    elif technique == "Gaussian Blur":
        result = cv2.GaussianBlur(
            original, (5, 5), 0
        )

    elif technique == "Median Blur":
        result = cv2.medianBlur(original, 5)

    elif technique == "Bilateral Filter":
        result = cv2.bilateralFilter(
            original, 9, 75, 75
        )

    elif technique == "Sharpening":
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        result = cv2.filter2D(
            original, -1, kernel
        )

    elif technique == "Canny Edge Detection":
        gray = cv2.cvtColor(
            original, cv2.COLOR_BGR2GRAY
        )
        result = cv2.Canny(gray, 100, 200)

    st.subheader("Before and After")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        if len(result.shape) == 2:
            display_result = result
        else:
            display_result = cv2.cvtColor(
                result, cv2.COLOR_BGR2RGB
            )

        st.image(
            display_result,
            caption=technique,
            use_container_width=True
        )

    success, encoded = cv2.imencode(
        ".png", result
    )

    if success:
        st.download_button(
            "Download Enhanced Image",
            data=encoded.tobytes(),
            file_name="enhanced_image.png",
            mime="image/png"
        )

    st.success("Image processing completed!")

else:
    st.info("Upload an image to get started.")

st.caption(
    "Prototype demonstration using OpenCV. "
    "Aerial photographs are illustrative inputs, "
    "not verified satellite observations."
)
