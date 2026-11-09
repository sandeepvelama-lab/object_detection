
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
from collections import Counter

# --------------------------------
# Page configuration
# --------------------------------
st.set_page_config(
    page_title="My Object Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("My - Object Detection")
st.write("Upload an image to identify objects using my model.")

# --------------------------------
# Load my model
# --------------------------------
@st.cache_resource
def load_model():
    return YOLO("sandeep_obj.pt")

try:
    model = load_model()
    st.success("My model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --------------------------------
# Sidebar settings
# --------------------------------
st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05
)

# --------------------------------
# Image upload
# --------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # --------------------------------
    # Object detection
    # --------------------------------
    if st.button("Detect Objects", type="primary"):

        with st.spinner("Detecting objects..."):

            results = model.predict(
                source=image,
                conf=confidence,
                verbose=False
            )

            result = results[0]

            # YOLO returns annotated image in BGR format
            annotated_image = result.plot()

            # Convert BGR to RGB for Streamlit
            annotated_image = annotated_image[:, :, ::-1]

            # --------------------------------
            # Display detected image
            # --------------------------------
            with col2:
                st.subheader("Detection Results")
                st.image(
                    annotated_image,
                    use_container_width=True
                )

            # --------------------------------
            # Extract detection information
            # --------------------------------
            detections = []

            for box in result.boxes:

                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                conf_score = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "Object": class_name,
                    "Confidence": round(conf_score, 4),
                    "X1": round(x1, 2),
                    "Y1": round(y1, 2),
                    "X2": round(x2, 2),
                    "Y2": round(y2, 2)
                })

            # --------------------------------
            # Display detection summary
            # --------------------------------
            st.divider()
            st.subheader("Detection Summary")

            if detections:

                df = pd.DataFrame(detections)

                total_objects = len(df)
                unique_objects = df["Object"].nunique()

                c1, c2, c3 = st.columns(3)

                c1.metric("Total Objects", total_objects)
                c2.metric("Unique Classes", unique_objects)
                c3.metric(
                    "Average Confidence",
                    f"{df['Confidence'].mean():.2%}"
                )

                st.subheader("Detected Objects")
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                # Object counts
                st.subheader("Object Counts")

                counts = Counter(df["Object"])

                count_df = pd.DataFrame(
                    counts.items(),
                    columns=["Object", "Count"]
                )

                st.bar_chart(
                    count_df.set_index("Object")
                )

            else:
                st.warning("No objects detected in this image.")
