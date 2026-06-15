"""
app.py — Streamlit front-end for Face Mask Detection with Grad-CAM.

Run with:
    streamlit run app/app.py
"""

import sys
from pathlib import Path
import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import CLASSES
from src.predict import detect_and_predict, annotate_frame, load_models
from src.gradcam import compute_heatmap, overlay_heatmap, make_gradcam_figure
from src.preprocessing import apply_preprocessing

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="wide",
)

@st.cache_resource(show_spinner="Loading models…")
def get_models():
    return load_models()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.title("😷 Face Mask Detection")
    st.caption(
        "Binary face mask classification using **MobileNetV2** + OpenCV SSD  \n"
        "with **Grad-CAM** explainability · Digital Image Processing"
    )
    st.divider()

    # ── Sidebar ───────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Settings")
    conf_thresh  = st.sidebar.slider("Face detection confidence", 0.3, 0.9, 0.5, 0.05)
    show_gradcam = st.sidebar.toggle("Show Grad-CAM heatmaps", value=True)
    gc_alpha     = st.sidebar.slider(
        "Heatmap opacity", 0.1, 0.9, 0.45, 0.05,
        disabled=not show_gradcam,
    )

    st.sidebar.divider()
    st.sidebar.header("🔧 Preprocessing")
    st.sidebar.caption("Applied to each face crop before classification.")
    pre_gaussian  = st.sidebar.toggle("Gaussian blur",        value=True)
    pre_median    = st.sidebar.toggle("Median filter",        value=False)
    pre_clahe     = st.sidebar.toggle("CLAHE equalization",   value=True)
    pre_contrast  = st.sidebar.toggle("Contrast stretching",  value=False)

    preprocessing_config = {
        "gaussian":         pre_gaussian,
        "median":           pre_median,
        "clahe":            pre_clahe,
        "contrast_stretch": pre_contrast,
    }

    st.sidebar.divider()
    st.sidebar.markdown(
        "**Grad-CAM** highlights the regions the model focused on.  \n"
        "🔴 Red = high activation · 🔵 Blue = low activation"
    )

    # ── Load models ───────────────────────────────────────────────────────
    try:
        classifier, face_net = get_models()
    except Exception as e:
        st.error(
            f"Could not load models: {e}\n\n"
            "Train first: `python -m src.train`"
        )
        return

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_upload, tab_webcam = st.tabs(["📁 Upload Image", "📷 Webcam"])

    # ── Upload tab ────────────────────────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a photo", type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            frame   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            with st.spinner("Detecting…"):
                results = detect_and_predict(
                    frame, classifier, face_net, conf_thresh,
                    preprocessing_config=preprocessing_config,
                )
                annotated     = annotate_frame(frame, results)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # Detection result
            st.subheader("Detection")
            st.image(annotated_rgb, use_column_width=True)

            if not results:
                st.warning("No faces detected. Try a clearer image or lower the confidence threshold.")
                return

            # Summary metrics
            counts = {cls: 0 for cls in CLASSES}
            for *_, label, _ in results:
                counts[label] += 1

            col1, col2 = st.columns(2)
            col1.metric("✅ With Mask",    counts["WithMask"])
            col2.metric("❌ Without Mask", counts["WithoutMask"])

            # ── Preprocessing before/after strip ──────────────────────────
            any_pre = any(preprocessing_config.values())
            if any_pre and results:
                st.divider()
                st.subheader("🔧 Preprocessing Preview")
                st.caption("Face crops: original vs after preprocessing pipeline.")

                for i, (x1, y1, x2, y2, label, conf) in enumerate(results[:3]):
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crop_pre = apply_preprocessing(crop, preprocessing_config)

                    col_before, col_after = st.columns(2)
                    col_before.image(
                        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                        caption=f"Face {i+1} — Original",
                        use_column_width=True,
                    )
                    col_after.image(
                        cv2.cvtColor(crop_pre, cv2.COLOR_BGR2RGB),
                        caption=f"Face {i+1} — After preprocessing",
                        use_column_width=True,
                    )

            # ── Grad-CAM section ──────────────────────────────────────────
            if show_gradcam:
                st.divider()
                st.subheader("🔍 Grad-CAM Explainability")
                st.caption(
                    "Each row shows a detected face and the regions "
                    "the model used to make its prediction. "
                    "A well-trained model should activate over the **nose and mouth** area."
                )

                face_crops, heatmaps, pred_labels = [], [], []

                for (x1, y1, x2, y2, label, conf) in results:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    # Apply preprocessing then prepare for classifier
                    crop_pre     = apply_preprocessing(crop, preprocessing_config)
                    crop_rgb     = cv2.cvtColor(crop_pre, cv2.COLOR_BGR2RGB)
                    crop_resized = cv2.resize(crop_rgb, (224, 224))
                    img_array    = np.expand_dims(crop_resized / 255.0, axis=0)

                    with st.spinner(f"Computing Grad-CAM for face {len(face_crops)+1}…"):
                        heatmap = compute_heatmap(img_array, classifier)

                    face_crops.append(crop_pre)
                    heatmaps.append(heatmap)
                    pred_labels.append(f"{label} ({conf:.0%})")

                # Display as grid
                for i, (crop, heatmap, label) in enumerate(
                    zip(face_crops, heatmaps, pred_labels)
                ):
                    overlay, heatmap_colored = overlay_heatmap(crop, heatmap, gc_alpha)

                    col_orig, col_heat, col_over = st.columns(3)
                    col_orig.image(
                        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                        caption=f"Face {i+1}: {label}", use_column_width=True,
                    )
                    col_heat.image(
                        cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB),
                        caption="Heatmap", use_column_width=True,
                    )
                    col_over.image(
                        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                        caption="Overlay", use_column_width=True,
                    )

                # Download button for full Grad-CAM figure
                if face_crops:
                    fig = make_gradcam_figure(face_crops, heatmaps, pred_labels)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                    buf.seek(0)
                    st.download_button(
                        "⬇️ Download Grad-CAM figure",
                        data=buf,
                        file_name="gradcam_result.png",
                        mime="image/png",
                    )

    # ── Webcam tab ────────────────────────────────────────────────────────
    with tab_webcam:
        st.info(
            "**Webcam mode** runs in a terminal OpenCV window (not the browser).  \n\n"
            "```bash\npython -m src.predict\n```\n\n"
            "Press **q** in the OpenCV window to quit."
        )
        st.caption(
            "Note: Grad-CAM is not shown in webcam mode due to real-time "
            "performance constraints."
        )


if __name__ == "__main__":
    main()
