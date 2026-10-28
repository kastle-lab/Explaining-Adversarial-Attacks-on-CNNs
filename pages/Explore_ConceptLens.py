import os
import time
import uuid
from io import BytesIO

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from streamlit_image_select import image_select

from demonstrator import (
    GradCAMError,
    figure_to_png_bytes,
    generate_gradcam,
    image_processing_function,
    plot_concept_bars,
    render_concept_visualization,
    render_gradcam_visualization,
)


ANALYSIS_STATE_KEY = "conceptlens_analysis_results"


def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def create_unique_folder(base_path):
    unique_folder_path = os.path.join(base_path, str(uuid.uuid4()))
    os.makedirs(unique_folder_path)
    return unique_folder_path


def image_to_png_bytes(image):
    """Keep a Grad-CAM overlay available after a Streamlit rerun."""
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def build_analysis_results(unique_folder_path, class_folder_path):
    """Run inference once and retain all display/download artifacts in memory."""
    concepts_frame = image_processing_function(unique_folder_path)
    results = []

    for image_name in sorted(os.listdir(class_folder_path)):
        image_path = os.path.join(class_folder_path, image_name)
        if not os.path.isfile(image_path):
            continue
        concept_results = concepts_frame[concepts_frame["Image Name"] == image_name].copy()
        with Image.open(image_path) as source_image:
            concept_report = render_concept_visualization(source_image, image_name, concept_results)
            concept_report_bytes = figure_to_png_bytes(concept_report)
        concept_report.clear()

        result = {
            "image_path": image_path,
            "image_name": image_name,
            "concept_results": concept_results,
            "concept_visualization_bytes": concept_report_bytes,
            "gradcam_error": None,
        }
        try:
            gradcam_result = generate_gradcam(image_path)
            with Image.open(image_path) as source_image:
                gradcam_report = render_gradcam_visualization(source_image, image_name, gradcam_result)
                gradcam_report_bytes = figure_to_png_bytes(gradcam_report)
            gradcam_report.clear()
            result.update({
                "predicted_class": gradcam_result["predicted_class"],
                "prediction_score": gradcam_result["prediction_score"],
                "gradcam_overlay_bytes": image_to_png_bytes(gradcam_result["overlay"]),
                "gradcam_visualization_bytes": gradcam_report_bytes,
            })
        except GradCAMError as exc:
            result["gradcam_error"] = str(exc)
        results.append(result)
    return results


def render_saved_results(results):
    """Render session-state artifacts only; no model work occurs on reruns."""
    st.markdown("### Detected Concepts and Error-rate Percentages")
    for result in results:
        concept_results = result["concept_results"]
        if concept_results.empty:
            st.info(f"No concepts were detected for {result['image_name']}.")
        else:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(result["image_path"], caption=result["image_name"], use_column_width=True)
            with col2:
                figure, axis = plt.subplots(figsize=(7, 5))
                plot_concept_bars(axis, concept_results)
                st.pyplot(figure)
                plt.close(figure)

        st.download_button(
            "Download Concept Visualization",
            data=result["concept_visualization_bytes"],
            file_name=f"{os.path.splitext(result['image_name'])[0]}_concepts.png",
            mime="image/png",
            key=f"concept-download-{result['image_name']}",
        )

    st.markdown("### Grad-CAM Saliency Map")
    st.caption(
        "ConceptLens describes the 64-neuron concept representation. "
        "Grad-CAM uses the same scene classifier to highlight regions that "
        "influenced its predicted class score."
    )
    for result in results:
        if result["gradcam_error"]:
            st.error(f"Could not generate Grad-CAM for {result['image_name']}: {result['gradcam_error']}")
            continue
        st.markdown(f"**{result['image_name']}**")
        st.write(
            f"Predicted Class: {result['predicted_class']}  |  "
            f"Prediction Score: {result['prediction_score']:.4f}"
        )
        original_col, overlay_col = st.columns(2)
        with original_col:
            st.image(result["image_path"], caption="Original Image", use_column_width=True)
        with overlay_col:
            st.image(result["gradcam_overlay_bytes"], caption="Grad-CAM Overlay", use_column_width=True)
        st.download_button(
            "Download Grad-CAM Visualization",
            data=result["gradcam_visualization_bytes"],
            file_name=f"{os.path.splitext(result['image_name'])[0]}_gradcam.png",
            mime="image/png",
            key=f"gradcam-download-{result['image_name']}",
        )


st.header("Upload Your Images")
st.markdown("You have the option to select images from our curated gallery or upload your own unique photos to personalize your experience.")
st.markdown("Scroll down for results ⏬ It may take a few seconds, so thank you for your patience! 😊")

gallery_placeholder = st.empty()
with gallery_placeholder.container():
    selected_gallery_image = image_select(
        label="Choose an image from the options below",
        images=[
            "examples/bedroom.jpeg",
            "examples/highway.jpg",
            "examples/living_space.jpg",
            "examples/road.jpeg",
            "examples/vehicle.jpeg",
        ],
        use_container_width=True,
    )
process_gallery_image = st.button("Process Selected Image")

st.markdown("Or upload your own images:")
with st.form("my-form", clear_on_submit=True):
    uploaded_files = st.file_uploader("Upload file", accept_multiple_files=True, type=["jpg", "jpeg"])
    if uploaded_files and len(uploaded_files) > 3:
        uploaded_files = uploaded_files[:3]
        st.warning("Maximum 3 files allowed.")
    submitted_upload = st.form_submit_button("Process Uploaded Image")

if submitted_upload or process_gallery_image:
    if not uploaded_files and not selected_gallery_image:
        st.warning("Please upload at least one image file.")
    else:
        with st.spinner("This might take a couple seconds, please wait..."):
            time.sleep(3)
            unique_folder_path = create_unique_folder("uploads/")
            class_folder_path = os.path.join(unique_folder_path, "building")
            os.makedirs(class_folder_path, exist_ok=True)

            if selected_gallery_image and process_gallery_image:
                image_name = os.path.basename(selected_gallery_image)
                with open(selected_gallery_image, "rb") as source, open(
                    os.path.join(class_folder_path, image_name), "wb"
                ) as destination:
                    destination.write(source.read())

            for uploaded_file in uploaded_files or []:
                if allowed_file(uploaded_file.name, {"jpg", "jpeg"}):
                    with open(os.path.join(class_folder_path, uploaded_file.name), "wb") as destination:
                        destination.write(uploaded_file.getbuffer())

            # This is the only point that replaces a completed analysis: a
            # user explicitly began another processing workflow. Download
            # reruns never enter this branch.
            st.session_state[ANALYSIS_STATE_KEY] = build_analysis_results(
                unique_folder_path, class_folder_path
            )

if st.session_state.get(ANALYSIS_STATE_KEY):
    render_saved_results(st.session_state[ANALYSIS_STATE_KEY])
