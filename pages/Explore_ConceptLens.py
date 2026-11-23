import os
import uuid
from io import BytesIO

import altair as alt
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from PIL import Image
from streamlit_image_select import image_select

from conceptlens_model import activation_model_for, load_conceptlens_model, preprocess_image_path
from demonstrator import (
    ActivationAnalysisError,
    GradCAMError,
    figure_to_png_bytes,
    generate_gradcam,
    render_gradcam_visualization,
    scene_rate_rows_for,
)


ANALYSIS_STATE_KEY = "conceptlens_analysis_results"
SCENE_RATES_TITLE = "Scene Associations and Non-target Activation Rates"
SCENE_RATES_AXIS_TITLE = "Held-out non-target activation (%)"
SCENE_RATES_BAR_COLOR = "#336BB3"


def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def create_unique_folder(base_path):
    unique_folder_path = os.path.join(base_path, str(uuid.uuid4()))
    os.makedirs(unique_folder_path)
    return unique_folder_path


def image_to_png_bytes(image):
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def build_analysis_results(unique_folder_path, class_folder_path):
    """Run model work once and retain display/download artifacts for reruns."""
    results = []
    extractor = None
    for image_name in sorted(os.listdir(class_folder_path)):
        image_path = os.path.join(class_folder_path, image_name)
        if not os.path.isfile(image_path):
            continue

        result = {
            "image_path": image_path,
            "image_name": image_name,
            "gradcam_error": None,
            "scene_rates": [],
            "scene_rate_error": None,
        }
        try:
            gradcam_result = generate_gradcam(image_path)
            with Image.open(image_path) as source_image:
                gradcam_report = render_gradcam_visualization(
                    source_image, image_name, gradcam_result
                )
                gradcam_report_bytes = figure_to_png_bytes(gradcam_report)
            gradcam_report.clear()
            result.update(
                {
                    "predicted_class": gradcam_result["predicted_class"],
                    "predicted_display_name": gradcam_result["predicted_display_name"],
                    "prediction_score": gradcam_result["prediction_score"],
                    "gradcam_overlay_bytes": image_to_png_bytes(gradcam_result["overlay"]),
                    "gradcam_visualization_bytes": gradcam_report_bytes,
                }
            )
        except GradCAMError as exc:
            result["gradcam_error"] = str(exc)
        else:
            try:
                if extractor is None:
                    extractor = activation_model_for(load_conceptlens_model())
                vector = extractor(preprocess_image_path(image_path), training=False).numpy()[0]
                result["scene_rates"] = scene_rate_rows_for(vector)
            except (ActivationAnalysisError, OSError) as exc:
                result["scene_rate_error"] = str(exc)
        results.append(result)
    return results


def scene_rates_to_png(rows, image_name):
    figure = Figure(figsize=(10, 1.6 + 0.25 * len(rows)), layout="constrained")
    axis = figure.subplots()
    axis.barh(rows["association"], rows["non_target_pct"], color=SCENE_RATES_BAR_COLOR)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xticks(range(0, 101, 10))
    axis.set_xlabel(SCENE_RATES_AXIS_TITLE)
    axis.set_title(image_name, loc="left")
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#e6e9ef")
    figure.suptitle(SCENE_RATES_TITLE, fontweight="bold")
    return figure_to_png_bytes(figure)


def render_scene_rates(results):
    st.markdown(f"### {SCENE_RATES_TITLE}")
    st.caption(
        "The bars show how often the associated neuron exceeded the shown cutoff on "
        "held-out images labeled as other scenes."
    )
    for result in results:
        st.markdown(f"**{result['image_name']}**")
        if result["scene_rate_error"] or result["gradcam_error"]:
            st.info("Scene activation rates are unavailable for this image.")
            continue
        if not result["scene_rates"]:
            st.info("No mapped neuron exceeded a reference cutoff for this image.")
            continue

        rows = pd.DataFrame(result["scene_rates"])
        rows["association"] = [
            f"{row.scene} · {row.non_target_pct:.1f}% ({row.non_target_activated}/{row.non_target_total})"
            f" · N{row.neuron_id} · >{row.cutoff_pct}%"
            for row in rows.itertuples()
        ]
        chart = alt.Chart(rows).mark_bar(color=SCENE_RATES_BAR_COLOR).encode(
            x=alt.X("non_target_pct:Q", title=SCENE_RATES_AXIS_TITLE,
                    scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("association:N", title=None, sort=rows["association"].tolist(),
                    axis=alt.Axis(labelLimit=440)),
            tooltip=[
                alt.Tooltip("scene:N", title="Associated scene"),
                alt.Tooltip("neuron_id:Q", title="Neuron"),
                alt.Tooltip("cutoff_pct:Q", title="Cutoff (% of discovery max)"),
                alt.Tooltip("non_target_pct:Q", title="Non-target activation (%)", format=".1f"),
                alt.Tooltip("non_target_activated:Q", title="Other-scene images activated"),
                alt.Tooltip("non_target_total:Q", title="Other-scene images held out"),
            ],
        ).properties(height=alt.Step(25))
        st.altair_chart(chart, use_container_width=True)
        st.download_button(
            "Download Scene Association Chart",
            data=scene_rates_to_png(rows, result["image_name"]),
            file_name=f"{os.path.splitext(result['image_name'])[0]}_scene_associations.png",
            mime="image/png",
            key=f"scene-rates-download-{result['image_name']}",
        )


def render_saved_results(results):
    render_scene_rates(results)

    st.markdown("### Grad-CAM Saliency Map")
    st.caption(
        "ConceptLens uses Grad-CAM to highlight regions that influenced the "
        "predicted scene-class score."
    )
    for result in results:
        if result["gradcam_error"]:
            st.error(
                f"Could not generate prediction/Grad-CAM for {result['image_name']}: "
                f"{result['gradcam_error']}"
            )
            continue
        st.markdown(f"**{result['image_name']}**")
        st.write(
            f"Predicted Class: {result['predicted_display_name']}  |  "
            f"Prediction Score: {result['prediction_score']:.4f}"
        )
        original_col, overlay_col = st.columns(2)
        with original_col:
            st.image(result["image_path"], caption="Original Image", use_column_width=True)
        with overlay_col:
            st.image(
                result["gradcam_overlay_bytes"],
                caption="Grad-CAM Overlay",
                use_column_width=True,
            )
        st.download_button(
            "Download Grad-CAM Visualization",
            data=result["gradcam_visualization_bytes"],
            file_name=f"{os.path.splitext(result['image_name'])[0]}_gradcam.png",
            mime="image/png",
            key=f"gradcam-download-{result['image_name']}",
        )


st.header("Upload Your Images")
st.markdown(
    "You have the option to select images from our curated gallery or upload your "
    "own unique photos to personalize your experience."
)
st.markdown("Scroll down for results. It may take a few seconds; thank you for your patience!")

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
    uploaded_files = st.file_uploader(
        "Upload file", accept_multiple_files=True, type=["jpg", "jpeg"]
    )
    if uploaded_files and len(uploaded_files) > 3:
        uploaded_files = uploaded_files[:3]
        st.warning("Maximum 3 files allowed.")
    submitted_upload = st.form_submit_button("Process Uploaded Image")

if submitted_upload or process_gallery_image:
    if not uploaded_files and not selected_gallery_image:
        st.warning("Please upload at least one image file.")
    else:
        with st.spinner("This might take a couple seconds, please wait..."):
            unique_folder_path = create_unique_folder("uploads/")
            class_folder_path = os.path.join(unique_folder_path, "uploaded")
            os.makedirs(class_folder_path, exist_ok=True)

            if selected_gallery_image and process_gallery_image:
                image_name = os.path.basename(selected_gallery_image)
                with open(selected_gallery_image, "rb") as source, open(
                    os.path.join(class_folder_path, image_name), "wb"
                ) as destination:
                    destination.write(source.read())

            for uploaded_file in uploaded_files or []:
                if allowed_file(uploaded_file.name, {"jpg", "jpeg", "png"}):
                    with open(os.path.join(class_folder_path, uploaded_file.name), "wb") as destination:
                        destination.write(uploaded_file.getbuffer())

            st.session_state[ANALYSIS_STATE_KEY] = build_analysis_results(
                unique_folder_path, class_folder_path
            )

if st.session_state.get(ANALYSIS_STATE_KEY):
    render_saved_results(st.session_state[ANALYSIS_STATE_KEY])
