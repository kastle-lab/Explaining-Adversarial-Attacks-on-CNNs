import streamlit as st
import pandas as pd

st.set_page_config(page_title="ConceptLens",
                   page_icon=":bridge_at_night:",
                   layout="wide")

# Create two columns
col1, col2 = st.columns([3,1.5])

# First vertical container in the first column
with col1:
    st.image('example1.JPG', use_column_width = True)
    # ### video
    # video_file = open('video.mp4', 'rb')
    # video_bytes = video_file.read()
    # #width = st.sidebar.slider(
    # #label="Width", min_value=0, max_value=100, value=100, format="%d%%"
    # #)
    # width = max(100, 0.01)
    # side = max((100 - width) / 2, 0.01)
    # _, container, _ = st.columns([side, width, side])
    # container.video(data=video_bytes,loop=True, autoplay=True)

    with st.sidebar:
        st.header("Link to related resources")
        #st.link_button("ECII", url = 'https://arxiv.org/abs/1812.03243')
        st.link_button("Error-margin Analysis", url = 'https://arxiv.org/abs/2405.09580')
        st.link_button("Hidden Neuron Activation Analysis", url = 'http://arxiv.org/abs/2404.13567')
        st.link_button("Github repo", url = 'https://github.com/abhilekha-dalal/ConceptLens')

        
    # button to the "Upload Images" page
    #st.link_button("Explore ConceptLens now!", url = '/Upload_Images')
    
# Second vertical container in the second column
with col2:
    st.markdown("# :rainbow[ConceptLens]")
    st.subheader("_from Pixels to Understanding_")

#####################
st.header("Summary")
st.write("Welcome to _ConceptLens_, where we illuminate the mysteries of artificial neural networks with our innovative approach to visualizing hidden neuron activations! At _ConceptLens_, we've combined cutting-edge deep learning models with symbolic methods to shed light on the decision-making processes of neural networks. With our error-margin analysis, you'll unravel the secrets of neuron activations, understanding what triggers them and how they respond to different stimuli.")
st.write("Now, what exactly is the _Error-rate Percentages_? It's a measure of how confident the system is in labeling neurons that get activated by the stimulus in a given image. Lower percentages mean higher confidence, while higher percentages indicate more uncertainty. In simpler terms, it tells you how accurate the network's predictions are.")
st.write("Our user-friendly interface lets you upload images and inspect 50-class predictions, probability scores, and Grad-CAM saliency maps.")
st.write("ConceptLens now uses a ResNet50V2 scene classifier trained for 50 ADE20K scene categories. The Explore page preserves the original workflow while using the new model for prediction and Grad-CAM.")
st.markdown("For a deep dive into the science behind our platform, our [*Research Paper*](https://arxiv.org/abs/2405.09580) - \"Error-margin Analysis for Hidden Neuron Activation Labels\" awaits your curiosity. We envision a world where researchers like you leverage ConceptLens to unlock groundbreaking insights into images.")
st.write("So, step into the lens of ConceptLens and embark on a journey to reshape the future of neural network interpretability! Let's uncover groundbreaking insights together.")
#Dive into our world and discover how we decode the intricate workings of neural networks. 
# st.write("Dive into the vibrant world of ConceptLens, where we turn the complex labyrinths of artificial neural networks into a visual spectacle of understanding! At ConceptLens, we pride ourselves on our pioneering approach to interpreting hidden neuron activations. Itâ€™s like having a conversation with the AI itself, as you uncover what activates neurons and how they react to various stimuliâ€”this is our error-margin analysis in action, a feature youâ€™ll find only with us.")
# st.write("Our intuitive interface invites you to effortlessly upload images and witness the magic unfold in real time. The results? A detailed, colorful visualization that not only educates but also captivates. Itâ€™s a gateway to understanding neuron functionality like never before, pushing the boundaries of whatâ€™s possible in deep learning.")
# st.markdown("For a deep dive into the science behind our platform, our [*Research Paper*](https://arxiv.org/abs/2405.09580) - â€œError-margin Analysis for Hidden Neuron Activation Labelsâ€ awaits your curiosity. We envision a world where researchers like you leverage ConceptLens to unlock groundbreaking insights into images.")
# st.write("So why wait? Step into the lens of ConceptLens and see the unseen. Letâ€™s reshape the future of neural network interpretability together!")
#####################
st.header("Usage")
st.write("To the left, is the main menu for navigating to different pages in the *ConceptLens*:")
st.markdown("- **Home Page:** We are here!")
st.markdown("- **Explore ConceptLens:** Select an image from the gallery or upload your own. ConceptLens reports the 50-class prediction score and Grad-CAM saliency map.")
#st.markdown("- **Explore ConceptLens now:** This takes you directly to the Upload Images page.")
st.markdown("- **Related Resources:** Interested in learning more about the research behind *ConceptLens*? Visit this section to find links to the relevant paper." )
#####################
