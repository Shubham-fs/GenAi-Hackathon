import sys
import os

# Add the parent directory to Python path to allow imports from backend and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from PIL import Image
import numpy as np

from backend.segmentation_model import DiagramSegmentationModel
from backend.region_extraction import extract_detected_regions, format_regions_for_prompt
from backend.genai_engine import GenAIEngine
from utils.visualization import overlay_mask, draw_bounding_boxes, get_bounding_boxes
from utils.image_processing import resize_image

# Page config
st.set_page_config(page_title="AI Visual Tutor", page_icon="🧠", layout="wide")

# Sidebar
with st.sidebar:
    st.title("🧠 AI Visual Tutor")
    st.write("Upload a diagram to see semantic segmentation in action and generate study materials using Generative AI.")
    
    st.header("Settings")
    api_key_input = st.text_input("Groq API Key (optional if in .env)", type="password")
    
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input

# Caching the models so they aren't reloaded on every run
@st.cache_resource
def load_models():
    seg_model = DiagramSegmentationModel()
    genai_model = GenAIEngine()
    return seg_model, genai_model

st.title("Smart Study Material Generator 📚")

try:
    seg_model, genai_model = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload an Educational Diagram (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 1. Image Upload
    image = Image.open(uploaded_file).convert("RGB")
    image = resize_image(image, max_size=800)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Diagram")
        st.image(image, use_container_width=True)
        
    # 2. Semantic Segmentation 
    with st.spinner("Running Semantic Segmentation... 🤔"):
        mask, id2label = seg_model.predict(image)
        
        # 3. Region Extraction
        detected_objects = extract_detected_regions(mask, id2label)
        regions_context = format_regions_for_prompt(detected_objects)
        
        overlayed_image = overlay_mask(image, mask, alpha=0.4)
        
        with col2:
            st.subheader("Semantic Representation")
            st.image(overlayed_image, use_container_width=True)
            
            with st.expander("View Detected Regions"):
                st.write(detected_objects if detected_objects else "No specific standard objects detected in this general view. AI will rely on visual inspection.")

    st.divider()
    
    st.header("Interactive Study Materials 📝")
    st.write("Generate different formats based on the AI's understanding of the diagram.")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Explanation", "Flashcards", "Practice Quiz", "Summary Poster", "Ask AI (Chat)"])
    
    with tab1:
        st.subheader("Diagram Explanation")
        complexity = st.radio("Explanation Complexity:", ["Explain Simply (Like I'm 10)", "Explain Normally", "Explain Technically"], index=1)
        
        if st.button("Generate Explanation"):
            with st.spinner(f"Generating Explanation ({complexity})..."):
                explanation = genai_model.generate_explanation(image, regions_context, complexity)
                st.markdown(explanation)
                
    with tab2:
        st.subheader("Study Flashcards")
        st.write("Test your memory!")
        if st.button("Generate Flashcards"):
            with st.spinner("Creating Flashcards..."):
                flashcards = genai_model.generate_flashcards(image, regions_context)
                st.markdown(flashcards)
                
    with tab3:
        st.subheader("Knowledge Check")
        if st.button("Generate Practice Quiz"):
            with st.spinner("Generating Quiz Questions..."):
                quiz = genai_model.generate_quiz(image, regions_context)
                st.markdown(quiz)
                
    with tab4:
        st.subheader("Visual Poster Content")
        if st.button("Create Summary Poster Text"):
            with st.spinner("Designing textual poster..."):
                poster = genai_model.generate_poster(image, regions_context)
                st.markdown(poster)
                
    with tab5:
        st.subheader("Ask AI Tutor 🤓")
        st.write("Have a specific question about this diagram?")
        user_q = st.text_input("Your Question:")
        if st.button("Ask"):
            if user_q:
                with st.spinner("Thinking..."):
                    answer = genai_model.answer_question(image, regions_context, user_q)
                    st.success(answer)
            else:
                st.warning("Please type a question first.")
