# 🟢 Always the first Streamlit call
import streamlit as st
st.set_page_config(page_title="Kids' AI Storybook Generator", layout="centered")
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
# 🚀 Imports
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from fpdf import FPDF
from pathlib import Path


import requests
import base64
from dotenv import load_dotenv


# 🔐 Load HuggingFace Token
load_dotenv() 

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# 🔧 Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Device set to use: **{device.upper()}**")

# 🎨 Load models
@st.cache_resource
def load_models():
    # Story generator
    
    story_gen = pipeline("text-generation", model="gpt2", device=0 if device == "cuda" else -1)


    # Image generator
    
    image_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",token=HF_TOKEN)

    image_pipe.to(device)

    return story_gen, image_pipe

story_gen, image_pipe = load_models()

# 📘 UI
st.markdown("""
    <style>
    body {
        background-image: url('https://www.transparenttextures.com/patterns/cartographer.png');
        background-size: cover;
        background-repeat: repeat;
        font-family: Comic Sans MS, cursive, sans-serif;
    }
    .stApp {
        background-color: #FFFAF0;
    }
    h1 {
        color: #FF69B4;
        text-align: center;
    }
    .css-1cpxqw2 {
        color: #2E8B57;
    }
    .stTextInput > div > div > input {
        background-color: #FFFACD;
        border: 2px solid #FFD700;
        border-radius: 10px;
        padding: 10px;
        font-size: 18px;
    }
    .stButton > button {
        background-color: #FFB6C1;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


st.title("🌈✨ Kids’ Magical Storybook Generator ✨🌈")

prompt = st.text_input("🧠💡 Enter a magical idea for your story (e.g., 'a panda who learns to fly'):")

# st.image("https://i.imgur.com/3ZQ3Z5L.png", use_container_width=True)

def truncate_prompt(text, max_tokens=75):
    return " ".join(text.split()[:max_tokens])

if st.button("Generate Storybook"):
    if not prompt.strip():
        st.warning("Please enter a prompt to continue.")
    else:
        with st.spinner("Generating story and images..."):
            # Generate story
            
            story_output = story_gen(prompt, max_length=200, do_sample=True, truncation=True)[0]["generated_text"]



            # Split into paragraphs and generate images
            paragraphs = story_output.split(". ")
            images = []
            for para in paragraphs[:5]:  # Limit to 5 pages
                try:
                    image = image_pipe(truncate_prompt(para)).images[0]

                except Exception as e:
                    st.error(f"Image generation failed for: {para}\nError: {e}")
                    continue
                images.append((para, image))

            # Save storybook as PDF
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            for i, (text, img) in enumerate(images):
                img_path = f"page_{i}.png"
                img.save(img_path)

                pdf.add_page()
                pdf.image(img_path, x=10, y=10, w=180, h=100)
                pdf.ln(95)
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, text)

                os.remove(img_path)  # Clean up temp image

            pdf_path = "storybook.pdf"
            pdf.output(pdf_path)

        # 📤 Download
        with open(pdf_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            st.markdown(f"### ✅ Storybook ready!\n[Download PDF](data:application/pdf;base64,{b64})", unsafe_allow_html=True)
