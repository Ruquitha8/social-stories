import streamlit as st
import requests
import os
from dotenv import load_dotenv
import base64
import re
from gtts import gTTS

# ✅ LangChain Groq (CORRECT WAY)
from langchain_groq import ChatGroq


# ==============================
# Load environment variables
# ==============================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("hf_token")

# ==============================
# LLM Client (ChatGroq)
# ==============================
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

HEADERS_HF = {"Authorization": f"Bearer {HF_TOKEN}"}

# ==============================
# Streamlit setup
# ==============================
st.set_page_config(page_title="🧩 Social Story Creator", layout="wide")
st.title("🧩 Personalized Social Story Creator for Kids")

# ==============================
# Session State
# ==============================
if "scenes" not in st.session_state:
    st.session_state.scenes = []
if "approved" not in st.session_state:
    st.session_state.approved = []
if "final_generated" not in st.session_state:
    st.session_state.final_generated = False

# ==============================
# Sidebar - API key check
# ==============================
def _mask_key(k):
    if not k:
        return "<missing>"
    return k[:4] + "..." + k[-4:]

st.sidebar.header("🔐 API Keys Status")
st.sidebar.write("GROQ_API_KEY:", _mask_key(GROQ_API_KEY))
st.sidebar.write("HF_TOKEN:", _mask_key(HF_TOKEN))

# ==============================
# User Inputs
# ==============================
col1, col2 = st.columns(2)
with col1:
    child_name = st.text_input("Child's Name", "Aarav")
    child_age = st.number_input("Child's Age", 3, 12, 6)

with col2:
    scenario = st.text_input("Describe the daily situation:", "Traveling in a car")
    traits = st.text_area(
        "Child's Behavior Traits",
        "e.g., gets irritated, avoids talking"
    )

num_scenes = st.number_input("Number of scenes", 2, 6, 3)
image_style = st.selectbox(
    "Choose Image Style",
    ["Cartoon", "Animation", "3D Style", "Simple Drawing", "Realistic"]
)

language_choice = st.selectbox(
    "Choose voice language",
    ["en", "hi", "te"],
    format_func=lambda x: {"en": "English", "hi": "Hindi", "te": "Telugu"}[x]
)

# ==============================
# Story Generation (LangChain)
# ==============================
def generate_story(prompt: str) -> str:
    response = llm.invoke(prompt)
    return response.content

# ==============================
# Image Generation (Hugging Face)
# ==============================
def generate_detailed_image(scene_text, style, child_name, child_age, traits, scenario):

    style_map = {
        "Cartoon": "bright colorful cartoon, Pixar style",
        "Animation": "2D animation style, kid friendly",
        "3D Style": "3D Pixar style, cinematic lighting",
        "Simple Drawing": "storybook illustration, soft pastel",
        "Realistic": "ultra realistic photo, DSLR, natural skin"
    }

    prompt = (
        f"{style_map[style]}. "
        f"Scene about {scenario}. "
        f"A {child_age}-year-old child named {child_name}. "
        f"Scene: {scene_text}. "
        f"Emotionally gentle and positive."
    )

    HF_API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
    payload = {"inputs": prompt}

    r = requests.post(HF_API_URL, headers=HEADERS_HF, json=payload)
    return r.content if r.status_code == 200 else None

# ==============================
# Voice (GTTS)
# ==============================
def generate_voice(scene_text, lang, filename):
    tts = gTTS(text=scene_text, lang=lang)
    tts.save(filename)
    return filename

# ==============================
# Step 1: Generate Scenes
# ==============================
if st.button("📝 Generate Editable Scenes"):
    prompt = f"""
    Write a {num_scenes}-scene children's story.

    Child: {child_name}, age {child_age}
    Situation: {scenario}
    Traits: {traits}

    Each scene should be 2–3 short sentences.
    """

    story = generate_story(prompt)
    scenes = re.split(r"\n+", story)
    st.session_state.scenes = scenes[:num_scenes]
    st.session_state.approved = [False] * len(st.session_state.scenes)
    st.session_state.final_generated = False

# ==============================
# Step 2: Approve Scenes
# ==============================
for i, scene in enumerate(st.session_state.scenes):
    st.markdown(f"## Scene {i+1}")
    if not st.session_state.approved[i]:
        edited = st.text_area("Edit", scene, key=f"s{i}")
        if st.button("Approve", key=f"a{i}"):
            st.session_state.scenes[i] = edited
            st.session_state.approved[i] = True
    else:
        st.success(scene)

# ==============================
# Step 3: Images + Voice
# ==============================
if st.session_state.approved and all(st.session_state.approved) and not st.session_state.final_generated:
    for i, scene in enumerate(st.session_state.scenes):
        st.markdown(f"## 🖼 Scene {i+1}")

        img = generate_detailed_image(
            scene, image_style, child_name, child_age, traits, scenario
        )
        if img:
            st.image(img, use_container_width=True)

        audio = generate_voice(scene, language_choice, f"scene_{i+1}.mp3")
        st.audio(audio)

    st.session_state.final_generated = True
    st.success("🎉 Story Generated Successfully!")
