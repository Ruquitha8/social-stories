from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
import re
from dotenv import load_dotenv
from gtts import gTTS
from langchain_groq import ChatGroq

# =============================
# Load Environment Variables
# =============================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("hf_token")

HEADERS_HF = {"Authorization": f"Bearer {HF_TOKEN}"}

# =============================
# FastAPI App
# =============================
app = FastAPI(title="Social Story Creator API")

# =============================
# LLM Setup
# =============================
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)


class StoryRequest(BaseModel):
    child_name: str
    child_age: int
    scenario: str
    traits: str
    num_scenes: int
    image_style: str
    language: str



def generate_story(prompt: str):
    response = llm.invoke(prompt)
    return response.content



def generate_image(scene_text, style, child_name, child_age, scenario, index):

    style_map = {
        "Cartoon": "bright colorful cartoon, Pixar style",
        "Animation": "2D animation style",
        "3D Style": "3D Pixar style",
        "Simple Drawing": "storybook illustration",
        "Realistic": "ultra realistic photo"
    }

    prompt = (
        f"{style_map[style]}. "
        f"Scene about {scenario}. "
        f"A {child_age}-year-old child named {child_name}. "
        f"{scene_text}. emotionally gentle."
    )

    HF_API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"

    payload = {"inputs": prompt}

    r = requests.post(HF_API_URL, headers=HEADERS_HF, json=payload)

    if r.status_code == 200:
        filename = f"scene_{index}.png"
        with open(filename, "wb") as f:
            f.write(r.content)
        return filename

    return None



def generate_voice(scene_text, lang, index):

    filename = f"scene_{index}.mp3"

    tts = gTTS(text=scene_text, lang=lang)
    tts.save(filename)

    return filename



@app.post("/generate-story")
def create_story(data: StoryRequest):

    prompt = f"""
    Write a {data.num_scenes}-scene children's story.

    Child: {data.child_name}, age {data.child_age}
    Situation: {data.scenario}
    Traits: {data.traits}

    Each scene should be 2–3 short sentences.
    """

    story = generate_story(prompt)

    scenes = re.split(r"\n+", story)

    return {
        "scenes": scenes[:data.num_scenes]
    }


# =============================
# API: Generate Image + Voice
# =============================
@app.post("/generate-media")
def create_media(data: StoryRequest):

    prompt = f"""
    Write a {data.num_scenes}-scene children's story.

    Child: {data.child_name}, age {data.child_age}
    Situation: {data.scenario}
    Traits: {data.traits}

    Each scene should be 2–3 short sentences.
    """

    story = generate_story(prompt)

    scenes = re.split(r"\n+", story)[:data.num_scenes]

    results = []

    for i, scene in enumerate(scenes):

        image = generate_image(
            scene,
            data.image_style,
            data.child_name,
            data.child_age,
            data.scenario,
            i + 1
        )

        audio = generate_voice(
            scene,
            data.language,
            i + 1
        )

        results.append({
            "scene": scene,
            "image": image,
            "audio": audio
        })

    return {
        "story": results
    }