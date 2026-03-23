import requests
import os
import re
from dotenv import load_dotenv
from gtts import gTTS

from langchain_groq import ChatGroq

# ==============================
# Load environment variables
# ==============================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("hf_token")

HEADERS_HF = {"Authorization": f"Bearer {HF_TOKEN}"}

# ==============================
# LLM Client
# ==============================
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

# ==============================
# Generate Story
# ==============================
def generate_story(prompt: str):
    response = llm.invoke(prompt)
    return response.content


# ==============================
# Image Generation
# ==============================
def generate_image(scene_text, style, child_name, child_age, scenario, index):

    style_map = {
        "Cartoon": "bright colorful cartoon, Pixar style",
        "Animation": "2D animation style, kid friendly",
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

        print(f"🖼 Image saved: {filename}")
    else:
        print("❌ Image generation failed")


# ==============================
# Voice Generation
# ==============================
def generate_voice(scene_text, lang, index):

    filename = f"scene_{index}.mp3"

    tts = gTTS(text=scene_text, lang=lang)
    tts.save(filename)

    print(f"🔊 Voice saved: {filename}")


# ==============================
# Terminal Chatbot
# ==============================
print("\n🧩 Personalized Social Story Creator\n")

child_name = input("Child Name: ")
child_age = int(input("Child Age: "))
scenario = input("Situation (example: traveling in a car): ")
traits = input("Child behavior traits: ")

num_scenes = int(input("Number of scenes (2-6): "))

print("\nChoose Image Style")
print("1 Cartoon")
print("2 Animation")
print("3 3D Style")
print("4 Simple Drawing")
print("5 Realistic")

style_choice = input("Enter number: ")

style_map = {
    "1": "Cartoon",
    "2": "Animation",
    "3": "3D Style",
    "4": "Simple Drawing",
    "5": "Realistic"
}

image_style = style_map.get(style_choice, "Cartoon")

print("\nChoose Voice Language")
print("1 English")
print("2 Hindi")
print("3 Telugu")

lang_choice = input("Enter number: ")

lang_map = {
    "1": "en",
    "2": "hi",
    "3": "te"
}

language = lang_map.get(lang_choice, "en")

# ==============================
# Generate Story
# ==============================
print("\n🤖 Generating Story...\n")

prompt = f"""
Write a {num_scenes}-scene children's story.

Child: {child_name}, age {child_age}
Situation: {scenario}
Traits: {traits}

Each scene should be 2–3 short sentences.
"""

story = generate_story(prompt)

scenes = re.split(r"\n+", story)
scenes = scenes[:num_scenes]

# ==============================
# Approve Scenes
# ==============================
approved_scenes = []

for i, scene in enumerate(scenes):

    print(f"\nScene {i+1}:")
    print(scene)

    edit = input("Edit scene? (y/n): ")

    if edit.lower() == "y":
        scene = input("Enter edited scene: ")

    approved_scenes.append(scene)

# ==============================
# Generate Images + Voice
# ==============================
print("\n🎨 Generating Images and Voice...\n")

for i, scene in enumerate(approved_scenes):

    print(f"\nScene {i+1}")

    generate_image(scene, image_style, child_name, child_age, scenario, i+1)
    generate_voice(scene, language, i+1)

print("\n🎉 Story Generated Successfully!")
print("Check your folder for images and audio.")