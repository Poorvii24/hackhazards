import streamlit as st
from PIL import Image
import datetime
import matplotlib.pyplot as plt
import os
import base64
import torch
import json
import requests
import random
from transformers import pipeline
from groq import Groq
from dotenv import load_dotenv
import speech_recognition as sr
import tempfile
import pyttsx3

# Must be first
st.set_page_config(page_title="AI Wellness Companion", layout="centered")

# Load env vars
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")
spoon_key = os.getenv("SPOON_KEY")

st.title("🌿 AI Wellness Companion")

# APIs
client = Groq(api_key=groq_api_key)
classifier = pipeline("image-classification", model="microsoft/resnet-50", token=hf_token)

# Mood Input
mood = st.selectbox("🧠 How are you feeling today?", ["neutral", "happy", "sad", "tired", "stressed"])
with st.expander("🎤 Or speak your mood"):
    audio_file = st.file_uploader("Upload a short voice note", type=["wav", "mp3"])
    if audio_file:
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        with sr.AudioFile(tmp_path) as source:
            audio = recognizer.record(source)
            try:
                mood_voice = recognizer.recognize_google(audio).lower()
                st.success(f"Recognized mood: {mood_voice}")
                if mood_voice in ["happy", "sad", "neutral", "tired", "stressed"]:
                    mood = mood_voice
            except sr.UnknownValueError:
                st.error("Could not understand audio.")

# Image Upload (exclusivity logic)
uploaded_file = st.file_uploader("📷 Upload your meal photo", type=["jpg", "jpeg", "png"])
camera_image = None if uploaded_file else st.camera_input("📸 Or take one")

if uploaded_file or camera_image:
    image = Image.open(uploaded_file or camera_image).convert("RGB")
    st.image(image, caption="Your Meal", use_container_width=True)

    with st.spinner("🔍 Classifying..."):
        predictions = classifier(image)
        top_foods = ', '.join([item['label'] for item in predictions[:2]])
        st.success(f"🍽 Detected: {top_foods}")

    # Spoonacular Nutrition
    def get_nutrition_spoon(food):
        url = f"https://api.spoonacular.com/recipes/guessNutrition?title={food}&apiKey={spoon_key}"
        try:
            r = requests.get(url)
            data = r.json()
            return {
                "Calories": data["calories"]["value"],
                "Protein (g)": data["protein"]["value"],
                "Fat (g)": data["fat"]["value"]
            }
        except:
            return None

    st.subheader("📊 Nutrition Facts")
    for item in top_foods.split(", "):
        st.markdown(f"**{item}**")
        nutrition = get_nutrition_spoon(item)
        if nutrition:
            for key, val in nutrition.items():
                st.write(f"- {key}: {val}")
        else:
            st.warning("Nutrition data not available.")

    # Groq AI Insight
    st.subheader("🧠 AI Insight")
    user_query = (
        f"The meal shows: {top_foods}. Mood: {mood}. "
        f"Give a concise nutritional analysis, wellness tip, and a 5-minute home workout."
    )
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            stream = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": user_query}],
                stream=True
            )
            full_response = ""
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                full_response += token
            st.markdown(full_response)

    # TTS
    def speak_text(text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    if st.button("🎤 Read AI Insight Aloud"):
        speak_text(full_response)

    # Fitness Tracking
    st.subheader("🏃 Track Your Wellness")
    steps = st.slider("Steps walked", 0, 20000, 6000, step=100)
    water = st.slider("Water intake (L)", 0.0, 5.0, 2.0, step=0.1)
    st.write(f"You walked **{steps}** steps and drank **{water}L** water.")

    # Gamification
    points = int(steps / 1000 + water * 10)
    streak = st.session_state.get("streak", 1)
    if "last_entry" in st.session_state:
        last = datetime.datetime.strptime(st.session_state.last_entry, "%Y-%m-%d").date()
        today = datetime.date.today()
        if (today - last).days == 1:
            streak += 1
        elif (today - last).days > 1:
            streak = 1
    st.session_state.streak = streak
    st.session_state.last_entry = str(datetime.date.today())

    st.metric("🏅 Wellness Points", points)
    st.metric("🔥 Daily Streak", streak)
    if streak % 5 == 0:
        st.success(f"🎉 You're on a {streak}-day streak! You earned a wellness badge! 🥇")

    # Save History
    history_data = {
        "time": str(datetime.datetime.now()),
        "mood": mood,
        "food": top_foods,
        "steps": steps,
        "water": water,
        "points": points
    }
    if os.path.exists("history.json"):
        with open("history.json", "r") as f:
            all_data = json.load(f)
    else:
        all_data = []
    all_data.append(history_data)
    with open("history.json", "w") as f:
        json.dump(all_data, f, indent=2)

    # Motivation
    quotes = [
        "Every step is progress. Keep going! 💪",
        "Hydration is self-care. Drink up! 🥤",
        "You don’t have to be perfect. Just be consistent. 🧘",
        "Healthy outside starts from the inside. 🍎"
    ]
    st.markdown(f"> 💬 *{random.choice(quotes)}*")

    # Coaching
    streak_prompt = f"Your streak is {streak}. Give a motivational quote or habit tip."
    streak_response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a friendly wellness coach."},
            {"role": "user", "content": streak_prompt}
        ]
    )
    st.subheader("🧑‍🏫 Streak Coaching")
    st.write(streak_response.choices[0].message.content)

    # Trends
    if os.path.exists("history.json"):
        with open("history.json", "r") as f:
            history = json.load(f)
        if len(history) > 1:
            moods = [entry["mood"] for entry in history]
            times = [entry["time"][-8:] for entry in history]
            scores = [entry["points"] for entry in history]
            mood_scores = {"happy": 5, "neutral": 3, "sad": 1, "tired": 2, "stressed": 2}
            mood_values = [mood_scores.get(m, 3) for m in moods]

            st.subheader("📈 Trends")
            fig, ax = plt.subplots()
            ax.plot(times, mood_values, label="Mood Score", marker='o')
            ax.plot(times, scores, label="Points", marker='x')
            ax.set_ylabel("Score")
            ax.set_xlabel("Time")
            ax.set_title("Mood & Wellness Over Time")
            ax.legend()
            st.pyplot(fig)

            st.download_button("⬇️ Export History as CSV", data="\n".join([
                "Time,Mood,Food,Steps,Water,Points"
            ] + [
                f"{x['time']},{x['mood']},{x['food']},{x['steps']},{x['water']},{x['points']}" for x in history
            ]), file_name="wellness_history.csv", mime="text/csv")

# Footer
st.markdown("---")
st.caption("Built for HackHazards 2025 | 🚀 Powered by Groq, Spoonacular, Hugging Face & Streamlit")
