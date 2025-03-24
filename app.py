import streamlit as st
import google.generativeai as genai
from pydub import AudioSegment
import numpy as np
import tempfile
import os

# Set up Streamlit app
st.set_page_config(page_title="AI Music Composer", page_icon="🎵", layout="wide")
st.title("🎵 AI Music Composer")
st.markdown("Generate AI-powered music based on mood!")

# Sidebar for user inputs
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key = st.text_input("Enter your Gemini API Key", type="password")
    if not api_key:
        st.warning("Please enter your Gemini API Key to continue.")
    else:
        genai.configure(api_key=api_key)
    
    st.header("🎼 Music Settings")
    mood = st.selectbox("Select Mood", ["Happy", "Sad", "Energetic", "Calm", "Romantic", "Mysterious"])
    genre = st.selectbox("Select Genre", ["Classical", "Jazz", "Rock", "Electronic", "Ambient", "Lo-Fi"])
    duration = st.slider("Select Duration (seconds)", 5, 60, 20)
    
# Function to generate music prompt
def generate_music_prompt(mood, genre, duration):
    return f"Generate a {duration}-second instrumental music piece that reflects a {mood} mood in {genre} style. The melody should be unique and engaging."

# Generate AI Music
if st.button("🎶 Generate Music"):
    if not api_key:
        st.error("API Key is missing!")
    else:
        prompt = generate_music_prompt(mood, genre, duration)
        
        with st.spinner("Generating music... 🎵"):
            try:
                model = genai.GenerativeModel("gemini-2.0-flash")
                response = model.generate_content(prompt)
                
                # Simulate audio generation (Replace with actual AI model output in next step)
                audio_data = np.random.randn(duration * 44100)  # Placeholder random noise
                
                # Convert to WAV file
                temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                AudioSegment.from_raw(
                    audio_data.tobytes(), sample_width=2, frame_rate=44100, channels=1
                ).export(temp_wav.name, format="wav")
                
                # Display and download audio
                st.audio(temp_wav.name, format="audio/wav")
                st.download_button("Download Music", data=open(temp_wav.name, "rb").read(), file_name="generated_music.wav", mime="audio/wav")
            
            except Exception as e:
                st.error(f"Error generating music: {e}")
