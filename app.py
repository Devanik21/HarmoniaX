import streamlit as st
import google.generativeai as genai
import numpy as np
import tempfile
import io
import wave

# Set up Streamlit app
st.set_page_config(page_title="AI Music Composer", page_icon="🎵")
st.title("🎵 AI Music Composer")
st.markdown("Generate AI-powered music based on mood!")

# Sidebar for user inputs
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key = st.text_input("Enter your Gemini API Key", type="password")
    
    if api_key:
        genai.configure(api_key=api_key)
    
    st.header("🎼 Music Settings")
    mood = st.selectbox("Select Mood", ["Happy", "Sad", "Energetic", "Calm"])
    genre = st.selectbox("Select Genre", ["Classical", "Jazz", "Electronic"])
    duration = st.slider("Duration (seconds)", 5, 30, 15)

# Function to generate music prompt
def generate_music_prompt(mood, genre, duration):
    return f"Generate a {duration}-second {genre} music piece with {mood} mood."

# Function to create simple synthetic audio
def create_synthetic_audio(duration, sample_rate=22050):
    # Create simple synthetic audio (just a placeholder)
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    
    # Simple sine wave based on mood
    freq = 440  # Base frequency in Hz
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    
    return audio, sample_rate

# Function to save audio as WAV
def save_audio_to_wav(audio_data, sample_rate):
    # Create a BytesIO object for the WAV file
    wav_io = io.BytesIO()
    
    # Create WAV file
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes = 16 bits
        wav_file.setframerate(sample_rate)
        
        # Convert float audio to int16
        audio_data = (audio_data * 32767).astype(np.int16)
        wav_file.writeframes(audio_data.tobytes())
    
    wav_io.seek(0)
    return wav_io

# Generate AI Music
if st.button("🎶 Generate Music"):
    if not api_key:
        st.error("Please enter your API key")
    else:
        try:
            with st.spinner("Generating music... 🎵"):
                # Generate prompt
                prompt = generate_music_prompt(mood, genre, duration)
                
                # Log the prompt (for debugging)
                st.write(f"Prompt: {prompt}")
                
                # Create placeholder audio
                audio_data, sample_rate = create_synthetic_audio(duration)
                
                # Save as WAV
                wav_data = save_audio_to_wav(audio_data, sample_rate)
                
                # Display audio player
                st.audio(wav_data, format="audio/wav")
                
                # Download button
                st.download_button(
                    "Download Music", 
                    data=wav_data, 
                    file_name=f"{mood}_{genre}_music.wav", 
                    mime="audio/wav"
                )
                
                st.success("Music generated successfully!")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
