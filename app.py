import streamlit as st
import google.generativeai as genai
import numpy as np
import soundfile as sf
import tempfile
import os
from scipy import signal
import random
import io

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
    mood = st.selectbox("Select Mood", ["Happy", "Sad", "Energetic", "Calm", "Mysterious"])
    genre = st.selectbox("Select Genre", ["Classical", "Jazz", "Electronic", "Lo-Fi"])
    duration = st.slider("Duration (seconds)", 10, 30, 15)
    complexity = st.slider("Complexity", 1, 10, 5)

class MusicGenerator:
    def __init__(self, mood, genre, duration, complexity):
        self.mood = mood
        self.genre = genre
        self.duration = duration
        self.complexity = complexity
        self.sample_rate = 44100
        
    def generate_music(self):
        # Map moods to musical parameters
        mood_params = {
            "Happy": {"scale": "major", "tempo": 120, "octave_base": 4},
            "Sad": {"scale": "minor", "tempo": 80, "octave_base": 3},
            "Energetic": {"scale": "major", "tempo": 140, "octave_base": 4},
            "Calm": {"scale": "major", "tempo": 70, "octave_base": 3},
            "Mysterious": {"scale": "minor", "tempo": 90, "octave_base": 3}
        }
        
        # Map genres to musical parameters
        genre_params = {
            "Classical": {"rhythm": "steady", "reverb": 0.3, "harmonics": 0.5},
            "Jazz": {"rhythm": "swing", "reverb": 0.2, "harmonics": 0.7},
            "Electronic": {"rhythm": "beat", "reverb": 0.5, "harmonics": 0.9},
            "Lo-Fi": {"rhythm": "irregular", "reverb": 0.8, "harmonics": 0.6}
        }
        
        # Get parameters for the selected mood and genre
        params = {**mood_params[self.mood], **genre_params[self.genre]}
        
        # Create scales
        scales = {
            "major": [0, 2, 4, 5, 7, 9, 11],  # C major: C D E F G A B
            "minor": [0, 2, 3, 5, 7, 8, 10]   # C minor: C D Eb F G Ab Bb
        }
        selected_scale = scales[params["scale"]]
        
        # Generate a melody
        n_samples = int(self.duration * self.sample_rate)
        melody = np.zeros(n_samples)
        
        # Tempo in beats per minute to seconds per beat
        beat_length = 60 / params["tempo"]
        samples_per_beat = int(beat_length * self.sample_rate)
        
        # Generate notes for the melody
        num_notes = self.duration * 4  # Approximately 4 notes per second
        melody_sequence = []
        
        # Start with the root note
        current_note = 0
        
        for i in range(int(num_notes)):
            # Choose a note from the scale
            if i == 0:
                note_idx = 0  # Start with root note
            else:
                # More complex melodies have larger jumps
                max_jump = 1 + int(self.complexity / 3)
                jump = random.randint(-max_jump, max_jump)
                
                # Keep within scale boundaries
                current_scale_pos = selected_scale.index(current_note % 12) if current_note % 12 in selected_scale else 0
                new_scale_pos = (current_scale_pos + jump) % len(selected_scale)
                new_note = selected_scale[new_scale_pos]
                
                # Adjust octave
                current_octave = current_note // 12
                if new_scale_pos < current_scale_pos and random.random() < 0.5:
                    current_octave += 1
                elif new_scale_pos > current_scale_pos and random.random() < 0.5:
                    current_octave -= 1
                
                # Keep within a reasonable range
                current_octave = max(params["octave_base"] - 1, min(params["octave_base"] + 2, current_octave))
                current_note = new_note + (current_octave * 12)
            
            # Convert note to frequency (A4 = 440Hz, and each semitone is a factor of 2^(1/12))
            note_freq = 440 * (2 ** ((current_note - 69) / 12))
            
            # Note duration varies by rhythm type
            if params["rhythm"] == "steady":
                note_duration = beat_length
            elif params["rhythm"] == "swing":
                note_duration = beat_length * (1.5 if i % 2 == 0 else 0.5)
            elif params["rhythm"] == "beat":
                note_duration = beat_length * (0.5 if i % 4 == 0 else 0.25)
            else:  # irregular
                note_duration = beat_length * random.choice([0.25, 0.5, 0.75, 1.0])
            
            melody_sequence.append((note_freq, note_duration))
        
        # Convert melody sequence to audio
        position = 0
        for freq, duration in melody_sequence:
            # Number of samples for this note
            note_samples = int(duration * self.sample_rate)
            
            # Ensure we don't go out of bounds
            if position + note_samples > n_samples:
                note_samples = n_samples - position
            
            if note_samples <= 0:
                continue
            
            # Generate sine wave for the note
            t = np.linspace(0, duration, note_samples, False)
            
            # Add harmonics based on genre
            note = 0.5 * np.sin(2 * np.pi * freq * t)  # Fundamental
            if params["harmonics"] > 0.2:
                note += 0.3 * params["harmonics"] * np.sin(2 * np.pi * freq * 2 * t)  # 1st harmonic
            if params["harmonics"] > 0.5:
                note += 0.15 * params["harmonics"] * np.sin(2 * np.pi * freq * 3 * t)  # 2nd harmonic
            
            # Apply envelope (ADSR: Attack, Decay, Sustain, Release)
            attack = int(0.1 * note_samples)
            decay = int(0.1 * note_samples)
            release = int(0.2 * note_samples)
            sustain_level = 0.7
            
            envelope = np.ones(note_samples)
            # Attack
            envelope[:attack] = np.linspace(0, 1, attack)
            # Decay
            envelope[attack:attack+decay] = np.linspace(1, sustain_level, decay)
            # Release
            if release > 0:
                envelope[-release:] = np.linspace(sustain_level, 0, release)
            
            # Apply envelope to note
            note = note * envelope
            
            # Add to melody
            melody[position:position+note_samples] += note
            position += note_samples
        
        # Normalize
        melody = melody / (np.max(np.abs(melody)) + 1e-6)
        
        # Apply reverb if specified
        if params["reverb"] > 0:
            reverb_length = int(params["reverb"] * self.sample_rate)
            reverb_filter = np.exp(-np.linspace(0, 5, reverb_length))
            melody = signal.convolve(melody, reverb_filter, mode='full')[:n_samples]
            melody = melody / (np.max(np.abs(melody)) + 1e-6)
        
        return melody, self.sample_rate

# Generate AI Music
if st.button("🎶 Generate Music"):
    if not api_key:
        st.error("Please enter your API key")
    else:
        try:
            with st.spinner("Composing your music... 🎵"):
                # Generate AI music
                generator = MusicGenerator(mood, genre, duration, complexity)
                audio_data, sample_rate = generator.generate_music()
                
                # Save as WAV using temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    sf.write(tmp_file.name, audio_data, sample_rate)
                    
                    # Display audio player
                    st.audio(tmp_file.name, format="audio/wav")
                    
                    # Description from Gemini (if API key provided)
                    try:
                        model = genai.GenerativeModel("gemini-1.5-flash")
                        prompt = f"Describe a {duration}-second {genre} music piece with {mood} mood in 1-2 sentences. Be concise and evocative."
                        response = model.generate_content(prompt)
                        st.markdown(f"**AI Description**: _{response.text}_")
                    except Exception as e:
                        st.warning(f"Couldn't generate description: {str(e)}")
                    
                    # Download button
                    with open(tmp_file.name, "rb") as f:
                        st.download_button(
                            "Download Music", 
                            data=f.read(), 
                            file_name=f"{mood}_{genre}_music.wav", 
                            mime="audio/wav"
                        )
                
                # Clean up temp file
                os.unlink(tmp_file.name)
                
                st.success("Music generated successfully!")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
