import streamlit as st
import google.generativeai as genai

# ---------- Page Config ----------
st.set_page_config(
    page_title="LostLanguages AI",
    page_icon="📜",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS Theme ----------
st.markdown("""
    <style>
    body {
        background-image: url('https://www.transparenttextures.com/patterns/paper-fibers.png');
        background-color: #1a1a1a;
        color: #e0dcbf;
    }
    .stApp {
        background: linear-gradient(135deg, #1f1f1f 0%, #2d2d2d 100%);
        font-family: 'Garamond', serif;
    }
    h1, h2, h3, h4 {
        color: #e6c967 !important;
        text-shadow: 0 0 4px #b29f57;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    textarea, .stTextInput>div>div>input {
        background-color: #2c2c2c !important;
        color: #f1e8d7 !important;
        border: 1px solid #8a7737 !important;
    }
    button {
        border-radius: 0px !important;
    }
    .stButton>button {
        background-color: #3d2f1b;
        color: #ffd700;
        border: 1px solid #b49d5b;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #5a421d;
        color: #fff8dc;
    }
    .sidebar .sidebar-content {
        background-color: #1c1c1c;
        color: #f5deb3;
    }
    footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.title("🔐 Ancient Oracle")
api_key = st.sidebar.text_input("Gemini API Key", type="password")

# ---------- Title ----------
st.markdown("""
    <div style="text-align:center;">
        <h1>📜 LostLanguages AI</h1>
        <p style="color:#c0b283;">Unveil the whispers of forgotten worlds...</p>
    </div>
""", unsafe_allow_html=True)

# ---------- API Setup ----------
if not api_key:
    st.warning("Please enter your Gemini API key in the sidebar to begin your linguistic quest.")
    st.stop()
else:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# ---------- Input Form ----------
st.subheader("🔍 Provide Ancient Symbols / Description")
user_input = st.text_area("Enter symbols, fragments, or language traits", height=200)

simulate = st.checkbox("✨ Simulate a mythical conversation in this language")
generate_btn = st.button("🔓 Reveal Lost Meanings")

# ---------- AI Logic ----------
if generate_btn and user_input:
    with st.spinner("Translating the echoes of ancient minds..."):

        prompt = f"""
        You are an AI expert in ancient and fictional languages.
        Input: {user_input}

        Step 1: Hypothesize the language structure based on ancient patterns.
        Step 2: Attempt a symbolic or contextual translation.
        Step 3: Explain your translation logic.

        {'Step 4: Simulate a conversation in this language between two fictional individuals, with English translations.' if simulate else ''}
        """

        try:
            response = model.generate_content(prompt)
            st.success("✨ Ancient Insight Unlocked:")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ---------- Footer ----------
st.markdown("---")
st.markdown("<div style='text-align:center; color: #b4a077;'>🔮 Crafted by the Oracle of Lost Scripts ✧</div>", unsafe_allow_html=True)
