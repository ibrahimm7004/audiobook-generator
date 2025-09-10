import os
import platform
import shutil
from pathlib import Path
from pydub import AudioSegment
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Hybrid Secrets Handling ---
if platform.system() == "Windows":  # Local
    load_dotenv()

# Always pull from env vars first
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
APP_PASSWORD_CLIENT = os.getenv("APP_PASSWORD_CLIENT")
APP_PASSWORD_TEAM = os.getenv("APP_PASSWORD_TEAM")

# Fallback: if not found, try st.secrets (only works on Streamlit Cloud)
if not ELEVENLABS_API_KEY and "ELEVENLABS_API_KEY" in st.secrets:
    ELEVENLABS_API_KEY = st.secrets["ELEVENLABS_API_KEY"]
    APP_PASSWORD_CLIENT = st.secrets["APP_PASSWORD_CLIENT"]
    APP_PASSWORD_TEAM = st.secrets["APP_PASSWORD_TEAM"]

VALID_PASSWORDS = {
    APP_PASSWORD_CLIENT: "client",
    APP_PASSWORD_TEAM: "team",
}

# --- Hybrid FFmpeg Handling ---
if platform.system() == "Windows":  # Local Windows → bundled ffmpeg
    FFMPEG_DIR = os.path.join(PROJECT_ROOT, "ffmpeg", "bin")
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ["PATH"]
else:  # Linux → system ffmpeg
    AudioSegment.converter = shutil.which("ffmpeg") or "ffmpeg"
    AudioSegment.ffprobe = shutil.which("ffprobe") or "ffprobe"
