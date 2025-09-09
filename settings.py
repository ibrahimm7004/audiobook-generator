import os
import platform
from pathlib import Path
from pydub import AudioSegment
from dotenv import load_dotenv
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Secrets Handling ---
if platform.system() == "Windows":  # Local dev
    load_dotenv()  # load from .env

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
APP_PASSWORD_CLIENT = os.environ.get("APP_PASSWORD_CLIENT")
APP_PASSWORD_TEAM = os.environ.get("APP_PASSWORD_TEAM")

VALID_PASSWORDS = {
    APP_PASSWORD_CLIENT: "client",
    APP_PASSWORD_TEAM: "team",
}

# --- Hybrid FFmpeg Handling ---
if platform.system() == "Windows":  # Local Windows → bundled ffmpeg
    FFMPEG_DIR = os.path.join(PROJECT_ROOT, "ffmpeg", "bin")
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ["PATH"]
else:  # Streamlit Cloud (Linux) → system ffmpeg
    AudioSegment.converter = shutil.which("ffmpeg") or "ffmpeg"
    AudioSegment.ffprobe = shutil.which("ffprobe") or "ffprobe"
