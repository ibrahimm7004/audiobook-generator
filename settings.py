import os
import platform
import shutil
from pathlib import Path
from pydub import AudioSegment
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Secrets Handling (prefer st.secrets) ---
ELEVENLABS_API_KEY = st.secrets.get("elevenlabs", {}).get(
    "ELEVENLABS_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
APP_PASSWORD_CLIENT = st.secrets.get("app", {}).get(
    "APP_PASSWORD_CLIENT") or os.getenv("APP_PASSWORD_CLIENT")
APP_PASSWORD_TEAM = st.secrets.get("app", {}).get(
    "APP_PASSWORD_TEAM") or os.getenv("APP_PASSWORD_TEAM")

OPENAI_API_KEY = st.secrets.get("openai", {}).get(
    "OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# AWS settings from Streamlit secrets (fallback to env for local/dev)
AWS_ACCESS_KEY_ID = st.secrets.get("aws", {}).get(
    "AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = st.secrets.get("aws", {}).get(
    "AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = st.secrets.get("aws", {}).get(
    "AWS_DEFAULT_REGION") or os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION")
AWS_S3_BUCKET = st.secrets.get("aws", {}).get(
    "AWS_S3_BUCKET") or os.getenv("AWS_S3_BUCKET") or os.getenv("S3_BUCKET")


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
