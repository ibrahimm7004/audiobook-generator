import streamlit as st
import os
import tempfile
import re
import json
from pathlib import Path
from elevenlabs import ElevenLabs
import io
import base64
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import hashlib
from datetime import datetime
import platform
import shutil
import imageio_ffmpeg
from pydub import AudioSegment, effects
from dotenv import load_dotenv
import streamlit as st

ELEVENLABS_API_KEY = st.secrets["ELEVENLABS_API_KEY"]
APP_PASSWORD_CLIENT = st.secrets["APP_PASSWORD_CLIENT"]
APP_PASSWORD_TEAM = st.secrets["APP_PASSWORD_TEAM"]

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if platform.system() == "Windows":
    # Local Windows → bundled ffmpeg
    ffmpeg_dir = os.path.join(PROJECT_ROOT, "ffmpeg", "bin")
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
else:
    # Streamlit Cloud (Linux) → imageio-ffmpeg for ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    AudioSegment.converter = ffmpeg_path

    # ffprobe is not provided by imageio → fall back to system-installed ffprobe
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        AudioSegment.ffprobe = ffprobe_path
    else:
        # fallback: let it warn, but won’t block speech decoding
        print("⚠️ ffprobe not found — continuing without it.")

BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "configs"

with open(CONFIG_DIR / "character_voices.json", "r", encoding="utf-8") as f:
    CHARACTER_VOICES = json.load(f)

with open(CONFIG_DIR / "emotion_tags.json", "r", encoding="utf-8") as f:
    EMOTION_TAGS = json.load(f)

with open(CONFIG_DIR / "speech_verbs.json", "r", encoding="utf-8") as f:
    _SPEECH_VERBS = json.load(f)

with open(CONFIG_DIR / "adverb_to_emotion.json", "r", encoding="utf-8") as f:
    _ADVERB_TO_EMOTION = json.load(f)

with open(CONFIG_DIR / "verb_to_emotion.json", "r", encoding="utf-8") as f:
    _VERB_TO_EMOTION = json.load(f)


def normalize_effect_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


def load_sound_effects(fx_dir="fx_library"):
    effects_map = {}
    for file in Path(fx_dir).glob("*.wav"):
        norm = normalize_effect_name(file.stem)
        effects_map[norm] = str(file.resolve())
    return effects_map


# Auto-load all effects at startup
SOUND_EFFECTS = load_sound_effects()


def get_flat_character_voices():
    """Flatten CHARACTER_VOICES into {character_name: voice_id}"""
    flat_voices = {}
    for category in CHARACTER_VOICES.values():
        for char, vid in category.items():
            flat_voices[char] = vid
    return flat_voices


def get_flat_emotion_tags():
    flat_emotions = {}
    for category in EMOTION_TAGS.values():
        flat_emotions.update(category)
    return flat_emotions


# Normalize curly quotes to straight quotes
_QUOTE_NORMALIZER = [
    ("\u201c", '"'), ("\u201d", '"'), ("\u201e", '"'), ("\u201f", '"'),
    ("\u2033", '"'), ("\u2036", '"'),
]

VALID_PASSWORDS = {
    APP_PASSWORD_CLIENT: "client",
    APP_PASSWORD_TEAM: "team",
}


@dataclass
class ParsedDialogue:
    """Data class for parsed dialogue line"""
    character: str
    text: str
    emotions: List[str]
    sound_effects: List[str]
    line_number: int
    original_line: str


@dataclass
class ParseAnalysis:
    """Data class for text parsing analysis"""
    characters_found: Dict[str, int]
    emotions_found: Dict[str, int]
    sound_effects_found: Dict[str, int]
    unsupported_characters: List[str]
    unsupported_emotions: List[str]
    unsupported_sound_effects: List[str]
    total_lines: int
    dialogue_lines: int


def check_password():
    """Check if user has entered correct password"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_type = None

    if not st.session_state.authenticated:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin: 2rem 0;
        ">
            <h1 style="color: white; margin-bottom: 1rem;">🎭 Audiomachine project</h1>
            <p style="color: white; opacity: 0.9;">Professional Audiobook Production Suite</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔐 Enter Access Password")
        password = st.text_input(
            "Password:", type="password", key="password_input")

        if st.button("🚀 Access System", type="primary", use_container_width=True):
            if password in VALID_PASSWORDS:
                st.session_state.authenticated = True
                st.session_state.user_type = VALID_PASSWORDS[password]
                st.success("✅ Access granted! Redirecting...")
                st.rerun()
            else:
                st.error(
                    "❌ Invalid password. Please contact the administrator for access.")

        st.markdown("---")
        st.info("🔒 This system is password-protected for authorized users only.")
        return False

    return True


def create_output_folders():
    """Create organized output folder structure"""
    base_output = Path("./audio_output")
    base_output.mkdir(exist_ok=True)

    # Create subfolders
    folders = {
        "teasers": base_output / "teasers",
        "chapters": base_output / "chapters",
        "voice_tests": base_output / "voice_tests",
        "books": base_output / "books"
    }

    for folder in folders.values():
        folder.mkdir(exist_ok=True)

    return folders


def save_voice_mappings(mappings: Dict[str, str], project_name: str):
    """Save voice mappings to JSON file"""
    mappings_dir = Path("./voice_mappings")
    mappings_dir.mkdir(exist_ok=True)

    # Clean project name for filename
    clean_name = re.sub(r'[^\w\s-]', '', project_name).strip()
    clean_name = re.sub(r'[-\s]+', '_', clean_name)

    mappings_file = mappings_dir / f"{clean_name}_voices.json"

    with open(mappings_file, 'w') as f:
        json.dump({
            "project_name": project_name,
            "created_date": datetime.now().isoformat(),
            "voice_mappings": mappings
        }, f, indent=2)

    return mappings_file


def load_voice_mappings(project_name: str) -> Optional[Dict[str, str]]:
    """Load voice mappings from JSON file"""
    mappings_dir = Path("./voice_mappings")
    if not mappings_dir.exists():
        return None

    clean_name = re.sub(r'[^\w\s-]', '', project_name).strip()
    clean_name = re.sub(r'[-\s]+', '_', clean_name)

    mappings_file = mappings_dir / f"{clean_name}_voices.json"

    if mappings_file.exists():
        try:
            with open(mappings_file, 'r') as f:
                data = json.load(f)
                return data.get("voice_mappings", {})
        except Exception as e:
            st.warning(f"Error loading voice mappings: {e}")

    return None


def get_saved_voice_projects() -> List[str]:
    """Get list of saved voice mapping projects"""
    mappings_dir = Path("./voice_mappings")
    if not mappings_dir.exists():
        return []

    projects = []
    for file in mappings_dir.glob("*.json"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                projects.append(data.get("project_name", file.stem))
        except:
            continue

    return sorted(projects)


@dataclass
class RawParseResult:
    formatted_text: str
    dialogues: List[Dict]
    stats: Dict[str, int]

# --- NEW PARSER ---------------------------------------------------------------


class RawProseParser:
    """Converts raw prose into standardized dialogue lines that your existing
    DialogueParser can consume: [Character] (emotion1)(emotion2): text *fx*.

    Heuristics used:
    1) Detect quoted dialogue spans (supports smart quotes → normalized).
    2) Attribute speaker from nearby attribution patterns:
       - "...," said Dante.  |  Dante said, "..."
       - verb list is extensible via _SPEECH_VERBS
       - chooses among known characters first; else falls back to last speaker; else Narrator.
    3) Infer basic emotion from verb/adverb (maps to your existing tags if present).
    4) Convert obvious narration cues into FX (optional), e.g., "gasped" → *gasps*.
    5) Optionally include narration between dialogues as [Narrator]: lines.
    """

    def __init__(self, known_characters: Optional[List[str]] = None,
                 include_narration: bool = True,
                 attach_detected_fx: bool = True):
        self.known_characters = [c.strip()
                                 for c in (known_characters or []) if c.strip()]
        self.include_narration = include_narration
        self.attach_detected_fx = attach_detected_fx

        # normalize smart quotes → plain quotes
        self.quote_pat = re.compile(r'"([^"\n]+)"')

        # Build regex with case-insensitive verbs, case-sensitive names
        verb_alt = "|".join(_SPEECH_VERBS)

        # e.g. “said Dante”
        self.right_said_name = re.compile(
            rf"(?:^|[\s,;—-])(?P<verb>(?i:{verb_alt}))\s+(?P<name>(?-i:[A-Z][a-zA-Z]+))\b"
        )
        # e.g. “Dante said”
        self.right_name_said = re.compile(
            rf"(?:^|[\s,;—-])(?P<name>(?-i:[A-Z][a-zA-Z]+))\s+(?P<verb>(?i:{verb_alt}))\b"
        )
        # e.g. “…, Dante said.”
        self.left_name_said = re.compile(
            rf"(?P<name>(?-i:[A-Z][a-zA-Z]+))\s+(?P<verb>(?i:{verb_alt}))\b"
        )
        # e.g. “…,” he said.
        self.right_pronoun_said = re.compile(
            rf"(?:^|[\s,;—-])(?P<pronoun>(?i:he|she|they))\s+(?P<verb>(?i:{verb_alt}))\b"
        )

        # helper patterns
        self.name_token = re.compile(r"\b([A-Z][a-zA-Z]+)\b")
        self.adverb_pat = re.compile(r"\b([A-Za-z]+)ly\b")
        self.sent_split = re.compile(r"([.!?]\s+)")

    def _normalize_quotes(self, text: str) -> str:
        for frm, to in _QUOTE_NORMALIZER:
            text = text.replace(frm, to)
        return text

    def _infer_emotions(self, context: str, verb: Optional[str]) -> List[str]:
        emotions = []
        if verb:
            v = verb.lower()
            if v in _VERB_TO_EMOTION:
                emotions.append(_VERB_TO_EMOTION[v])
        for adv in self.adverb_pat.findall(context or ""):
            tag = _ADVERB_TO_EMOTION.get(adv.lower())
            if tag and tag not in emotions:
                emotions.append(tag)
        return emotions

    def _choose_speaker(self, candidates: List[str], last_speaker: Optional[str]) -> str:
        # prefer known characters, then last speaker, else Narrator
        for c in candidates:
            if c in self.known_characters:
                return c
        if last_speaker and (not candidates or last_speaker in candidates or not self.known_characters):
            return last_speaker
        # If any candidate remains, pick the first capitalized name-like token
        if candidates:
            return candidates[0]
        return "Narrator"

    def _nearest_prior_known(self, context: str) -> Optional[str]:
        """Look back in left context for the last capitalized token that matches a known character."""
        if not context or not self.known_characters:
            return None
        candidates = self.name_token.findall(context)
        # Walk backwards to find nearest match
        for token in reversed(candidates):
            if token in self.known_characters:
                return token
        return None

    def convert(self, raw_text: str) -> RawParseResult:
        text = self._normalize_quotes(raw_text or "").strip()
        if not text:
            return RawParseResult(formatted_text="", dialogues=[], stats={})

        paragraphs = [p.strip()
                      for p in re.split(r"\n\s*\n+", text) if p.strip()]

        dialogues: List[Dict] = []
        formatted_lines: List[str] = []
        last_speaker: Optional[str] = None
        stats = {
            "quotes_found": 0,
            "lines_emitted": 0,
            "speaker_from_after": 0,
            "speaker_from_before": 0,
            "speaker_from_last": 0,
            "speaker_unknown": 0,
            "narration_blocks": 0,
        }

        for para in paragraphs:
            spans = list(self.quote_pat.finditer(para))

            # narration-only paragraph
            if not spans:
                if self.include_narration:
                    fx_list = self._scan_for_fx(para)
                    fx_text = ''.join(f"*{fx}*" for fx in fx_list)
                    formatted_lines.append(
                        f"[Narrator]: {para} {fx_text}".strip())
                    dialogues.append({
                        "type": "speech",
                        "character": "Narrator",
                        "text": para,
                        "emotions": [],
                        "fx": fx_list,
                    })
                    stats["narration_blocks"] += 1
                continue

            for i, m in enumerate(spans):
                stats["quotes_found"] += 1
                quote_text = m.group(1).strip()

                # Tight windows per quote (BrE-friendly: everything outside the quotes is fair game)
                prev_end = spans[i-1].end() if i > 0 else 0
                next_start = spans[i+1].start() if i + \
                    1 < len(spans) else len(para)
                left_ctx = para[max(prev_end, m.start()-160): m.start()]
                right_ctx = para[m.end(): min(next_start, m.end()+160)]

                candidates: List[str] = []
                inferred_emotions: List[str] = []

                # --- 1) Pre-quote: "Name said ..." in LEFT context (nearest to the quote)
                before_matches = list(self.left_name_said.finditer(left_ctx))
                if before_matches:
                    # the nearest "Name said" before the quote
                    bm = before_matches[-1]
                    name = bm.group("name")
                    candidates.append(name)
                    verb = bm.group("verb")
                    inferred_emotions.extend(
                        self._infer_emotions(bm.group(0), verb))
                    stats["speaker_from_before"] += 1

                # --- 2) Post-quote: prefer "Name said ..." in RIGHT context
                if not candidates:
                    after_name_first = self.right_name_said.search(
                        right_ctx)  # "Rafael said ..."
                    after_said_name = self.right_said_name.search(
                        right_ctx)  # "said Dante"

                    # choose whichever is closer to the quote start
                    best = None
                    if after_name_first and after_said_name:
                        best = after_name_first if after_name_first.start(
                        ) < after_said_name.start() else after_said_name
                    else:
                        best = after_name_first or after_said_name

                    if best:
                        name = best.group("name")
                        candidates.append(name)
                        verb = best.group("verb")
                        inferred_emotions.extend(
                            self._infer_emotions(best.group(0), verb))
                        stats["speaker_from_after"] += 1

                # --- 3) Pronoun attribution: "...," he/she/they VERB ...
                if not candidates:
                    pron = self.right_pronoun_said.search(right_ctx)
                    if pron:
                        verb = pron.group("verb")
                        inferred_emotions.extend(
                            self._infer_emotions(pron.group(0), verb))
                        nearest = self._nearest_prior_known(left_ctx)
                        if nearest:
                            candidates.append(nearest)

                # --- 4) Final speaker choice
                speaker = self._choose_speaker(candidates, last_speaker)
                if speaker == last_speaker and not candidates:
                    stats["speaker_from_last"] += 1
                if speaker == "Narrator" and not candidates:
                    stats["speaker_unknown"] += 1

                # emit
                emotion_text = ''.join(
                    f"({e})" for e in dict.fromkeys(inferred_emotions) if e)
                formatted_lines.append(
                    f"[{speaker}] {emotion_text}: {quote_text}".strip())
                dialogues.append({
                    "type": "speech",
                    "character": speaker,
                    "text": quote_text,
                    "emotions": inferred_emotions,
                    "fx": [],
                })
                stats["lines_emitted"] += 1
                last_speaker = speaker

            # trailing narration after the last quote
            tail = para[spans[-1].end():]

            # --- Fix stray leading punctuation from BrE (".”" then . outside) ---
            if tail:  # don't .strip() yet; we want to inspect the very first char
                # 1) Optional: attach a single leading sentence-final mark to the last spoken line
                #    (Uncomment this block if you want the period to be added to the previous dialogue line)
                """
                if tail.lstrip().startswith(('.', '!', '?', '…')):
                    # find the very first non-space char
                    i = 0
                    while i < len(tail) and tail[i].isspace():
                        i += 1
                    if i < len(tail) and tail[i] in '.!?…':
                        # append to the last emitted dialogue line if present
                        if formatted_lines and formatted_lines[-1].startswith('[') and ']: ' in formatted_lines[-1]:
                            formatted_lines[-1] = formatted_lines[-1] + tail[i]
                        # drop that punctuation from tail
                        tail = tail[:i] + tail[i+1:]
                """

                # 2) Always: if the remaining head of tail is ONLY punctuation/whitespace,
                #    drop it so we don't emit "[Narrator]: ."
                #    (This also covers the case where you don't want to attach it.)
                head = re.match(r'^\s*([.!?…]+)\s*', tail)
                if head:
                    # remove the leading punctuation token entirely
                    tail = tail[head.end():]

            tail = tail.strip()
            if tail and self.include_narration:
                fx_list = self._scan_for_fx(tail)
                fx_text = ''.join(f"*{fx}*" for fx in fx_list)
                formatted_lines.append(f"[Narrator]: {tail} {fx_text}".strip())
                dialogues.append({
                    "type": "speech",
                    "character": "Narrator",
                    "text": tail,
                    "emotions": [],
                    "fx": fx_list,
                })
                stats["narration_blocks"] += 1

        return RawParseResult("\n".join(formatted_lines), dialogues, stats)


class FileExtractor:
    """Extract text from various file formats"""

    @staticmethod
    def extract_from_txt(file_content: bytes) -> str:
        """Extract text from .txt file"""
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return file_content.decode('latin-1')
            except:
                return file_content.decode('utf-8', errors='ignore')

    @staticmethod
    def extract_from_docx(file_content: bytes) -> str:
        """Extract text from .docx file"""
        try:
            # Try using python-docx if available
            import docx
            from io import BytesIO

            doc = docx.Document(BytesIO(file_content))
            text_parts = []

            for paragraph in doc.paragraphs:
                text_parts.append(paragraph.text)

            return '\n'.join(text_parts)

        except ImportError:
            st.warning(
                "python-docx not available. Attempting basic text extraction...")
            # Fallback: try to extract readable text
            text = file_content.decode('utf-8', errors='ignore')
            # Remove common docx artifacts
            text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
            return text
        except Exception as e:
            st.error(f"Error extracting from DOCX: {e}")
            return file_content.decode('utf-8', errors='ignore')

    @staticmethod
    def extract_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            import PyPDF2
            from io import BytesIO

            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text_parts = []

            for page in pdf_reader.pages:
                text_parts.append(page.extract_text())

            return '\n'.join(text_parts)

        except ImportError:
            st.warning("PyPDF2 not available. Cannot extract PDF text.")
            return "Error: PDF extraction requires PyPDF2 library"
        except Exception as e:
            st.error(f"Error extracting from PDF: {e}")
            return f"Error extracting PDF: {e}"


class TextParser:
    """Advanced text parser for various dialogue formats"""

    def __init__(self):
        self.character_voices = get_flat_character_voices()
        self.emotion_tags = get_flat_emotion_tags()
        self.sound_effects = SOUND_EFFECTS

    def analyze_text(self, text: str) -> ParseAnalysis:
        """Analyze text and provide comprehensive statistics"""
        lines = text.strip().split('\n')
        characters_found = defaultdict(int)
        emotions_found = defaultdict(int)
        sound_effects_found = defaultdict(int)
        unsupported_characters = set()
        unsupported_emotions = set()
        unsupported_sound_effects = set()
        dialogue_lines = 0

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Try different dialogue formats
            parsed = self._parse_line_flexible(line)
            if parsed:
                dialogue_lines += 1

                # Count character usage
                characters_found[parsed.character] += 1
                if parsed.character not in self.character_voices:
                    unsupported_characters.add(parsed.character)

                # Count emotions
                for emotion in parsed.emotions:
                    emotions_found[emotion] += 1
                    if emotion not in self.emotion_tags:
                        unsupported_emotions.add(emotion)

                for effect in parsed.sound_effects:
                    sound_effects_found[effect] += 1
                    norm_effect = normalize_effect_name(effect)
                    if norm_effect not in self.sound_effects:
                        unsupported_sound_effects.add(effect)

        return ParseAnalysis(
            characters_found=dict(characters_found),
            emotions_found=dict(emotions_found),
            sound_effects_found=dict(sound_effects_found),
            unsupported_characters=list(unsupported_characters),
            unsupported_emotions=list(unsupported_emotions),
            unsupported_sound_effects=list(unsupported_sound_effects),
            total_lines=len(lines),
            dialogue_lines=dialogue_lines
        )

    def _parse_line_flexible(self, line: str) -> Optional[ParsedDialogue]:
        """Parse a single line with flexible format support"""
        original_line = line

        # Format 1: [Character] (emotion): Text *effect*
        char_match = re.match(r'\[([^\]]+)\]\s*(\([^)]+\))*\s*:', line)
        if char_match:
            character = char_match.group(1).strip()
            emotion_matches = re.findall(r'\(([^)]+)\)', char_match.group(0))
            text_part = re.sub(
                r'^\[([^\]]+)\]\s*(\([^)]+\))*\s*:', '', line).strip()

        # Format 2: Character: Text (no brackets)
        elif ':' in line and not line.startswith('['):
            parts = line.split(':', 1)
            if len(parts) == 2:
                character = parts[0].strip()
                text_part = parts[1].strip()
                emotion_matches = []
            else:
                return None

        # Format 3: "Character said" or Character said
        elif ' said' in line.lower():
            said_match = re.match(
                r'(["\']?)([^"\']+)\1\s+said[:\s]*(.+)', line, re.IGNORECASE)
            if said_match:
                character = said_match.group(2).strip()
                text_part = said_match.group(3).strip()
                emotion_matches = []
            else:
                return None

        # Format 4: Narrative with quoted speech
        elif '"' in line or '"' in line or '"' in line:
            # Extract character name (usually before the quote or after)
            quote_match = re.search(r'["""]([^"""]+)["""]', line)
            if quote_match:
                text_part = quote_match.group(1).strip()
                # Try to find character name in the remaining text
                remaining = line.replace(quote_match.group(0), '').strip()
                # Look for capitalized words that might be names
                name_candidates = re.findall(r'\b[A-Z][a-zA-Z]+\b', remaining)
                character = name_candidates[0] if name_candidates else "Unknown"
                emotion_matches = []
            else:
                return None
        else:
            return None

        # Extract sound effects and clean text
        sound_effects = re.findall(r'\*([^*]+)\*', text_part)
        clean_text = re.sub(r'\*[^*]+\*', '', text_part).strip()

        if not clean_text:
            return None

        return ParsedDialogue(
            character=character,
            text=clean_text,
            emotions=[e.strip() for e in emotion_matches],
            sound_effects=[e.strip() for e in sound_effects],
            line_number=0,  # Will be set later
            original_line=original_line
        )

    def parse_to_dialogue_format(self, text: str) -> Tuple[str, List[ParsedDialogue]]:
        """Parse text and convert to standard dialogue format"""
        lines = text.strip().split('\n')
        parsed_dialogues = []
        formatted_lines = []

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                formatted_lines.append(line)
                continue

            parsed = self._parse_line_flexible(line)
            if parsed:
                parsed.line_number = line_num
                parsed_dialogues.append(parsed)

                # Format to standard format
                emotion_text = ''.join(
                    [f"({emotion})" for emotion in parsed.emotions])
                effect_text = ''.join(
                    [f"*{effect}*" for effect in parsed.sound_effects])

                formatted_line = f"[{parsed.character}] {emotion_text}: {parsed.text} {effect_text}".strip(
                )
                formatted_lines.append(formatted_line)
            else:
                formatted_lines.append(f"# UNPARSED: {line}")

        return '\n'.join(formatted_lines), parsed_dialogues


class VoiceManager:
    """Manage character voice assignments"""

    def __init__(self):
        self.voice_assignments = get_flat_character_voices().copy()

    def get_available_voices(self) -> Dict[str, Dict[str, str]]:
        """Get all available voices with descriptions"""
        return CHARACTER_VOICES

    def assign_voice(self, character: str, voice_id: str):
        """Assign a voice to a character"""
        self.voice_assignments[character] = voice_id

    def get_voice_for_character(self, character: str) -> Optional[str]:
        """Get voice ID for character"""
        return self.voice_assignments.get(character)

    def get_character_description(self, character: str) -> str:
        """Get description of character's current voice"""
        voice_id = self.voice_assignments.get(character)
        if not voice_id:
            return "No voice assigned"

        for category in CHARACTER_VOICES.values():
            for char, data in category.items():
                if data["voice_id"] == voice_id:
                    return data["description"]
        return "Custom voice assignment"


class DialogueAudioGenerator:
    def __init__(self, api_key=ELEVENLABS_API_KEY, fx_library_path="./fx_library/"):
        """Initialize the dialogue audio generator"""
        self.client = ElevenLabs(api_key=api_key)
        self.fx_library_path = Path(fx_library_path)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.character_voices = get_flat_character_voices()
        self.sound_effects = SOUND_EFFECTS
        self.output_folders = create_output_folders()

    def apply_post_processing(self, audio: AudioSegment) -> AudioSegment:
        """Apply mastering chain: normalization, compression, fades, EQ/gain matching"""
        if not audio or len(audio) == 0:
            return audio

        try:
            # 1. Normalize loudness (target RMS ~ -20 dBFS)
            normalized = effects.normalize(audio, headroom=1.0)

            # 2. Light compression (tames peaks but keeps natural range)
            compressed = effects.compress_dynamic_range(
                normalized,
                threshold=-18.0,   # start compressing above -18 dBFS
                ratio=2.0,         # gentle compression
                attack=5.0,        # ms
                release=50.0       # ms
            )

            # 3. Add fades at edges (avoid clicks/pops)
            faded = compressed.fade_in(20).fade_out(50)

            # 4. Gain match fallback voices if needed (tweakable constant)
            # adjust if non-ElevenLabs sounds are quieter/louder
            final = faded.apply_gain(0.0)

            return final

        except Exception as e:
            st.warning(f"Post-processing error: {e}")
            return audio

    def generate_speech(self, voice_id, text, model_id="eleven_v3"):
        """Generate TTS audio and return as AudioSegment"""
        try:
            # Ensure text is not empty
            if not text or not text.strip():
                return AudioSegment.silent(duration=500)

            audio_generator = self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=model_id,
            )

            temp_file = self.temp_dir / f"temp_{voice_id}_{hash(text)}.mp3"

            # Write all audio data to file
            with open(temp_file, "wb") as f:
                for chunk in audio_generator:
                    if chunk:  # Ensure chunk is not empty
                        f.write(chunk)

            # Verify file was created and has content
            if temp_file.exists() and temp_file.stat().st_size > 0:
                audio = AudioSegment.from_mp3(temp_file)
                # Ensure audio has minimum duration
                if len(audio) < 100:  # If less than 100ms, add some silence
                    audio = audio + AudioSegment.silent(duration=100)
                return audio
            else:
                st.warning(f"Failed to generate audio for: {text[:30]}...")
                return AudioSegment.silent(duration=1000)

        except Exception as e:
            st.error(f"Error generating speech for '{text[:30]}...': {e}")
            return AudioSegment.silent(duration=1000)

    def load_sound_effect(self, effect_name):
        """Load sound effect from fx library using configured mappings"""
        if effect_name not in self.sound_effects:
            st.warning(
                f"Unknown sound effect: '{effect_name}'. Available: {list(self.sound_effects.keys())}")
            return AudioSegment.silent(duration=500)

        filename = self.sound_effects[effect_name]
        fx_file = self.fx_library_path / filename

        if not fx_file.exists():
            st.warning(f"Sound effect file not found: {fx_file}")
            return AudioSegment.silent(duration=500)

        try:
            ext = fx_file.suffix.lower()
            if ext == '.mp3':
                return AudioSegment.from_mp3(fx_file)
            elif ext == '.wav':
                return AudioSegment.from_wav(fx_file)
            elif ext == '.m4a':
                return AudioSegment.from_file(fx_file, format="m4a")
            elif ext == '.ogg':
                return AudioSegment.from_ogg(fx_file)
            else:
                return AudioSegment.from_file(fx_file)
        except Exception as e:
            st.warning(f"Error loading {fx_file}: {e}")
            return AudioSegment.silent(duration=500)

    def _save_to_organized_folder(self, audio_data: bytes, output_type: str, project_name: str):
        """Save audio to organized folder structure"""
        try:
            # Clean project name for filename
            clean_name = re.sub(r'[^\w\s-]', '', project_name).strip()
            clean_name = re.sub(r'[-\s]+', '_', clean_name)

            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Determine output folder and filename
            if output_type == "teaser":
                folder = self.output_folders["teasers"]
                filename = f"{clean_name}_teaser_{timestamp}.mp3"
            elif output_type == "voice_test":
                folder = self.output_folders["voice_tests"]
                filename = f"voice_test_{timestamp}.mp3"
            elif output_type == "chapter":
                folder = self.output_folders["chapters"]
                filename = f"{clean_name}_chapter_{timestamp}.mp3"
            else:
                folder = self.output_folders["books"]
                filename = f"{clean_name}_{timestamp}.mp3"

            # Save file
            output_path = folder / filename
            with open(output_path, 'wb') as f:
                f.write(audio_data)

            st.success(f"💾 Audio saved to: {output_path}")

        except Exception as e:
            st.warning(f"Could not save to organized folder: {e}")

    def _ensure_compatible(self, segment: AudioSegment) -> AudioSegment:
        """Force audio segment to 16-bit PCM, stereo, 44.1kHz for safe merging"""
        if not segment:
            return AudioSegment.silent(duration=100)
        return segment.set_sample_width(2).set_channels(2).set_frame_rate(44100)

    def process_dialogue(self, dialogue_data, voice_assignments=None, output_type="chapter", project_name="project"):
        """Process the entire dialogue with TTS and sound effects"""
        audio_segments = []

        # Use custom voice assignments if provided
        if voice_assignments:
            working_voices = voice_assignments.copy()
        else:
            working_voices = self.character_voices

        # Create containers for progress tracking
        progress_container = st.container()

        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

        for i, entry in enumerate(dialogue_data):
            progress = (i + 1) / len(dialogue_data)
            progress_bar.progress(progress)

            if entry["type"] == "speech":
                char = entry.get("character", "Unknown")
                text_preview = entry["text"][:50] + \
                    ("..." if len(entry["text"]) > 50 else "")
                status_text.text(
                    f"🎤 Generating speech for {char}: {text_preview}")

                # Use custom voice assignment if available
                voice_id = working_voices.get(char, entry.get("voice_id"))

                # Generate speech with retry mechanism
                max_retries = 3
                audio = None
                for attempt in range(max_retries):
                    try:
                        audio = self.generate_speech(
                            voice_id=voice_id,
                            text=entry["text"],
                            model_id=entry.get("model_id", "eleven_v3")
                        )
                        if len(audio) > 0:  # Successfully generated audio
                            break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            st.warning(
                                f"Failed to generate speech after {max_retries} attempts: {e}")
                            audio = AudioSegment.silent(duration=1000)
                        else:
                            time.sleep(0.5)  # Wait before retry

                if audio:
                    audio_segments.append(audio)

            elif entry["type"] == "sound_effect":
                status_text.text(
                    f"🔊 Adding sound effect: {entry['effect_name']}")
                effect_audio = self.load_sound_effect(entry["effect_name"])
                audio_segments.append(effect_audio)

            elif entry["type"] == "pause":
                duration = entry.get("duration", 500)
                silence = AudioSegment.silent(duration=duration)
                audio_segments.append(silence)

            # Small delay to show progress and ensure proper processing
            time.sleep(0.1)

        status_text.text("🔄 Merging audio segments...")

        # Merge audio segments with error handling
        try:
            if audio_segments:
                final_audio = self._ensure_compatible(audio_segments[0])
                for segment in audio_segments[1:]:
                    if segment and len(segment) > 0:
                        final_audio = final_audio + \
                            self._ensure_compatible(segment)
            else:
                final_audio = AudioSegment.silent(duration=1000)
        except Exception as e:
            st.error(f"Error merging audio segments: {e}")
            final_audio = AudioSegment.silent(duration=1000)

        status_text.text("✨ Finalizing audio...")

        # After merging segments into final_audio:
        try:
            final_audio = self.apply_post_processing(final_audio)
        except Exception as e:
            st.warning(f"Skipping post-processing due to error: {e}")

        # Export to buffer with error handling
        try:
            audio_buffer = io.BytesIO()
            final_audio.export(audio_buffer, format="mp3", bitrate="128k")
            audio_data = audio_buffer.getvalue()

            # Save to organized folder structure
            self._save_to_organized_folder(
                audio_data, output_type, project_name)

        except Exception as e:
            st.error(f"Error exporting audio: {e}")
            # Create minimal audio file as fallback
            audio_buffer = io.BytesIO()
            AudioSegment.silent(duration=1000).export(
                audio_buffer, format="mp3")
            audio_data = audio_buffer.getvalue()

        progress_bar.progress(1.0)
        status_text.text("✅ Audio generation complete!")

        return audio_data

    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class DialogueParser:
    def __init__(self):
        self.character_voices = get_flat_character_voices()
        self.emotion_tags = get_flat_emotion_tags()

    def parse_dialogue(self, dialogue_text, voice_assignments=None):
        """Parse dialogue text in format: [Character] (emotion1)(emotion2): Text *sound_effect*"""
        lines = dialogue_text.strip().split('\n')
        dialogue_sequence = []

        # Use custom voice assignments if provided
        working_voices = voice_assignments if voice_assignments else self.character_voices

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            char_match = re.match(r'\[([^\]]+)\]\s*(\([^)]+\))*\s*:', line)
            if not char_match:
                st.warning(f"Line {line_num}: Invalid format - {line[:50]}...")
                continue

            character = char_match.group(1).strip()
            emotion_matches = re.findall(r'\(([^)]+)\)', char_match.group(0))
            text_part = re.sub(
                r'^\[([^\]]+)\]\s*(\([^)]+\))*\s*:', '', line).strip()

            # Extract sound effects and clean text
            sound_effects = re.findall(r'\*([^*]+)\*', text_part)
            clean_text = re.sub(r'\*[^*]+\*', '', text_part).strip()

            # Build emotion text
            emotion_text = ''.join(
                [f"[{emotion.strip()}]" for emotion in emotion_matches])
            final_text = f"{emotion_text} {clean_text}".strip()

            # Use working voice assignments
            if character not in working_voices:
                st.warning(
                    f"Line {line_num}: No voice assigned for character '{character}'")
                continue

            # Add the speech segment
            dialogue_sequence.append({
                "type": "speech",
                "voice_id": working_voices[character],
                "text": final_text,
                "model_id": "eleven_v3",
                "character": character,
                "line_number": line_num,
                "emotions": emotion_matches,
                "original_text": clean_text
            })

            for effect in sound_effects:
                dialogue_sequence.append({
                    "type": "pause",
                    "duration": 300
                })

                # Normalize effect name before saving
                norm_name = normalize_effect_name(effect.strip())
                dialogue_sequence.append({
                    "type": "sound_effect",
                    "effect_name": norm_name
                })

            # Add pause after each line (but not if it's the last line)
            # This ensures natural spacing between dialogue lines
            dialogue_sequence.append({
                "type": "pause",
                "duration": 300  # 0.3 seconds between dialogue lines
            })

        return dialogue_sequence


def create_teaser_generator_tab():
    """Create teaser generator interface for TikTok/Shorts content"""
    st.markdown("### 🎬 Teaser Line Generator")
    st.markdown(
        "Create **TikTok/Shorts-ready** teaser content (1-5 lines) for marketing purposes")

    col1, col2 = st.columns([3, 1])

    with col1:
        # Project name for teaser
        project_name = st.text_input(
            "Project/Book Name:",
            value=st.session_state.get('teaser_project_name', ''),
            placeholder="Enter book or project name...",
            key="teaser_project_name"
        )

    with col2:
        if st.button("🗑️ Reset Teaser", type="secondary", use_container_width=True, key="reset_teaser_btn"):
            # Clear teaser-related session state
            keys_to_clear = [
                'teaser_text', 'teaser_project_name'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Teaser tab reset!")
            st.rerun()

    # Handle emotion/effect additions from sidebar for teaser tab
    if 'emotion_to_add' in st.session_state and st.session_state.get('teaser_text', ''):
        st.session_state.teaser_text += f" {st.session_state.emotion_to_add}"
        del st.session_state.emotion_to_add
        st.rerun()

    if 'effect_to_add' in st.session_state and st.session_state.get('teaser_text', ''):
        st.session_state.teaser_text += f" {st.session_state.effect_to_add}"
        del st.session_state.effect_to_add
        st.rerun()

    # Teaser text input
    teaser_text = st.text_area(
        "Enter 1-5 teaser lines:",
        value=st.session_state.get('teaser_text', ''),
        height=200,
        placeholder="""[Dante] (whispers)(excited): The security system is down. This is our chance.
[Luca] (frustrated): I still don't like this plan, Dante.
[Rafael] (mischievously): Relax, tesoro. What could go wrong?""",
        help="Perfect for TikTok/Shorts! Keep it short and punchy (1-5 lines max)",
        key="teaser_text_input"
    )

    # Update session state
    if teaser_text != st.session_state.get('teaser_text', ''):
        st.session_state.teaser_text = teaser_text

    # Quick stats
    if teaser_text.strip():
        lines = [line.strip() for line in teaser_text.split(
            '\n') if line.strip() and not line.startswith('#')]
        line_count = len(lines)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Lines", line_count)
        with col2:
            color = "🟢" if line_count <= 5 else "🔴"
            st.metric("📱 TikTok Ready",
                      f"{color} {'Yes' if line_count <= 5 else 'Too long'}")
        with col3:
            est_duration = line_count * 3  # Rough estimate: 3 seconds per line
            st.metric("⏱️ Est. Duration", f"~{est_duration}s")

        if line_count > 5:
            st.warning(
                "⚠️ Recommended: Keep teasers to 5 lines or less for social media")
        elif line_count == 0:
            st.info("💡 Add some dialogue lines to generate a teaser")

    # Generate teaser button
    if st.button("🎬 Generate Teaser Audio", type="primary", use_container_width=True, key="generate_teaser_btn"):
        if not teaser_text.strip():
            st.error("Please enter some teaser dialogue!")
        elif not project_name.strip():
            st.error("Please enter a project name!")
        else:
            try:
                # Initialize generator and parser
                generator = DialogueAudioGenerator()
                parser = DialogueParser()

                # Parse dialogue
                dialogue_sequence = parser.parse_dialogue(teaser_text)

                if not dialogue_sequence:
                    st.error("No valid dialogue found!")
                else:
                    # Generate audio
                    with st.spinner("Generating teaser audio..."):
                        audio_data = generator.process_dialogue(
                            dialogue_sequence,
                            output_type="teaser",
                            project_name=project_name
                        )

                    # Display success
                    st.success("✅ Teaser audio generated!")

                    # Audio player
                    st.audio(audio_data, format="audio/mp3")

                    # Download button
                    filename = f"{project_name}_teaser.mp3"
                    st.markdown(
                        get_audio_download_link(audio_data, filename),
                        unsafe_allow_html=True
                    )

                    # Cleanup
                    generator.cleanup()

            except Exception as e:
                st.error(f"Error generating teaser: {e}")


def create_emotion_preview_tab():
    """Create emotion preview interface for testing voices with emotions"""
    st.markdown("### 😊 Emotion Preview")
    st.markdown("Test how different voices sound with various emotions")

    # Voice selection
    all_voices = {}
    for category, chars in CHARACTER_VOICES.items():
        for char, vid in chars.items():
            all_voices[char] = {
                "voice_id": vid,
                "character": char
            }

    col1, col2 = st.columns(2)

    with col1:
        selected_voice_label = st.selectbox(
            "Select Voice to Test:",
            options=list(all_voices.keys()),
            key="emotion_preview_voice"
        )

        selected_voice = all_voices[selected_voice_label]

        # Show voice info
        st.info(f"**{selected_voice['character']}**")

    with col2:
        # Emotion selection
        emotion_options = ["None"] + list(get_flat_emotion_tags().keys())
        selected_emotion = st.selectbox(
            "Select Emotion to Test:",
            options=emotion_options,
            key="emotion_preview_emotion"
        )

    # Test text input
    test_text = st.text_input(
        "Test Text:",
        value="Hello there, this is a voice test with the selected emotion.",
        key="emotion_preview_text"
    )

    # Build preview text with emotion
    if selected_emotion != "None":
        preview_text = f"[{selected_emotion}] {test_text}"
    else:
        preview_text = test_text

    st.text(f"Preview: {preview_text}")

    # Generate preview button
    if st.button("🎤 Generate Voice Preview", type="primary", use_container_width=True, key="generate_emotion_preview"):
        if not test_text.strip():
            st.error("Please enter some test text!")
        else:
            try:
                # Initialize generator
                generator = DialogueAudioGenerator()

                # Generate audio
                with st.spinner("Generating voice preview..."):
                    audio = generator.generate_speech(
                        voice_id=selected_voice["voice_id"],
                        text=preview_text
                    )

                    # Convert to bytes
                    audio_buffer = io.BytesIO()
                    audio.export(audio_buffer, format="mp3", bitrate="128k")
                    audio_data = audio_buffer.getvalue()

                    # Save to voice tests folder
                    generator._save_to_organized_folder(
                        audio_data,
                        "voice_test",
                        f"{selected_voice['character']}_{selected_emotion}"
                    )

                st.success("✅ Voice preview generated!")

                # Audio player
                st.audio(audio_data, format="audio/mp3")

                # Download button
                filename = f"{selected_voice['character']}_{selected_emotion}_preview.mp3"
                st.markdown(
                    get_audio_download_link(audio_data, filename),
                    unsafe_allow_html=True
                )

                # Cleanup
                generator.cleanup()

            except Exception as e:
                st.error(f"Error generating preview: {e}")

    # Emotion reference guide
    with st.expander("📚 Available Emotions Reference"):
        for category, emotions in EMOTION_TAGS.items():
            st.markdown(f"**{category.replace('_', ' ').title()}:**")
            emotion_list = list(emotions.keys())
            cols = st.columns(3)
            for i, emotion in enumerate(emotion_list):
                with cols[i % 3]:
                    st.markdown(f"• {emotion}")
            st.divider()


def create_voice_manager_tab():
    """Create comprehensive voice manager interface"""
    st.markdown("### 🎭 Voice Manager")
    st.markdown("Assign AI voices to characters and save mappings for projects")

    # Project management section
    st.markdown("#### 📁 Project Management")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        project_name = st.text_input(
            "Project Name:",
            value=st.session_state.get('vm_project_name', ''),
            placeholder="Enter project/book name...",
            key="vm_project_name"
        )

    with col2:
        # Load existing project
        saved_projects = get_saved_voice_projects()
        if saved_projects:
            selected_project = st.selectbox(
                "Load Saved Project:",
                options=[""] + saved_projects,
                key="vm_load_project"
            )

            if st.button("📂 Load", type="secondary", use_container_width=True):
                if selected_project:
                    loaded_mappings = load_voice_mappings(selected_project)
                    if loaded_mappings:
                        st.session_state.vm_voice_mappings = loaded_mappings
                        st.session_state.vm_project_name = selected_project
                        st.success(f"✅ Loaded project: {selected_project}")
                        st.rerun()

    with col3:
        # Save current project
        if st.button("💾 Save Project", type="secondary", use_container_width=True):
            if project_name.strip() and 'vm_voice_mappings' in st.session_state:
                save_file = save_voice_mappings(
                    st.session_state.vm_voice_mappings,
                    project_name
                )
                st.success(f"✅ Project saved: {save_file.name}")
            else:
                st.error(
                    "Please enter project name and create voice mappings first!")

    st.markdown("---")

    # Character management
    st.markdown("#### 👥 Character Voice Assignments")

    # Initialize voice mappings if not exists
    if 'vm_voice_mappings' not in st.session_state:
        st.session_state.vm_voice_mappings = get_flat_character_voices().copy()

    # Add new character section
    with st.expander("➕ Add New Character", expanded=False):
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            new_char_name = st.text_input(
                "Character Name:",
                placeholder="Enter new character name...",
                key="vm_new_char_name"
            )

        with col2:
            # All available voices for dropdown
            all_voice_options = {}
            for category, chars in CHARACTER_VOICES.items():
                for char, vid in chars.items():
                    all_voice_options[char] = vid

            selected_voice_for_new = st.selectbox(
                "Assign Voice:",
                options=list(all_voice_options.keys()),
                key="vm_new_char_voice"
            )

        with col3:
            if st.button("➕ Add", type="secondary", use_container_width=True, key="vm_add_char"):
                if new_char_name.strip():
                    voice_id = all_voice_options[selected_voice_for_new]
                    st.session_state.vm_voice_mappings[new_char_name.strip(
                    )] = voice_id
                    st.success(f"✅ Added {new_char_name} with voice!")
                    st.rerun()
                else:
                    st.error("Please enter a character name!")

    # Display current character assignments
    if st.session_state.vm_voice_mappings:
        st.markdown("#### 🎯 Current Assignments")

        # Get all available voices for dropdowns
        all_voice_options = {}
        for category, chars in CHARACTER_VOICES.items():
            for char, vid in chars.items():
                all_voice_options[char] = vid

        # Create assignment interface
        characters_to_remove = []

        for char_name, current_voice_id in st.session_state.vm_voice_mappings.items():
            col1, col2, col3 = st.columns([2, 3, 1])

            with col1:
                st.markdown(f"**{char_name}**")

            with col2:
                # Find current selection
                current_selection = None
                for label, voice_id in all_voice_options.items():
                    if voice_id == current_voice_id:
                        current_selection = label
                        break

                if current_selection is None:
                    current_selection = list(all_voice_options.keys())[0]

                # Voice selection dropdown
                new_voice_selection = st.selectbox(
                    f"Voice for {char_name}",
                    options=list(all_voice_options.keys()),
                    index=list(all_voice_options.keys()).index(
                        current_selection),
                    key=f"vm_voice_select_{char_name}",
                    label_visibility="collapsed"
                )

                # Update if changed
                new_voice_id = all_voice_options[new_voice_selection]
                if new_voice_id != current_voice_id:
                    st.session_state.vm_voice_mappings[char_name] = new_voice_id
                    st.rerun()

            with col3:
                if st.button("🗑️", key=f"vm_remove_{char_name}", help=f"Remove {char_name}"):
                    characters_to_remove.append(char_name)

        # Remove characters if requested
        for char_to_remove in characters_to_remove:
            del st.session_state.vm_voice_mappings[char_to_remove]
            st.rerun()

    else:
        st.info(
            "No character voice assignments yet. Add characters above or load a saved project.")

    # Available voices reference
    st.markdown("---")
    st.markdown("#### 🎤 Available Voices")

    for category, characters in CHARACTER_VOICES.items():
        with st.expander(f"{category} ({len(characters)} voices)"):
            for char, vid in characters.items():
                col1, col2, col3 = st.columns([2, 3, 1])

                with col1:
                    st.markdown(f"**{char}**")

                with col2:
                    st.code(vid, language=None)

                with col3:
                    # Quick test button
                    if st.button("🔊 Test", key=f"vm_test_{char}", help=f"Quick voice test for {char}"):
                        try:
                            generator = DialogueAudioGenerator()
                            test_audio = generator.generate_speech(
                                voice_id=vid,
                                text=f"Hello, I am {char}. This is how my voice sounds."
                            )
                            audio_buffer = io.BytesIO()
                            test_audio.export(audio_buffer, format="mp3")
                            audio_data = audio_buffer.getvalue()
                            st.audio(audio_data, format="audio/mp3")
                            generator.cleanup()
                        except Exception as e:
                            st.error(f"Error testing voice: {e}")


def create_raw_parser_tab(get_known_characters_callable):
    import streamlit as st

    st.markdown("### 📚 Raw Text → Dialogue Parser")
    st.markdown(
        "Paste raw book text below. The parser will detect quotes, infer speakers from narration like _\"…\" said Dante_, assign basic emotions (e.g., whispered → (whispers)), and optionally add narration lines as [Narrator]."
    )

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        include_narration = st.checkbox(
            "Include Narration as [Narrator]", value=True, key="raw_inc_narr")
    with col2:
        attach_fx = st.checkbox(
            "Detect FX from narration (gasp/laugh/etc.)", value=True, key="raw_attach_fx")
    with col3:
        use_saved_characters = st.checkbox(
            "Use current voice-mapped characters", value=True, key="raw_use_saved_chars")

    raw_text = st.text_area(
        "Raw Prose:",
        height=280,
        placeholder=(
            "Example:\n"
            "Dante’s eyes narrowed. \"The security system is down,\" he whispered. \"This is our chance.\"\n"
            "Luca sighed. \"I still don't like this plan, Dante.\"\n"
            "\"Relax, tesoro. What could go wrong?\" Rafael said mischievously.\n"
            "Nikolai said coldly, \"Everything. That’s what experience teaches you.\"\n"
            "There was a sharp gasp as the door slammed."
        ),
        key="raw_parser_input",
    )

    # Helper to compute known characters once
    known = []
    if use_saved_characters:
        try:
            known = list(get_known_characters_callable())
        except Exception:
            known = []

    # --- Convert action: store result in session, then rerun so the "Send" button can exist on the next run
    if st.button("🔍 Convert Raw → Dialogue", type="primary", use_container_width=True, key="raw_convert_btn"):
        if not raw_text.strip():
            st.error("Please paste some raw prose first.")
        else:
            parser = RawProseParser(
                known_characters=known,
                include_narration=include_narration,
                attach_detected_fx=attach_fx,
            )
            result = parser.convert(raw_text)

            st.session_state["raw_last_formatted_text"] = result.formatted_text
            st.session_state["raw_last_dialogues"] = result.dialogues
            st.session_state["raw_last_stats"] = result.stats
            st.session_state["raw_parsed_ready"] = True

            # Important: rerun so the Send button exists *outside* this branch
            st.rerun()

    # --- Results area: rendered whenever we have a parsed result in session
    if st.session_state.get("raw_parsed_ready") and st.session_state.get("raw_last_formatted_text"):
        st.success("✅ Parsed successfully.")

        stats = st.session_state.get("raw_last_stats", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Quotes",   stats.get("quotes_found", 0))
        with c2:
            st.metric("Lines",    stats.get("lines_emitted", 0))
        with c3:
            st.metric("From after", stats.get("speaker_from_after", 0))
        with c4:
            st.metric("From before", stats.get("speaker_from_before", 0))
        with c5:
            st.metric("Narration", stats.get("narration_blocks", 0))

        st.markdown("#### ▶ Standardized Output")
        st.code(
            st.session_state["raw_last_formatted_text"], language="markdown")

        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("→ Send to Main Generator", key="raw_send_to_main", type="primary", use_container_width=True):
                # 1) Hand the parsed text to the Main tab
                st.session_state.dialogue_text = st.session_state["raw_last_formatted_text"]
                # 2) Clear Main analysis so user re-parses there (optional)
                for k in ("paste_text_analysis", "paste_formatted_dialogue", "paste_parsed_dialogues", "paste_voice_assignments"):
                    st.session_state.pop(k, None)
                # 3) Switch tabs and rerun
                st.session_state.current_tab = "main"
                st.rerun()

        with colB:
            if st.button("🗑 Reset Parsed Output", key="raw_reset", type="secondary", use_container_width=True):
                for k in ("raw_last_formatted_text", "raw_last_dialogues", "raw_last_stats", "raw_parsed_ready"):
                    st.session_state.pop(k, None)
                st.rerun()


def create_navigation_sidebar():
    """Create enhanced sidebar with navigation and resources"""
    with st.sidebar:
        # User info
        if st.session_state.get('authenticated'):
            user_type = st.session_state.get('user_type', 'unknown')
            st.success(f"✅ Logged in as: **{user_type.title()}**")

            if st.button("🚪 Logout", type="secondary", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.user_type = None
                st.rerun()

            st.markdown("---")

        # Navigation buttons
        st.markdown("### 🧭 Navigation")

        # Initialize current tab if not set
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = "main"

        nav_buttons = [
            ("main", "📝 Main Generator", "Main dialogue generation"),
            ("teaser", "🎬 Teaser Generator", "TikTok/Shorts teasers"),
            ("emotion", "😊 Emotion Preview", "Test voices with emotions"),
            ("voice_manager", "🎭 Voice Manager", "Manage character voices"),
            ("raw", "📚 Raw Parser", "Convert raw prose to dialogue")
        ]

        for tab_key, tab_label, tab_help in nav_buttons:
            if st.button(
                tab_label,
                key=f"nav_{tab_key}",
                use_container_width=True,
                type="primary" if st.session_state.current_tab == tab_key else "secondary",
                help=tab_help
            ):
                st.session_state.current_tab = tab_key
                st.rerun()

        st.markdown("---")

        # Resources section
        st.title("🎭 Resources")

        # Emotion Tags Accordion
        with st.expander("😊 Emotion Tags", expanded=False):
            for category, emotions in EMOTION_TAGS.items():
                st.markdown(f"**{category.replace('_', ' ').title()}:**")
                cols = st.columns(2)
                for i, (emotion, tag) in enumerate(emotions.items()):
                    with cols[i % 2]:
                        if st.button(f"({emotion})", key=f"emotion_{emotion}", use_container_width=True):
                            # Store emotion to add to the current active tab
                            st.session_state.emotion_to_add = f"({emotion})"
                st.divider()

        # Characters Accordion
        with st.expander("👥 Available Characters", expanded=False):
            for category, characters in CHARACTER_VOICES.items():
                st.markdown(f"**{category}**")

                for char, vid in characters.items():
                    with st.container():
                        st.markdown(f"**{char}**")
                        st.code(vid, language=None)
                st.divider()

        # Sound Effects Accordion
        with st.expander("🔊 Sound Effects", expanded=False):
            st.markdown("**Available in fx_library:**")
            cols = st.columns(1)
            for effect, filename in SOUND_EFFECTS.items():
                if st.button(f"*{effect}*", key=f"fx_{effect}", use_container_width=True):
                    # Store effect to add to the current active tab
                    st.session_state.effect_to_add = f"*{effect}*"
                st.caption(f"📁 {filename}")

        # Output folders info
        with st.expander("📁 Output Folders", expanded=False):
            st.markdown("""
            **Organized Output Structure:**
            - 📁 `audio_output/teasers/` - TikTok/Shorts clips
            - 📁 `audio_output/chapters/` - Full chapters
            - 📁 `audio_output/voice_tests/` - Voice previews
            - 📁 `audio_output/books/` - Complete books
            - 📁 `voice_mappings/` - Saved voice assignments
            """)


def display_analysis_results(analysis: ParseAnalysis):
    """Display comprehensive analysis results"""

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📊 Total Lines", analysis.total_lines)
        st.metric("💬 Dialogue Lines", analysis.dialogue_lines)

    with col2:
        st.metric("👥 Characters Found", len(analysis.characters_found))
        st.metric("😊 Emotions Used", len(analysis.emotions_found))

    with col3:
        st.metric("🔊 Sound Effects", len(analysis.sound_effects_found))
        st.metric("⚠️ Issues Found",
                  len(analysis.unsupported_characters) +
                  len(analysis.unsupported_emotions) +
                  len(analysis.unsupported_sound_effects))

    # Detailed breakdowns
    col1, col2 = st.columns(2)

    with col1:
        if analysis.characters_found:
            st.markdown("### 👥 Characters Usage")
            for char, count in sorted(analysis.characters_found.items(), key=lambda x: x[1], reverse=True):
                status = "✅" if char in get_flat_character_voices() else "❌"
                st.markdown(f"{status} **{char}**: {count} lines")

        if analysis.emotions_found:
            st.markdown("### 😊 Emotions Usage")
            emotion_tags = get_flat_emotion_tags()
            for emotion, count in sorted(analysis.emotions_found.items(), key=lambda x: x[1], reverse=True):
                status = "✅" if emotion in emotion_tags else "❌"
                st.markdown(f"{status} **{emotion}**: {count} times")

    with col2:
        if analysis.sound_effects_found:
            st.markdown("### 🔊 Sound Effects Usage")
            for effect, count in sorted(analysis.sound_effects_found.items(), key=lambda x: x[1], reverse=True):
                norm_effect = normalize_effect_name(effect)
                status = "✅" if norm_effect in SOUND_EFFECTS else "❌"
                st.markdown(f"{status} **{effect}**: {count} times")

    # Issues section
    if (analysis.unsupported_characters or
        analysis.unsupported_emotions or
            analysis.unsupported_sound_effects):

        st.markdown("### ⚠️ Issues Found")

        if analysis.unsupported_characters:
            st.error("**Unsupported Characters:**")
            for char in analysis.unsupported_characters:
                st.markdown(f"❌ {char}")

        if analysis.unsupported_emotions:
            st.warning("**Unsupported Emotions:**")
            for emotion in analysis.unsupported_emotions:
                st.markdown(f"⚠️ {emotion}")

        if analysis.unsupported_sound_effects:
            st.warning("**Unsupported Sound Effects:**")
            for effect in analysis.unsupported_sound_effects:
                st.markdown(f"⚠️ {effect}")


def create_voice_management_interface(analysis: ParseAnalysis, tab_prefix: str):
    """Create interface for managing character voice assignments with tab-specific keys"""

    if not analysis.characters_found:
        st.info("No characters found in the text to assign voices to.")
        return None

    st.markdown("### 🎭 Character Voice Management")

    # Initialize voice manager for this tab
    voice_manager_key = f'{tab_prefix}_voice_manager'
    if voice_manager_key not in st.session_state:
        st.session_state[voice_manager_key] = VoiceManager()

    voice_manager = st.session_state[voice_manager_key]

    # Get all available voices for dropdown
    all_voices = {}
    for category, chars in CHARACTER_VOICES.items():
        for char, vid in chars.items():
            all_voices[char] = vid

    voice_assignments_changed = False

    # Create voice assignment interface
    for character in sorted(analysis.characters_found.keys()):
        col1, col2, col3 = st.columns([2, 3, 1])

        with col1:
            usage_count = analysis.characters_found[character]
            if character in get_flat_character_voices():
                st.markdown(f"✅ **{character}** ({usage_count} lines)")
            else:
                st.markdown(f"❌ **{character}** ({usage_count} lines)")

        with col2:
            current_voice = voice_manager.get_voice_for_character(character)

            # Find current selection for dropdown
            current_selection = None
            for label, voice_id in all_voices.items():
                if voice_id == current_voice:
                    current_selection = label
                    break

            # Voice selection dropdown with tab-specific key
            selected_voice = st.selectbox(
                f"Voice for {character}",
                options=list(all_voices.keys()),
                index=list(all_voices.keys()).index(
                    current_selection) if current_selection else 0,
                key=f"{tab_prefix}_voice_select_{character}",
                label_visibility="collapsed"
            )

            # Update voice assignment if changed
            new_voice_id = all_voices[selected_voice]
            if new_voice_id != current_voice:
                voice_manager.assign_voice(character, new_voice_id)
                voice_assignments_changed = True

        with col3:
            st.caption(f"Lines: {usage_count}")

    if voice_assignments_changed:
        st.rerun()

    return voice_manager.voice_assignments


def create_file_upload_interface():
    """Create interface for file upload processing"""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### 📄 Upload Text File")

    with col2:
        # Reset button for upload tab
        if st.button("🗑️ Reset Upload", type="secondary", use_container_width=True, key="reset_upload_btn"):
            # Clear all upload-related session state
            keys_to_clear = [
                'upload_dialogue_text',
                'upload_text_analysis',
                'upload_formatted_dialogue',
                'upload_parsed_dialogues',
                'upload_voice_assignments',
                'upload_voice_manager'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]

            st.session_state.upload_dialogue_text = ""
            st.success("Upload tab reset successfully!")
            st.rerun()

    uploaded_file = st.file_uploader(
        "Upload a text file:",
        type=['txt', 'docx', 'pdf'],
        help="Upload a .txt, .docx, or .pdf file containing dialogue text",
        key="file_uploader"
    )

    if uploaded_file:
        try:
            # Show processing status
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Read file content as bytes
                file_content = uploaded_file.read()

                # Extract text based on file type
                extractor = FileExtractor()

                if uploaded_file.type == "text/plain" or uploaded_file.name.endswith('.txt'):
                    content = extractor.extract_from_txt(file_content)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or uploaded_file.name.endswith('.docx'):
                    content = extractor.extract_from_docx(file_content)
                elif uploaded_file.type == "application/pdf" or uploaded_file.name.endswith('.pdf'):
                    content = extractor.extract_from_pdf(file_content)
                else:
                    # Fallback to text extraction
                    content = extractor.extract_from_txt(file_content)

            if not content or not content.strip():
                st.error("No text content found in the uploaded file.")
                return

            # Limit content size for performance
            if len(content) > 100000:  # 100k characters limit
                st.warning(
                    "⚠️ File is very large. Truncating to first 100,000 characters for performance.")
                content = content[:100000] + \
                    "\n\n# [Content truncated for performance]"

            word_count = len(content.split())
            st.success(f"✅ **File loaded:** {uploaded_file.name}")
            st.info(f"📊 **Word count:** {word_count:,} words")

            if word_count > 5000:
                st.warning(
                    "⚠️ Large file detected (>5k words). Processing may take longer.")
            elif word_count < 10:
                st.warning(
                    "⚠️ Very short file (<10 words). Please check file content.")

            # Preview first few lines
            lines = [line for line in content.split('\n')[:15] if line.strip()]
            if lines:
                with st.expander("👀 File Preview (first 15 non-empty lines)"):
                    for i, line in enumerate(lines, 1):
                        st.markdown(
                            f"**{i}.** {line[:120]}{'...' if len(line) > 120 else ''}")

            # Auto-load text into upload tab input field
            if 'upload_dialogue_text' not in st.session_state or not st.session_state.upload_dialogue_text:
                st.session_state.upload_dialogue_text = content
                st.success(
                    "File content automatically loaded into text input field below!")
            else:
                # Ask user if they want to replace existing content
                if st.button("🔄 Replace current text with uploaded file", type="secondary"):
                    st.session_state.upload_dialogue_text = content
                    st.success("Text replaced with uploaded file content!")
                    st.rerun()

        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.error(
                "Please ensure the file is a valid text, Word, or PDF document.")


def generate_audio_for_tab(text_content: str, tab_prefix: str, analysis_key: str, voice_assignments_key: str, output_type: str = "chapter", project_name: str = "project"):
    """Generate audio for a specific tab's content"""
    if not text_content.strip():
        st.error("Please enter some dialogue text!")
        return

    try:
        # Get analysis and voice assignments for this tab
        current_analysis = st.session_state.get(analysis_key)
        current_voice_assignments = st.session_state.get(voice_assignments_key)

        # Initialize generator and parser
        generator = DialogueAudioGenerator()
        parser = DialogueParser()

        # Parse dialogue
        dialogue_sequence = parser.parse_dialogue(
            text_content,
            current_voice_assignments
        )

        if not dialogue_sequence:
            st.error("No valid dialogue found!")
            return

        # Display summary before generation
        speech_count = sum(
            1 for entry in dialogue_sequence if entry["type"] == "speech")
        effect_count = sum(
            1 for entry in dialogue_sequence if entry["type"] == "sound_effect")
        st.info(
            f"📊 **Processing:** {speech_count} speech segments, {effect_count} sound effects")

        # Generate audio with progress tracking
        with st.spinner("Generating audio..."):
            audio_data = generator.process_dialogue(
                dialogue_sequence,
                current_voice_assignments,
                output_type=output_type,
                project_name=project_name
            )

        # Display success and download link
        st.success("✅ Audio generated successfully!")

        # Audio player
        st.audio(audio_data, format="audio/mp3")

        # Download button
        filename = f"{project_name}_{output_type}_audio.mp3"
        st.markdown(
            get_audio_download_link(audio_data, filename),
            unsafe_allow_html=True
        )

        # Cleanup
        generator.cleanup()

    except Exception as e:
        st.error(f"Error generating audio: {e}")
        st.error("Check your network connection and file paths.")


def get_audio_download_link(audio_data, filename="dialogue.mp3"):
    """Generate a download link for audio data"""
    b64 = base64.b64encode(audio_data).decode()
    href = f'''
    <a href="data:audio/mp3;base64,{b64}" download="{filename}" 
       style="background: linear-gradient(45deg, #ff6b6b, #4ecdc4); 
              color: white; 
              padding: 12px 24px; 
              text-decoration: none; 
              border-radius: 8px; 
              display: inline-block; 
              margin: 10px 0;
              font-weight: bold;
              box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
       📥 Download Audio File
    </a>'''
    return href


def create_main_generator_content():
    """Create the main dialogue generator interface"""
    st.markdown('<h1 class="main-header">🎭 Audiobook machine</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Generate realistic dialogue audio with TTS and sound effects!</p>',
                unsafe_allow_html=True)

    # Initialize session state with example text but no analysis
    if 'dialogue_text' not in st.session_state:
        st.session_state.dialogue_text = """[Brad] (whispers)(excited): The security system is down. This is our chance. *apartmentcreaks*
[Arabella] (sighs)(frustrated): I still don't like this plan, Dante. *gasps*
[Grandpa Spuds Oxley] (mischievously): Relax, tesoro. What could go wrong? *laughs*
[Christian] (cold)(calm): Everything. That's what experience teaches you. *growls*"""

    # Initialize upload dialogue text separately
    if 'upload_dialogue_text' not in st.session_state:
        st.session_state.upload_dialogue_text = ""

    # Create tabs for text input methods
    tab1, tab2 = st.tabs(["📝 Paste Text", "📄 Upload File"])

    with tab1:
        st.markdown("### 📝 Paste Dialogue Text")

        # Project name input
        col1, col2 = st.columns([3, 1])
        with col1:
            project_name = st.text_input(
                "Project/Chapter Name:",
                value=st.session_state.get('main_project_name', ''),
                placeholder="Enter project or chapter name...",
                key="main_project_name"
            )

        # Handle emotion/effect additions from sidebar for paste tab
        if 'emotion_to_add' in st.session_state:
            st.session_state.dialogue_text += f" {st.session_state.emotion_to_add}"
            del st.session_state.emotion_to_add
            st.rerun()

        if 'effect_to_add' in st.session_state:
            st.session_state.dialogue_text += f" {st.session_state.effect_to_add}"
            del st.session_state.effect_to_add
            st.rerun()

        paste_text = st.text_area(
            "Paste your dialogue text here:",
            value=st.session_state.dialogue_text,
            height=400,
            placeholder="""Supported formats:
[Character] (emotion): Dialogue text *sound_effect*
Character: Dialogue text
"Dialogue text," said Character.
Character said, "Dialogue text."

Or paste raw text from books/stories...""",
            key="paste_text_input"
        )

        # Update session state when text changes
        if paste_text != st.session_state.dialogue_text:
            st.session_state.dialogue_text = paste_text

        # Parse button for paste tab
        if st.button("🔍 Parse & Analyze Text", type="secondary", use_container_width=True, key="paste_parse_btn"):
            if not st.session_state.dialogue_text.strip():
                st.error("Please enter some text to parse!")
            else:
                # Initialize parser
                parser = TextParser()

                # Analyze the text
                with st.spinner("Analyzing text..."):
                    analysis = parser.analyze_text(
                        st.session_state.dialogue_text)

                # Store analysis in session state
                st.session_state.paste_text_analysis = analysis

                # Parse and format the text
                formatted_text, parsed_dialogues = parser.parse_to_dialogue_format(
                    st.session_state.dialogue_text)
                st.session_state.paste_formatted_dialogue = formatted_text
                st.session_state.paste_parsed_dialogues = parsed_dialogues

                st.success("✅ Text analysis complete!")
                st.rerun()

        # Display analysis results for paste tab
        if 'paste_text_analysis' in st.session_state:
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")

            display_analysis_results(st.session_state.paste_text_analysis)

            # Voice management interface for paste tab
            st.markdown("---")
            voice_assignments = create_voice_management_interface(
                st.session_state.paste_text_analysis,
                "paste"
            )
            if voice_assignments:
                st.session_state.paste_voice_assignments = voice_assignments

        # Generate Audio Button for Paste Tab
        st.markdown("---")
        st.markdown("### 🎬 Generate Audio")

        # Show metrics for paste tab
        if 'paste_text_analysis' in st.session_state:
            analysis = st.session_state.paste_text_analysis
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Characters", len(analysis.characters_found))
            with col2:
                st.metric("Dialogue Lines", analysis.dialogue_lines)
            with col3:
                st.metric("Emotions", len(analysis.emotions_found))
            with col4:
                st.metric("Sound Effects", len(analysis.sound_effects_found))

            # Show issues if any
            total_issues = (len(analysis.unsupported_characters) +
                            len(analysis.unsupported_emotions) +
                            len(analysis.unsupported_sound_effects))

            if total_issues > 0:
                with st.expander("⚠️ Issues to Review", expanded=False):
                    if analysis.unsupported_characters:
                        st.error(
                            f"**Unsupported Characters:** {', '.join(analysis.unsupported_characters)}")
                    if analysis.unsupported_emotions:
                        st.warning(
                            f"**Unsupported Emotions:** {', '.join(analysis.unsupported_emotions)}")
                    if analysis.unsupported_sound_effects:
                        st.warning(
                            f"**Unsupported Sound Effects:** {', '.join(analysis.unsupported_sound_effects)}")

        if st.button("🎬 Generate Audio", type="primary", use_container_width=True, key="paste_generate_btn"):
            if not project_name.strip():
                st.error("Please enter a project name!")
            else:
                generate_audio_for_tab(
                    st.session_state.dialogue_text,
                    "paste",
                    "paste_text_analysis",
                    "paste_voice_assignments",
                    output_type="chapter",
                    project_name=project_name
                )

        # Format guide
        with st.expander("📋 Format Guide & Examples"):
            st.markdown("""
            **Primary Format:** `[Character] (emotion1)(emotion2): Dialog text *sound_effect*`
            
            **Examples:**
            ```
            [Dante] (excited): Hello there!
            [Luca] (whispers)(nervous): Are you sure? *gasps*
            [Aria] (sarcastic): Oh really? *snarls*
            [Rafael] (laughs): That's hilarious! *automatic*
            ```
            
            **Also Supports:**
            - Simple format: `Character: Dialog text`
            - Narrative: `"Dialog text," said Character.`
            - Book format: Raw text with quotes and character names
            
            **Tips:**
            - Use multiple emotions: `(excited)(mischievous)`
            - Sound effects are optional: `*growls*`
            - Comments start with `#`
            - Empty lines are ignored
            - Click buttons in the sidebar to quickly add emotions and effects
            """)

    with tab2:
        create_file_upload_interface()

        # Text input field for upload tab
        st.markdown("### ✏️ Edit Uploaded Text")

        # Project name for upload
        col1, col2 = st.columns([3, 1])
        with col1:
            upload_project_name = st.text_input(
                "Project/Chapter Name:",
                value=st.session_state.get('upload_project_name', ''),
                placeholder="Enter project or chapter name...",
                key="upload_project_name"
            )

        # Handle emotion/effect additions from sidebar for upload tab
        if 'emotion_to_add' in st.session_state and st.session_state.upload_dialogue_text:
            st.session_state.upload_dialogue_text += f" {st.session_state.emotion_to_add}"
            del st.session_state.emotion_to_add
            st.rerun()

        if 'effect_to_add' in st.session_state and st.session_state.upload_dialogue_text:
            st.session_state.upload_dialogue_text += f" {st.session_state.effect_to_add}"
            del st.session_state.effect_to_add
            st.rerun()

        upload_text = st.text_area(
            "Edit the uploaded text here:",
            value=st.session_state.upload_dialogue_text,
            height=400,
            placeholder="Upload a file above to see its content here...",
            key="upload_text_input"
        )

        # Update session state when upload text changes
        if upload_text != st.session_state.upload_dialogue_text:
            st.session_state.upload_dialogue_text = upload_text

        # Parse button for upload tab
        if st.button("🔍 Parse & Analyze Text", type="secondary", use_container_width=True, key="upload_parse_btn"):
            if not st.session_state.upload_dialogue_text.strip():
                st.error("Please enter some text to parse!")
            else:
                # Initialize parser
                parser = TextParser()

                # Analyze the text
                with st.spinner("Analyzing text..."):
                    analysis = parser.analyze_text(
                        st.session_state.upload_dialogue_text)

                # Store analysis in session state
                st.session_state.upload_text_analysis = analysis

                # Parse and format the text
                formatted_text, parsed_dialogues = parser.parse_to_dialogue_format(
                    st.session_state.upload_dialogue_text)
                st.session_state.upload_formatted_dialogue = formatted_text
                st.session_state.upload_parsed_dialogues = parsed_dialogues

                st.success("✅ Text analysis complete!")

        # Display analysis results for upload tab
        if 'upload_text_analysis' in st.session_state:
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")

            display_analysis_results(st.session_state.upload_text_analysis)

            # Voice management interface for upload tab
            st.markdown("---")
            voice_assignments = create_voice_management_interface(
                st.session_state.upload_text_analysis,
                "upload"
            )
            if voice_assignments:
                st.session_state.upload_voice_assignments = voice_assignments

        # Generate Audio Button for Upload Tab
        st.markdown("---")
        st.markdown("### 🎬 Generate Audio")

        # Show metrics for upload tab
        if 'upload_text_analysis' in st.session_state:
            analysis = st.session_state.upload_text_analysis
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Characters", len(analysis.characters_found))
            with col2:
                st.metric("Dialogue Lines", analysis.dialogue_lines)
            with col3:
                st.metric("Emotions", len(analysis.emotions_found))
            with col4:
                st.metric("Sound Effects", len(analysis.sound_effects_found))

            # Show issues if any
            total_issues = (len(analysis.unsupported_characters) +
                            len(analysis.unsupported_emotions) +
                            len(analysis.unsupported_sound_effects))

            if total_issues > 0:
                with st.expander("⚠️ Issues to Review", expanded=False):
                    if analysis.unsupported_characters:
                        st.error(
                            f"**Unsupported Characters:** {', '.join(analysis.unsupported_characters)}")
                    if analysis.unsupported_emotions:
                        st.warning(
                            f"**Unsupported Emotions:** {', '.join(analysis.unsupported_emotions)}")
                    if analysis.unsupported_sound_effects:
                        st.warning(
                            f"**Unsupported Sound Effects:** {', '.join(analysis.unsupported_sound_effects)}")
        else:
            # Show zero metrics when not analyzed
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Characters", 0)
            with col2:
                st.metric("Dialogue Lines", 0)
            with col3:
                st.metric("Emotions", 0)
            with col4:
                st.metric("Sound Effects", 0)

            if st.session_state.upload_dialogue_text.strip():
                st.info(
                    "Click 'Parse & Analyze Text' above to see detailed metrics.")

        if st.button("🎬 Generate Audio", type="primary", use_container_width=True, key="upload_generate_btn"):
            if not upload_project_name.strip():
                st.error("Please enter a project name!")
            else:
                generate_audio_for_tab(
                    st.session_state.upload_dialogue_text,
                    "upload",
                    "upload_text_analysis",
                    "upload_voice_assignments",
                    output_type="chapter",
                    project_name=upload_project_name
                )


def main():
    st.set_page_config(
        page_title="Audiobook machine",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .analysis-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .nav-button-active {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Check password authentication
    if not check_password():
        return

    # Create enhanced sidebar with navigation
    create_navigation_sidebar()

    # Main content area - route based on current tab
    current_tab = st.session_state.get('current_tab', 'main')

    if current_tab == "main":
        create_main_generator_content()

    elif current_tab == "teaser":
        create_teaser_generator_tab()

    elif current_tab == "emotion":
        create_emotion_preview_tab()

    elif current_tab == "voice_manager":
        create_voice_manager_tab()

    elif current_tab == "raw":
        def _get_known():
            # Combine built-in characters + any saved mappings
            base = list(get_flat_character_voices().keys())
            vm = st.session_state.get('vm_voice_mappings', {})
            return list({*base, *vm.keys()})

        create_raw_parser_tab(_get_known)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🎭 <strong>Audiobook machine</strong> | 
        Powered by ElevenLabs AI | 
        Professional Audiobook Production Suite
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
