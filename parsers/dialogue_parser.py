import streamlit as st
import re
from config_loader import CHARACTER_VOICES, EMOTION_TAGS
from audio.utils import normalize_effect_name


class DialogueParser:
    def __init__(self):
        character_voices = CHARACTER_VOICES
        emotion_tags = EMOTION_TAGS

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
