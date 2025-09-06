import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
from audio.utils import normalize_effect_name, SOUND_EFFECTS
from audio.utils import get_flat_character_voices, get_flat_emotion_tags


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
