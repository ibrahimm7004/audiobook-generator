# parsers/openai_parser.py
import os
import re
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
from collections import defaultdict
from dotenv import load_dotenv
from audio.utils import get_flat_emotion_tags, normalize_effect_name, SOUND_EFFECTS

load_dotenv()


@dataclass
class RawParseResult:
    formatted_text: str
    dialogues: List[Dict]
    stats: Dict[str, int]


@dataclass
class ParsedDialogue:
    character: str
    text: str
    emotions: List[str]
    sound_effects: List[str]
    line_number: int
    original_line: str


@dataclass
class ParseAnalysis:
    characters_found: Dict[str, int]
    emotions_found: Dict[str, int]
    sound_effects_found: Dict[str, int]
    unsupported_characters: List[str]
    unsupported_emotions: List[str]
    unsupported_sound_effects: List[str]
    total_lines: int
    dialogue_lines: int


class OpenAIParser:
    def __init__(self, model: str = "gpt-5-mini", include_narration: bool = True, detect_fx: bool = True):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.include_narration = include_narration
        self.detect_fx = detect_fx
        self.emotion_tags = get_flat_emotion_tags()
        self.sound_effects = SOUND_EFFECTS

    def build_prompt(self, text: str) -> str:
        rules = [
            "Reformat the input prose into structured dialogue lines.",
            "Schema: [Character] (emotion1)(emotion2): dialogue *fx*",
            "One complete dialogue per line. Do not add commentary.",
            "Infer speaker names only from the input text (names, pronouns, context). Do not use any predefined list.",
        ]

        # Narrator instruction (also added to system message in parse_text)
        if self.include_narration:
            rules.append(
                "If no clear speaker is identified, attribute the line to [Narrator].")
        else:
            rules.append(
                "Ignore narration or unattributed lines completely. Only output dialogues spoken by characters.")

        if self.detect_fx:
            rules.append(
                "Include sound effects like slam, gasp, crack as *fx* tags.")

        # Do not inject any predefined characters; model must deduce speakers from text only

        examples = [
            # Character with FX
            "Input: Victor stared at the frame. \"Get up!\" he roared, slamming the door.",
            "Output: [Victor] (angry): Get up! *slam*",
            "",
            # Two consecutive lines by same speaker
            "Input: Aria folded her arms. \"You keep breaking things,\" she snapped. \"I'm not cleaning this up.\"",
            "Output: [Aria] (angry): You keep breaking things,",
            "[Aria]: I'm not cleaning this up.",
            "",
            # Pronoun switching + FX from narration
            "Input: He looked away. \"Sorry,\" he said softly. The window cracked. \"Did you hear that?\" she whispered.",
            "Output: [Victor] (gentle): Sorry,",
            "[Aria] (whispers): Did you hear that? *crack*",
            "",
            # Narrator ON example
            "Input: The lights went out. \"Who's there?\"",
            "Output: [Narrator]: Who's there?",
            "",
            # Narrator OFF example (no output)
            "Input: The lights went out. \"Who's there?\"",
            "Output:",
        ]

        return "\n".join([*rules, "", *examples, "", "Text to parse:", text])

    def parse_text(self, text: str) -> Tuple[str, List[str]]:
        prompt = self.build_prompt(text)

        narrator_policy = (
            "If no clear speaker is identified, attribute the line to [Narrator]."
            if self.include_narration
            else "Ignore narration or unattributed lines completely. Only output dialogues spoken by characters."
        )

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": f"You are a strict audiobook dialogue parser. {narrator_policy}"},
                {"role": "user", "content": prompt}
            ]
        )

        output = response.output_text.strip()
        lines = [line for line in output.split("\n") if line.strip()]
        return output, lines

    def convert(self, raw_text: str) -> RawParseResult:
        if not raw_text.strip():
            return RawParseResult(formatted_text="", dialogues=[], stats={})

        _, lines = self.parse_text(raw_text)
        dialogues, formatted_lines = [], []
        stats = {"quotes_found": 0, "lines_emitted": 0, "narration_blocks": 0}

        for line in lines:
            parsed = self._parse_line_flexible(line)
            if not parsed:
                continue
            if not self.include_narration and parsed.character == "Narrator":
                continue
            sound_effects = parsed.sound_effects if self.detect_fx else []

            emotion_text = "".join([f"({e})" for e in parsed.emotions])
            effect_text = "".join([f"*{fx}*" for fx in sound_effects])
            formatted_line = f"[{parsed.character}] {emotion_text}: {parsed.text} {effect_text}".strip(
            )

            formatted_lines.append(formatted_line)
            dialogues.append({
                "type": "speech",
                "character": parsed.character,
                "text": parsed.text,
                "emotions": parsed.emotions,
                "fx": sound_effects,
            })

            stats["lines_emitted"] += 1
            stats["quotes_found"] += 1
            if parsed.character == "Narrator":
                stats["narration_blocks"] += 1

        return RawParseResult("\n".join(formatted_lines), dialogues, stats)

    def _parse_line_flexible(self, line: str) -> Optional[ParsedDialogue]:
        original_line = line
        char_match = re.match(r"\[([^\]]+)\]\s*(\([^)]+\))*\s*:", line)
        if char_match:
            character = char_match.group(1).strip()
            emotions = re.findall(r"\(([^)]+)\)", char_match.group(0))
            text_part = re.sub(
                r"^\[([^\]]+)\]\s*(\([^)]+\))*\s*:", "", line).strip()
        elif ":" in line and not line.startswith("["):
            parts = line.split(":", 1)
            if len(parts) == 2:
                character, text_part, emotions = parts[0].strip(), parts[1].strip(), [
                ]
            else:
                return None
        else:
            return None

        sound_effects = re.findall(r"\*([^*]+)\*", text_part)
        clean_text = re.sub(r"\*[^*]+\*", "", text_part).strip()
        if not clean_text:
            return None

        return ParsedDialogue(
            character=character,
            text=clean_text,
            emotions=[e.strip() for e in emotions],
            sound_effects=[e.strip() for e in sound_effects],
            line_number=0,
            original_line=original_line,
        )
