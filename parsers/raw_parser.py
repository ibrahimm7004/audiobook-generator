import re
import streamlit as st
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from nltk.stem import WordNetLemmatizer
from config_loader import EMOTION_TAGS, VERB_TO_EMOTION, ADVERB_TO_EMOTION, SPEECH_VERBS
from audio.utils import FX_VARIANTS, get_flat_character_voices


@dataclass
class RawParseResult:
    formatted_text: str
    dialogues: List[Dict]
    stats: Dict[str, int]


_QUOTE_NORMALIZER = [
    ("\u201c", '"'), ("\u201d", '"'), ("\u201e", '"'), ("\u201f", '"'),
    ("\u2033", '"'), ("\u2036", '"'),
]


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


class RawProseParser:
    def __init__(self, known_characters: Optional[List[str]] = None,
                 include_narration: bool = True,
                 attach_detected_fx: bool = True):

        if known_characters and len(known_characters) > 0:
            self.known_characters = [c.strip()
                                     for c in known_characters if c.strip()]
        else:
            self.known_characters = list(get_flat_character_voices().keys())

        self.include_narration = include_narration
        self.attach_detected_fx = attach_detected_fx
        self.lemmatizer = WordNetLemmatizer()

        # normalize smart quotes → plain quotes
        self.quote_pat = re.compile(r'"([^"\n]+[\.!,?]?)"')

        # Build regex with case-insensitive verbs, case-sensitive names
        verb_alt = "|".join(SPEECH_VERBS)

        # ✅ Store multiword name regex as instance variable
        self.multiword_name = r"(?-i:[A-Z][a-zA-Z]*(?:[ _-][A-Z][a-zA-Z]*)*)"

        # e.g. “said Dante”
        self.right_said_name = re.compile(
            rf"(?:^|[\s,;—-])(?P<verb>(?i:{verb_alt}))\s+(?P<name>{self.multiword_name})\b"
        )

        # e.g. “Dante said”
        self.right_name_said = re.compile(
            rf"(?:^|[\s,;—-])(?P<name>{self.multiword_name})\s+(?P<verb>(?i:{verb_alt}))\b"
        )

        # e.g. “…, Dante said.”
        self.left_name_said = re.compile(
            rf"(?P<name>{self.multiword_name})\s+(?P<verb>(?i:{verb_alt}))\b"
        )

        # e.g. “…,” he said.
        self.right_pronoun_said = re.compile(
            rf"(?:^|[\s,;—-])(?P<pronoun>(?i:he|she|they))\s+(?P<verb>(?i:{verb_alt}))\b"
        )

        # helper patterns
        self.name_token = re.compile(
            r"\b([A-Z][a-zA-Z]*(?:[ _-][A-Z][a-zA-Z]*)*)\b")
        self.adverb_pat = re.compile(r"\b([A-Za-z]+)ly\b")
        self.sent_split = re.compile(r"([.!?]\s+)")

    def _scan_for_fx(self, text: str) -> list[str]:
        """Detect FX from narration text using FX_VARIANTS"""
        found = []
        norm_text = text.lower()
        for effect_key, variants in FX_VARIANTS.items():
            for v in variants:
                if v in norm_text:   # substring match, can be upgraded to regex \b
                    found.append(effect_key)
                    break  # stop after first match for this effect
        return found

    def _normalize_quotes(self, text: str) -> str:
        for frm, to in _QUOTE_NORMALIZER:
            text = text.replace(frm, to)
        return text

    def normalize_emotion(self, label: str) -> str:
        """Map a raw emotion label to a standardized tag if available"""
        label = label.lower()
        for category, mapping in EMOTION_TAGS.items():
            if label in mapping:
                return mapping[label]
        return f"[{label}]"  # fallback if not defined

    def _infer_emotions(self, context: str, verb: Optional[str]) -> List[str]:
        emotions: List[str] = []

        # --- Verb → Emotion
        if verb:
            lemma = self.lemmatizer.lemmatize(verb.lower())
            if lemma in VERB_TO_EMOTION:
                norm = self.normalize_emotion(VERB_TO_EMOTION[lemma])
                if norm not in emotions:
                    emotions.append(norm)

        # --- Adverb → Emotion
        for adv in self.adverb_pat.findall(context or ""):
            tag = ADVERB_TO_EMOTION.get(adv.lower())
            if tag:
                norm = self.normalize_emotion(tag)
                if norm not in emotions:
                    emotions.append(norm)

        return emotions

    def _choose_speaker(self, candidates: List[str], last_speaker: Optional[str]) -> str:
        """
        Choose the best speaker from candidates.
        Preference order:
        1. Known characters (custom or predefined)
        2. Last speaker if reasonable
        3. First candidate
        4. Narrator
        """
        for c in candidates:
            if c in self.known_characters:
                return c
        if last_speaker and (not candidates or last_speaker in candidates or not self.known_characters):
            return last_speaker
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

    def _handle_post_quote_attribution(self, quote_text: str, context: str) -> Optional[Tuple[str, List[str], int]]:
        """
        Try to attribute based on trailing pattern, e.g.:
        "I'm home," said Brad.
        "I'm home," Brad said.
        "I'm home," he said.
        "I'm fine," Brad said softly.

        Returns (speaker_or_pronoun, inferred_emotions, end_pos) if matched, else None.
        end_pos = how many characters of context were consumed (so caller can trim tail).
        """

        emotions: List[str] = []

        multiword_name = self.multiword_name

        # Pattern 1: "quote," said Brad (with optional adverb)
        m1 = re.match(
            rf'^\s*(?:,)?\s*(?P<verb>[A-Za-z]+)\s+(?P<name>{multiword_name})(?:\s+(?P<adv>[a-z]+ly))?',
            context
        )
        if m1:
            verb = m1.group("verb").lower()
            name = m1.group("name")
            adv = m1.groupdict().get("adv")
            lemma = self.lemmatizer.lemmatize(verb)
            if lemma in VERB_TO_EMOTION:
                emotions.append(self.normalize_emotion(VERB_TO_EMOTION[lemma]))
            if adv and adv.lower() in ADVERB_TO_EMOTION:
                emotions.append(self.normalize_emotion(
                    ADVERB_TO_EMOTION[adv.lower()]))
            return name, emotions, m1.end()

        # Pattern 2: "quote," Brad said (with optional adverb)
        m2 = re.match(
            rf'^\s*(?:,)?\s*(?P<name>{multiword_name})\s+(?P<verb>[A-Za-z]+)(?:\s+(?P<adv>[a-z]+ly))?',
            context
        )
        if m2:
            name = m2.group("name")
            verb = m2.group("verb").lower()
            adv = m2.groupdict().get("adv")
            lemma = self.lemmatizer.lemmatize(verb)
            if lemma in VERB_TO_EMOTION:
                emotions.append(self.normalize_emotion(VERB_TO_EMOTION[lemma]))
            if adv and adv.lower() in ADVERB_TO_EMOTION:
                emotions.append(self.normalize_emotion(
                    ADVERB_TO_EMOTION[adv.lower()]))
            return name, emotions, m2.end()

        # Pattern 3: "quote," he/she said (with optional adverb)
        m3 = re.match(
            r'^\s*(?:,)?\s*(?P<pronoun>he|she|they)\s+(?P<verb>[A-Za-z]+)(?:\s+(?P<adv>[a-z]+ly))?',
            context,
            re.IGNORECASE
        )
        if m3:
            pronoun = m3.group("pronoun").lower()
            verb = m3.group("verb").lower()
            adv = m3.groupdict().get("adv")
            lemma = self.lemmatizer.lemmatize(verb)
            if lemma in VERB_TO_EMOTION:
                emotions.append(self.normalize_emotion(VERB_TO_EMOTION[lemma]))
            if adv and adv.lower() in ADVERB_TO_EMOTION:
                emotions.append(self.normalize_emotion(
                    ADVERB_TO_EMOTION[adv.lower()]))
            return pronoun, emotions, m3.end()

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

        # Track last seen characters for pronoun resolution
        last_seen_male: Optional[str] = None
        last_seen_female: Optional[str] = None
        male_seen: set = set()
        female_seen: set = set()

        stats = {
            "quotes_found": 0,
            "lines_emitted": 0,
            "speaker_from_after": 0,
            "speaker_from_before": 0,
            "speaker_from_last": 0,
            "speaker_unknown": 0,
            "narration_blocks": 0,
        }

        # Allowed characters
        if st.session_state.get("use_custom_characters"):
            allowed_chars = set(st.session_state.get(
                "custom_characters", {}).keys())
        else:
            allowed_chars = set(get_flat_character_voices().keys())
        allowed_chars_lower = {c.lower() for c in allowed_chars}

        for para in paragraphs:
            spans = list(self.quote_pat.finditer(para))

            if not spans:
                if self.include_narration:
                    fx_hits = self._scan_for_fx(para)
                    formatted_lines.append(f"[Narrator]: {para}".strip())
                    dialogues.append({
                        "character": "Narrator",
                        "text": para,
                        "emotions": [],
                        "sound_effects": fx_hits
                    })
                    stats["narration_blocks"] += 1

                # Track names in narration for pronoun context
                for token in self.name_token.findall(para):
                    if token in self.known_characters:
                        char_info = get_flat_character_voices().get(token, {})
                        gender = char_info.get("gender", "")
                        if gender == "M":
                            last_seen_male, male_seen = token, {token}
                        elif gender == "F":
                            last_seen_female, female_seen = token, {token}

                last_speaker = None
                continue

            consumed_offset = 0

            for i, m in enumerate(spans):
                stats["quotes_found"] += 1
                quote_text = m.group(1)

                prev_end = spans[i - 1].end() if i > 0 else 0
                next_start = spans[i + 1].start() if i + \
                    1 < len(spans) else len(para)
                left_ctx = para[max(prev_end, m.start() - 160): m.start()]
                right_ctx = para[m.end(): min(next_start, m.end() + 160)]

                candidates: List[str] = []
                inferred_emotions: List[str] = []

                # --- Update last_seen trackers BEFORE attribution
                for token in self.name_token.findall(left_ctx + " " + right_ctx):
                    if token in self.known_characters:
                        char_info = get_flat_character_voices().get(token, {})
                        gender = char_info.get("gender", "")
                        if gender == "M":
                            last_seen_male, male_seen = token, {token}
                        elif gender == "F":
                            last_seen_female, female_seen = token, {token}

                # --- Explicit pre-quote attribution (Name verb "...")
                pre_attr = re.search(
                    rf'(?P<name>{self.multiword_name})\s+(?P<verb>{ "|".join(SPEECH_VERBS)})\s*$',
                    left_ctx,
                    re.IGNORECASE
                )
                if pre_attr:
                    name, verb = pre_attr.group(
                        "name"), pre_attr.group("verb").lower()
                    candidates.append(name)
                    lemma = self.lemmatizer.lemmatize(verb)
                    if lemma in VERB_TO_EMOTION:
                        inferred_emotions.append(
                            self.normalize_emotion(VERB_TO_EMOTION[lemma]))
                    stats["speaker_from_before"] += 1

                # --- Pre-quote attribution
                before_matches = list(self.left_name_said.finditer(left_ctx))
                if before_matches:
                    bm = before_matches[-1]
                    name, verb = bm.group("name"), bm.group("verb").lower()
                    candidates.append(name)
                    lemma = self.lemmatizer.lemmatize(verb)
                    if lemma in VERB_TO_EMOTION:
                        inferred_emotions.append(
                            self.normalize_emotion(VERB_TO_EMOTION[lemma]))
                    stats["speaker_from_before"] += 1
                    consumed_offset = max(consumed_offset, bm.end())

                # --- Post-quote attribution
                if not candidates:
                    after_name_first = self.right_name_said.search(right_ctx)
                    after_said_name = self.right_said_name.search(right_ctx)
                    best = after_name_first if (after_name_first and
                                                (not after_said_name or after_name_first.start() < after_said_name.start())) else after_said_name
                    if best:
                        name, verb = best.group(
                            "name"), best.group("verb").lower()
                        candidates.append(name)
                        lemma = self.lemmatizer.lemmatize(verb)
                        if lemma in VERB_TO_EMOTION:
                            inferred_emotions.append(
                                self.normalize_emotion(VERB_TO_EMOTION[lemma]))
                        stats["speaker_from_after"] += 1
                        consumed_offset = max(consumed_offset, best.end())

                # --- Trailing attribution
                if not candidates:
                    post_attr = self._handle_post_quote_attribution(
                        quote_text, right_ctx)
                    if post_attr:
                        name_or_pronoun, emots, consumed = post_attr
                        if name_or_pronoun.lower() in ["he", "she"]:
                            if name_or_pronoun == "he":
                                if len(male_seen) == 1 and last_seen_male:
                                    candidates.append(last_seen_male)
                                elif self.include_narration:
                                    candidates.append("Narrator")
                            elif name_or_pronoun == "she":
                                if len(female_seen) == 1 and last_seen_female:
                                    candidates.append(last_seen_female)
                                elif self.include_narration:
                                    candidates.append("Narrator")
                        else:
                            candidates.append(name_or_pronoun)
                        inferred_emotions.extend(emots)
                        stats["speaker_from_after"] += 1
                        consumed_offset = max(consumed_offset, consumed)

                # --- Intro attribution ("It's Victor…")
                if not candidates:
                    intro = re.search(
                        rf"(?:^|\b)(?:it['’]s|this is)\s+(?P<name>{self.multiword_name})",
                        left_ctx + " " + right_ctx,
                        re.IGNORECASE
                    )
                    if intro:
                        candidates.append(intro.group("name"))
                        stats["speaker_from_after"] += 1

                # --- Pronoun attribution with ambiguity guard
                if not candidates:
                    pron = self.right_pronoun_said.search(right_ctx)
                    if pron:
                        pronoun = pron.group("pronoun").lower()
                        verb = pron.group("verb").lower()
                        lemma = self.lemmatizer.lemmatize(verb)
                        if lemma in VERB_TO_EMOTION:
                            inferred_emotions.append(
                                self.normalize_emotion(VERB_TO_EMOTION[lemma]))

                        if pronoun == "he":
                            if len(male_seen) == 1 and last_seen_male:
                                candidates.append(last_seen_male)
                            elif self.include_narration:
                                candidates.append("Narrator")
                            else:
                                continue
                        if pronoun == "she":
                            if len(female_seen) == 1 and last_seen_female:
                                candidates.append(last_seen_female)
                            elif self.include_narration:
                                candidates.append("Narrator")
                            else:
                                continue
                        else:
                            if self.include_narration:
                                candidates.append("Narrator")
                        stats["speaker_from_last"] += 1

                # --- Adverbs → emotions
                for adv in list(right_ctx.lower().split()):
                    if adv in ADVERB_TO_EMOTION:
                        inferred_emotions.append(
                            self.normalize_emotion(ADVERB_TO_EMOTION[adv]))
                        consumed_offset = max(
                            consumed_offset, right_ctx.lower().find(adv) + len(adv))

                # --- Explicit mentions inside quote
                if not candidates:
                    for char in allowed_chars:
                        if re.search(rf"\b{re.escape(char)}\b", quote_text, re.IGNORECASE):
                            if len(allowed_chars) == 2:
                                other = next(
                                    c for c in allowed_chars if c.lower() != char.lower())
                                candidates.append(other)
                            break

                # --- Final choice
                filtered_candidates = [
                    c for c in candidates if c.lower() in allowed_chars_lower]
                speaker = self._choose_speaker(
                    filtered_candidates, last_speaker)

                # --- Handle missing/unknown speakers
                if not speaker:
                    speaker = "Narrator"
                    stats["speaker_unknown"] += 1
                elif (not st.session_state.get("use_custom_characters")
                      and speaker.lower() not in allowed_chars_lower):
                    # Only enforce voice-config check if using predefined voices
                    speaker = "Narrator"
                    stats["speaker_unknown"] += 1

                # --- Emit line
                unique_emotions = list(dict.fromkeys(inferred_emotions))
                emotion_text = "".join(f"({e})" for e in unique_emotions if e)
                formatted_lines.append(
                    f"[{speaker}] {emotion_text}: {quote_text}".strip())
                dialogues.append({
                    "type": "speech",
                    "character": speaker,
                    "text": quote_text,
                    "emotions": unique_emotions,
                    "fx": [],
                })
                stats["lines_emitted"] += 1
                last_speaker = speaker

                if speaker in allowed_chars:
                    if st.session_state.get("use_custom_characters"):
                        gender = st.session_state.custom_characters.get(
                            speaker, {}).get("gender", "")
                    else:
                        char_info = get_flat_character_voices().get(speaker, {})
                        gender = char_info.get("gender", "")
                    if gender == "M":
                        last_seen_male, male_seen = speaker, {speaker}
                    elif gender == "F":
                        last_seen_female, female_seen = speaker, {speaker}

            # --- Trailing narration
            tail = para[spans[-1].end() + consumed_offset:].strip()
            if tail and self.include_narration:
                if not re.fullmatch(r"^[\.,!?;:]+$", tail):
                    fx_hits = self._scan_for_fx(tail)
                    formatted_lines.append(f"[Narrator]: {tail}")
                    dialogues.append({
                        "character": "Narrator",
                        "text": tail,
                        "emotions": [],
                        "sound_effects": fx_hits
                    })
                    stats["narration_blocks"] += 1

        return RawParseResult("\n".join(formatted_lines), dialogues, stats)
