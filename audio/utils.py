import re
from pathlib import Path
from config_loader import CHARACTER_VOICES, EMOTION_TAGS, FX_EFFECTS


def normalize_effect_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


def load_sound_effects(fx_dir="fx_library"):
    effects_map = {}
    for file in Path(fx_dir).glob("*.wav"):
        norm = normalize_effect_name(file.stem)
        effects_map[norm] = str(file.resolve())
    return effects_map


def expand_fx_variants(effect_key: str) -> set[str]:
    """Generate basic tense/state variants for an FX effect key"""
    base = effect_key.replace("_", " ").lower()
    variants = {base}
    if base.endswith("s"):
        variants.add(base[:-1])        # singular
    variants.add(base + "ed")          # past tense
    variants.add(base + "ing")         # progressive
    return variants


def get_flat_character_voices():
    flat = {}
    for category, chars in CHARACTER_VOICES.items():
        for name, data in chars.items():
            if isinstance(data, dict):  # new format
                flat[name] = {
                    "voice_id": data.get("voice_id"),
                    "gender": data.get("gender", "U")  # U = Unknown
                }
            else:  # fallback for legacy format
                flat[name] = {
                    "voice_id": data,
                    "gender": "U"
                }
    return flat


def get_flat_emotion_tags():
    flat_emotions = {}
    for category in EMOTION_TAGS.values():
        flat_emotions.update(category)
    return flat_emotions


SOUND_EFFECTS = load_sound_effects()

# Build variant lookup
FX_VARIANTS = {}
for norm, data in FX_EFFECTS.items():
    FX_VARIANTS[norm] = expand_fx_variants(data["original"])
