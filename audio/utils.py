import re
import os
from pathlib import Path
from config_loader import CHARACTER_VOICES, EMOTION_TAGS, FX_EFFECTS


def normalize_effect_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


def _build_fx_lookup_from_json(fx_dir: str = "fx_library"):
    """Primary: build FX lookup from FX_EFFECTS config (effect key -> absolute file path).
    Fallback: if a file is missing, we emit a warning and skip; filesystem scan will still cover it.
    """
    lookup = {}
    base = Path(fx_dir)
    for norm_key, data in FX_EFFECTS.items():
        filename = data.get("file")
        if not filename:
            continue
        candidate = base / filename
        if candidate.exists():
            lookup[norm_key] = str(candidate.resolve())
        else:
            # Soft warning; do not crash
            print(
                f"⚠ Warning: FX file '{filename}' for '{norm_key}' not found in {fx_dir}/")
    return lookup


def _scan_fx_directory(fx_dir: str = "fx_library"):
    """Secondary: filesystem scan for .wav files as a fallback source."""
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


# Build FX lookup with JSON as primary, filesystem scan as fallback
_json_fx = _build_fx_lookup_from_json()
_scan_fx = _scan_fx_directory()
# Merge with preference for JSON-defined entries
SOUND_EFFECTS = {**_scan_fx, **_json_fx} if _json_fx else _scan_fx

# Build variant lookup
FX_VARIANTS = {}
for norm, data in FX_EFFECTS.items():
    FX_VARIANTS[norm] = expand_fx_variants(data["original"])
