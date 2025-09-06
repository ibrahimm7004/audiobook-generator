from pydub import AudioSegment
from elevenlabs import ElevenLabs
from settings import ELEVENLABS_API_KEY
from pathlib import Path
import re
import tempfile
import streamlit as st
import io
import time
from datetime import datetime
from pydub import AudioSegment, effects
from audio.utils import SOUND_EFFECTS, get_flat_character_voices
from utils.downloads import create_output_folders


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
                return AudioSegment.from_file(fx_file, format=ext.lstrip("."))
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
                voice_id = working_voices.get(char) \
                    or working_voices.get(char.lower()) \
                    or entry.get("voice_id") \
                    or st.session_state.get("default_custom_voice", "default_voice_id_here")

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
