import streamlit as st
from parsers.openai_parser import OpenAIParser, ParseAnalysis
from audio.voice_manager import VoiceManager
from audio.utils import get_flat_character_voices, get_flat_emotion_tags, normalize_effect_name, SOUND_EFFECTS, CHARACTER_VOICES


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
