import streamlit as st
from ui.auth import check_password
from ui.sidebar import create_navigation_sidebar
from ui.upload import create_file_upload_interface
from parsers.text_parser import TextParser
from audio.utils import get_flat_character_voices
from ui.analysis import display_analysis_results, create_voice_management_interface
from ui.tabs import (
    create_teaser_generator_tab,
    create_emotion_preview_tab,
    create_voice_manager_tab,
    create_raw_parser_tab,
    generate_audio_for_tab
)


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
