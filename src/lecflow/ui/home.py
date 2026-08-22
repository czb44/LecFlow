from pathlib import Path
from uuid import uuid4

import streamlit as st

from lecflow.audio import get_audio_transcription, video_to_audio
from lecflow.database import save_lecture
from lecflow.llm.ollama import OllamaClient
from lecflow.main import full_pipeline
from lecflow.models import load_all_models
from lecflow.notes import save_notes
from lecflow.transcript import save_transcript

housekeeping_model, sent_transformer, audio_model = load_all_models()
llm_client = OllamaClient()

st.title("LecFlow")
st.subheader("Bringing your lecture to life")

# Muliple file upload options
upload_type = st.radio("Upload type", ["Text", "Audio", "Video"])
if upload_type == "Text":
    file = st.file_uploader("Upload transcript", type=["txt"])
elif upload_type == "Audio":
    file = st.file_uploader("Upload audio", type=["mp3", "wav", "m4a"])
else:
    file = st.file_uploader("Upload video", type=["mp4", "mov"])

if file:
    file_name = Path(file.name).stem
    name = " ".join(Path(file.name).stem.replace("_", " ").replace("-", " ").split()).title()
    st.write(f"Lecture: {name}")

    if st.button("Generate Notes"):
        with st.spinner("Processing... this may take a few minutes."):
            if upload_type == "Text":
                raw_transcript_txt = file.read().decode(encoding="utf-8")
            else:
                # Whisper requires a file_path - use temporary path
                temp_path = Path(f"data/sample/temp_{file.name}")
                temp_audio_path = (
                    None  # initialize as None in case finally runs before successful assignment
                )

                try:
                    temp_path.parent.mkdir(parents=True, exist_ok=True)  # make directories if DNE
                    temp_path.write_bytes(file.read())

                    if upload_type == "Video":
                        temp_audio_path = Path("data/sample/temp_extracted_audio.wav")
                        video_to_audio(temp_path, temp_audio_path)
                    else:  # already have audio
                        temp_audio_path = temp_path

                    raw_transcript_txt = get_audio_transcription(temp_audio_path, audio_model)

                finally:  # temporary files deleted even with error
                    if (
                        temp_audio_path != temp_path
                        and temp_audio_path
                        and temp_audio_path.exists()
                    ):
                        temp_audio_path.unlink()
                    if temp_path.exists():
                        temp_path.unlink()

            filtered, notes = full_pipeline(
                raw_transcript_txt, housekeeping_model, sent_transformer, llm_client
            )

            # prevent duplicate file names: assign unique id
            uniq_id = uuid4().hex[:8]

            notes_path = f"outputs/notes/{file_name}.{uniq_id}.md"
            transcript_path = f"outputs/transcript/{file_name}.{uniq_id}.txt"
            save_notes(notes, Path(notes_path))
            save_transcript(filtered, Path(transcript_path))
            lecture_id = save_lecture(name, notes_path, transcript_path)

        st.session_state.selected_lecture = lecture_id
        st.switch_page("lecture.py")
