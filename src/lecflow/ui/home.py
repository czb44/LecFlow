import streamlit as st
from pathlib import Path
from lecflow.models import load_all_models
from lecflow.database import save_lecture
from lecflow.main import full_pipeline
from lecflow.notes import save_notes
from lecflow.transcript import save_transcript

housekeeping_model, sent_transformer = load_all_models()

st.title("LecFlow")
st.subheader('Bringing your lecture to life')

#only accept txt files for now
file = st.file_uploader("Upload transcript", type=["txt"])
if file:
    file_name = Path(file.name).stem
    name = " ".join(Path(file.name).stem.replace("_", " ").replace("-", " ").split()).title()
    st.write(f'Lecture: {name}')

    if st.button("Generate Notes"):
        with st.spinner("Processing..."):
            raw_transcript_txt = file.read().decode(encoding='utf-8')
            filtered, notes = full_pipeline(raw_transcript_txt, housekeeping_model, sent_transformer)

            notes_path = f'outputs/notes/{file_name}.md'
            transcript_path = f'outputs/transcript/{file_name}.txt'
            save_notes(notes, Path(notes_path))
            save_transcript(filtered, Path(transcript_path))
            
            lecture_id = save_lecture(name, notes_path, transcript_path)

        st.session_state.selected_lecture = lecture_id
        st.switch_page('lecture.py')
