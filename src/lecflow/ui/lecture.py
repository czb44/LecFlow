import streamlit as st
from pathlib import Path
from lecflow.database import get_lecture


lecture_id = st.session_state.get('selected_lecture')

if lecture_id is None: #no lecture selected
    st.switch_page('dashboard.py')
lecture = get_lecture(lecture_id)
if lecture is None: #ID exists but lecture DNE
    st.session_state.pop('selected_lecture', None)

if lecture_id is None:
    st.write('No lecture selected. Please select a lecture')
else:
    lecture = get_lecture(lecture_id)

    notes_txt = Path(lecture.notes_path).read_text(encoding='utf-8')
    transcript_txt = Path(lecture.transcript_path).read_text(encoding='utf-8')

    notes_tab, transcript_tab = st.tabs(["Notes", "Transcript"])

    with notes_tab:
        st.markdown(notes_txt)
    with transcript_tab:
        st.write(transcript_txt)

    st.download_button("Download Notes", notes_txt, file_name=f'{lecture.name}.md', mime='text/markdown')
