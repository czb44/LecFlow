import streamlit as st
from lecflow.database import get_all_lectures

lectures = get_all_lectures()

title_col, add_col = st.columns([8, 1])

with title_col:
    st.title('My Notes')
with add_col:
    if st.button('+'):
        st.switch_page('home.py')


for lecture in lectures:
    if st.button(lecture.name, key=lecture.id):
        st.session_state.selected_lecture = lecture.id
        st.switch_page("lecture.py")
