import streamlit as st
from lecflow.database import get_all_lectures, del_lecture

lectures = get_all_lectures()

title_col, add_col = st.columns([8, 1])

with title_col:
    st.title('My Notes')
with add_col:
    if st.button('+'):
        st.switch_page('home.py')


for lecture in lectures:
    lec_col, menu_col = st.columns([8, 1])

    with lec_col:
        if st.button(lecture.name, key=f'lecture_{lecture.id}'): #prevent duplicate keys
            st.session_state.selected_lecture = lecture.id
            st.switch_page("lecture.py")
    with menu_col:
        with st.popover("⋮"):
            if st.button('Delete', key=f'delete_{lecture.id}'): #unique key for each button
                del_lecture(lecture.id)
                st.rerun()

    