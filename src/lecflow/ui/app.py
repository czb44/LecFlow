
import streamlit as st

home = st.Page('home.py', title='Home')
dashboard = st.Page('dashboard.py', title='Dashboard')
#lecture page only accessible from after upload / dashboard re-directs
lecture = st.Page('lecture.py', title='Lecture', visibility='hidden')

page = st.navigation([home, dashboard, lecture])
page.run()
