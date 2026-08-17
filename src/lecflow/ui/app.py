import streamlit as st
from pathlib import Path
import joblib
from lecflow.data import load_embed_model


home = st.Page('home.py', title='Home')
dashboard = st.Page('dashboard.py', title='Dashboard')
lecture = st.Page('lecture.py', title='Lecture')

page = st.navigation([home, dashboard, lecture])
page.run()
