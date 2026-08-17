from pathlib import Path
import streamlit as st
from sentence_transformers import SentenceTransformer
from lecflow.data import load_embed_model
from lecflow.audio import load_audio_model



#Streamlit: Load all models once and cache
@st.cache_resource
def load_all_models():
    '''Load and cache all trained models once'''
    housekeeping_model = load_embed_model(Path('models'))
    sent_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    audio_model = load_audio_model()
    return housekeeping_model, sent_transformer, audio_model
    