import pytest
from pathlib import Path
from sentence_transformers import SentenceTransformer
from lecflow.data import load_embed_model
from lecflow.audio import load_audio_model

from lecflow.database import Base
import lecflow.database as database
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="session")
def sent_transformer():
    sent_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    return sent_transformer

@pytest.fixture(scope="session")
def housekeeping_model():
    housekeeping_model = load_embed_model(Path('models'))
    return housekeeping_model

@pytest.fixture(scope="session")
def audio_model():
    audio_model = load_audio_model()
    return audio_model

@pytest.fixture
def test_database(monkeypatch, tmp_path):
    db_path = tmp_path / "test_lecflow.db"
    test_engine = create_engine(f'sqlite:///{db_path}')
    TestSessionLocal = sessionmaker(bind=test_engine)
    Base.metadata.create_all(test_engine)

    monkeypatch.setattr(database, "SessionLocal", TestSessionLocal)

    yield TestSessionLocal
    test_engine.dispose() 



    
