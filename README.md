# LecFlow

## Description
LecFlow is a local Python app that converts lecture video, audio, or transcripts into organized notes via local transcription, machine learning techniques, and LLM refinement.

## Features
- Audio transcription: extracts audio from video with FFmpeg and transcribes locally with faster-whisper.
- Transcript processing: filters out extra white space and formats with consistency.
- Content filtering: separates academic content and housekeeping remarks with sentence embeddings and logistic regression.
- Local LLM refinement: filters insignificant or irrelevant academic content or housekeeping sentences using Qwen (Ollama).
- Topic segmentation: identifies sequential topic boundaries by comparing adjacent windows of sentence embeddings using cosine similarity.
- Unit classification: uses rule-based classification to identify definitions, questions, examples, and explanations.
- Notes generation: converts lecture content into organized Markdown notes with a table of contents.
- Lecture metadata storage: stores lecture name, ID, creation date, and local output paths for generated notes and filtered transcripts.

## Installation and Launch

### First-Time Setup

#### Normal User:

```bash
git clone
cd lecflow

brew install ffmpeg
brew install ollama

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Download Qwen model locally (only required once)
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

#### Developer:

```bash
git clone
cd lecflow

brew install ffmpeg
brew install ollama

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"

pre-commit install

# Download Qwen model locally (only required once)
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

### Run LecFlow

```bash
# Terminal 1
ollama serve

# Terminal 2 from project root
cd lecflow
source .venv/bin/activate

streamlit run src/lecflow/ui/app.py 
```

The local SQLite database is automatically initialized on first run.
Keep `ollama serve` running while LecFlow is in use. Press `Ctrl+C` to stop the Ollama server.

### Expected Runtime
Relevant examples run on 8GB M1 Macbook Air:
- 1 hr long lecture audio / video: up to ~7 minutes
- 1 hr long lecture transcript: up to ~ 3 minutes

## Architecture
```text
                     ┌───────────────┐
                     │ Audio / Video │
                     └───────┬───────┘
                             │
                  Video: FFmpeg + faster-whisper
                  Audio: faster-whisper
                             │
                             ▼
                     ┌───────────────┐
                     │  Transcript   │
                     └───────┬───────┘
                             │
                             ▼
                    Transcript Processing
                             │
                             ▼
                    Sentence Splitting (spaCy)
                             │
                             ▼
              Housekeeping / Content Classification
              (embeddings + Logistic Regression)
                             │
                             ▼
                  Housekeeping Refinement
                     (Ollama + Qwen)
                             │
                             ▼
                 Sequential Topic Segmentation
            (embeddings + neighboring-window cosine similarity)
                             │
                             ▼
                    Content Refinement
                     (Ollama + Qwen)
                             │
                             ▼
                      Topic Labeling
               (centroid + noun-phrase matching)
                             │
                             ▼
                    Unit Classification
       (definition / example / question / explanation)
                             │
                             ▼
                     Notes Generation
                 (Markdown + table of contents)
                             │
                             ▼
                       Lecture Storage
                    (SQLite + SQLAlchemy)
                             │
                             ▼
                       Streamlit UI
```
> **Diagram note:** After classifying housekeeping and academic content, the two groups are processed independently. Housekeeping sentences are filtered for relevance with the local LLM, while academic content undergoes topic segmentation, LLM refinement, topic labeling, and unit classification. Both are combined again during final note generation.

**Stack:** Python, scikit-learn, sentence-transformers, spaCy, faster-whisper, SQLAlchemy/SQLite, Streamlit, pytest, Ruff, mypy, pre-commit, GitHub Actions, Ollama (Qwen3 4B)

## Methodology
See [METHODOLOGY.md](METHODOLOGY.md) for details on the development process and signficant decisions.

## Testing
LecFlow uses the following:
- pytest (testing)
- Ruff (linting & formatting)
- mypy (type checking)
- pre-commit (automated checks)
- GitHub Actions (CI on pushes or pull requests)

To run local checks:
```bash
ruff check .
ruff format --check .
mypy src
pytest -v
```


## Project Structure
```text
lecflow/
├── .github/
│   └── workflows/
│       └── tests.yml             # CI test suite
├── models/
│   ├── housekeeping_embed_classifier.joblib
│   └── housekeeping_embed_classifier_v2.joblib  # Retrained, default classifier
├── src/
│   └── lecflow/
│       ├── classification/       # Housekeeping/content classification
│       │   ├── data.py
│       │   ├── evaluate.py
│       │   ├── predict.py
│       │   └── train.py
│       ├── llm/                  # Local LLM refinement
│       │   ├── ollama.py
│       │   └── prompts.py
│       ├── topics/               # Topic segmentation and labeling
│       │   └── cluster.py
│       ├── ui/                   # Streamlit interface
│       │   ├── app.py
│       │   ├── dashboard.py
│       │   ├── home.py
│       │   └── lecture.py
│       ├── unit_type/            # Definition/example/question/explanation classification
│       │   ├── data.py
│       │   ├── evaluate.py
│       │   ├── predict.py
│       │   ├── rules.py
│       │   └── train.py
│       ├── audio.py              # Audio/video transcription
│       ├── database.py           # SQLite database operations
│       ├── main.py               # Full transcript processing pipeline
│       ├── models.py             # Model loading
│       ├── notes.py              # Note generation
│       └── transcript.py         # Transcript processing
├── tests/
│   ├── assets/
│   │   ├── short_test_audio.m4a
│   │   └── short_test_video.MOV
│   ├── conftest.py
│   ├── test_audio.py
│   ├── test_database.py
│   ├── test_housekeeping.py
│   ├── test_llm.py
│   ├── test_notes.py
│   ├── test_topics.py
│   ├── test_transcript.py
│   └── test_unit_type.py
├── .gitignore
├── .pre-commit-config.yaml
├── METHODOLOGY.md
├── pyproject.toml
└── README.md
```

Model Artifacts: The repository includes the sentence-embedding housekeeping classification model implemented in the pipeline. Other models trained during development (TF-IDF and other embedding approaches) are not included in the repository as they are not necessary for the pipeline. The Qwen model used for the LLM refinement and filtering is downloaded locally and is not present in the repository.
