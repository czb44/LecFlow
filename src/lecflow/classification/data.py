from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..data import check_balanced, load_data, split_sentences_labels

label_key = {0: 'content', 1: 'housekeeping'}

def load_model(model_dir: Path) -> tuple[LogisticRegression, TfidfVectorizer]:
    '''Loads a trained LogisticRegression model and TfidfVectorizer from disk'''
    model = joblib.load(model_dir / 'housekeeping_classifier.joblib')
    vectorizer = joblib.load(model_dir / 'tfidf_vectorizer.joblib')
    return model, vectorizer

def load_embed_model(model_dir: Path) -> LogisticRegression:
    '''Loads a trained LogisticRegression model from sentence-transformer embeddings'''
    embed_model = joblib.load(model_dir / 'housekeeping_embed_classifier.joblib')
    return embed_model



if __name__ == '__main__':
    csv_path = Path('data/labeled/lecflow_contentvhousekeeping_data.csv')
    df = load_data(csv_path)
    balanced = check_balanced(df)
    sentences, labels = split_sentences_labels(df)
    
    print(sentences[:3])
    print(labels[:3])
    print(f'Data-balance state: {balanced}')
    print(f'Loaded {len(sentences)} sentences and {len(labels)} labels')
