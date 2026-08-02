import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


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


def load_data(file_path: Path) -> pd.DataFrame:
    '''Load training data csv'''
    df = pd.read_csv(file_path)
    return df

def check_balanced(df: pd.DataFrame, threshold: float=0.8) -> bool:
    '''Check class-balance of labels in training data. Returns True if data is balanced'''
    val_cts = df['Label'].value_counts()
    minor, major = sorted(val_cts.tolist())
    frac = minor / major

    if frac < threshold:
        print(f'Data unbalanced (minority/majority ratio = {frac:.2f})')
        return False
    else:
        print(f'Data balanced (minority/majority ratio = {frac:.2f})')
        return True

def split_sentences_labels(df: pd.DataFrame) -> tuple[list[str], list[int]]:
    '''Extract columns of data'''
    sentences = df['Sentence'].tolist()
    labels = df['Label'].tolist()
    return sentences, labels



if __name__ == '__main__':
    csv_path = Path('data/labeled/lecflow_contentvhousekeeping_data.csv')
    df = load_data(csv_path)
    balanced = check_balanced(df)
    sentences, labels = split_sentences_labels(df)
    
    print(sentences[:3])
    print(labels[:3])
    print(f'Data-balance state: {balanced}')
    print(f'Loaded {len(sentences)} sentences and {len(labels)} labels')
