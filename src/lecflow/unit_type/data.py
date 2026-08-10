import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import pandas as pd
from ..data import load_data, check_balanced, split_sentences_labels

unit_type_labels = {0: 'definition', 1: 'example', 2: 'question', 3: 'explanation'}

def load_embed_model_ut(model_dir: Path) -> LogisticRegression:
    '''Loads a trained LogisticRegression model from sentence-transformer embeddings'''
    embed_model = joblib.load(model_dir / 'unit_type_embed_classifier.joblib')
    return embed_model


if __name__ == '__main__':
    csv_path = Path('data/labeled/lecflow_gold_unit_types_500_final_v2.csv')
    df = load_data(csv_path)
    balanced = check_balanced(df)
    sentences, labels = split_sentences_labels(df)
    
    print(sentences[:3])
    print(labels[:3])
    print(f'Data-balance state: {balanced}')
    print(f'Loaded {len(sentences)} sentences and {len(labels)} labels')
