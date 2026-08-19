from pathlib import Path

import joblib  #save and reload trained sklearn model
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ..data import check_balanced, load_data, split_sentences_labels


def train_classifier_embeddings(sentences: list[str], labels: list[int]) -> tuple[LogisticRegression, SentenceTransformer, tuple[list[str], list[int]]]:
    '''Sentence-transforemer embeddings + Logistic Regression pipeline'''
    #Holdout 20% for test; random_state for reproducibility; stratify to keep class proportions similar
    x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=2, stratify=labels)

    sent_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = sent_transformer.encode(x_train)

    embed_model = LogisticRegression(max_iter=1000, class_weight='balanced') #compensate for skewed data
    embed_model.fit(embeddings, y_train)

    #return transformer and test for evaluation later
    return embed_model, sent_transformer, (x_test, y_test) 

def save_test_data(x_test: list[str], y_test: list[int], output_path: Path) -> None:
    '''Save held-out test data (sentences and labels) to CSV for later evaluation without re-training'''
    #make folder if doesn't already exist, create outputs if necessary
    output_path.parent.mkdir(parents=True, exist_ok=True)  
    df = pd.DataFrame({'Sentence': x_test, 'Label': y_test}) #same shape as original
    df.to_csv(output_path, index=False) #save w/o row numbers

def save_embed_classifier(embed_model: LogisticRegression, model_dir: Path) -> None:
    '''Save a trained LogisticRegression model from sentence-transformer embeddings.'''
    #make folder if doesn't already exist, create outputs if necessary
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(embed_model, model_dir / 'unit_type_embed_classifier.joblib')


if __name__ == '__main__':
    csv_path = Path('data/labeled/lecflow_gold_unit_types_500_final_v2.csv')
    df = load_data(csv_path)
    balanced = check_balanced(df)
    sentences, labels = split_sentences_labels(df)

    embed_model, sent_transformer, (x_test, y_test) = train_classifier_embeddings(sentences, labels)

    save_test_data(x_test, y_test, Path('data/splits/gold_unit_type_test_set.csv'))

    save_embed_classifier(embed_model, Path('models'))
    print('Sentence-embedding LR Model saved successfully')
