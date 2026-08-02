from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from ..transcript import load_transcript, filter_transcript, split_into_sentences


def train_clusters(sentences: list[str], k: int) -> tuple[TfidfVectorizer, KMeans, list[int]]:
    '''Vectorize sentences with TF-IDF classifier and cluster into K clusters with KMeans'''
    #Vectorize: learn vocab and TF-IDF weights
    vectorizer = TfidfVectorizer(stop_words='english') #remove stopwords
    sentence_vectors = vectorizer.fit_transform(sentences) #fit on training

    model = KMeans(n_clusters=k, random_state=2)
    model.fit(sentence_vectors)
    cluster_labels = model.labels_.tolist() #pull out cluster assignments, convert to list

    return vectorizer, model, cluster_labels

def sentence_embedding_cluster_train(sentences: list[str], k: int) -> tuple[KMeans, list[int]]:
    '''Vectorize sentences with sentence embeddings and cluster into K clusters with KMeans'''
    sent_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = sent_transformer.encode(sentences)

    model = KMeans(n_clusters=k, random_state=2)
    model.fit(embeddings)
    cluster_labels = model.labels_.tolist() #pull out cluster assignments, convert to list

    return model, cluster_labels


def group_by_cluster(sentences: list[str], cluster_labels: list[int]) -> dict[int, list[str]]:
    '''Group all sentences by their assigned cluster number, regardeless of poisition in transcript.
    Note: does not preserve origical sentence order - can merge non-adjacent references to a topic.'''
    groups = defaultdict(list)
    for sent, label in zip(sentences, cluster_labels):
        groups[label].append(sent)
    return groups


if __name__ == '__main__':
    file_path = Path("data/sample/lecture_2.txt")
    
    transcript = load_transcript(file_path)
    filtered = filter_transcript(transcript)
    sentences = split_into_sentences(filtered)

    for k in range(4,5):
        (vectorizer, model, cluster_labels) = train_clusters(sentences, k)
        groups = group_by_cluster(sentences, cluster_labels)
        print(f'K = {k}')
        for cluster_number, sentences_grouped in groups.items():
            print(f'Cluster {cluster_number}:')
            for sentence in sentences_grouped:
                print(f' - {sentence}')
        print('\n--------------------------\n')

    for k in range(4,5):
        (model, cluster_labels) = sentence_embedding_cluster_train(sentences, k)
        groups = group_by_cluster(sentences, cluster_labels)
        print(f'K = {k}')
        for cluster_number, sentences_grouped in groups.items():
            print(f'Cluster {cluster_number}:')
            for sentence in sentences_grouped:
                print(f' - {sentence}')
        print('\n--------------------------\n')
