from collections import defaultdict
from pathlib import Path

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..transcript import filter_transcript, load_transcript, split_into_sentences


def train_clusters(sentences: list[str], k: int) -> tuple[TfidfVectorizer, KMeans, list[int]]:
    '''Vectorize sentences with TF-IDF classifier and cluster into K clusters with KMeans'''
    #Vectorize: learn vocab and TF-IDF weights
    vectorizer = TfidfVectorizer(stop_words='english') #remove stopwords
    sentence_vectors = vectorizer.fit_transform(sentences) #fit on training

    model = KMeans(n_clusters=k, random_state=2)
    model.fit(sentence_vectors)
    cluster_labels = model.labels_.tolist() #pull out cluster assignments, convert to list

    return vectorizer, model, cluster_labels

def sentence_embedding_cluster_train(sentences: list[str], sent_transformer: SentenceTransformer, k: int) -> tuple[KMeans, list[int]]:
    '''Vectorize sentences with sentence embeddings and cluster into K clusters with KMeans'''
    #Edge case handling: 0 or <k content sentences 
    if not sentences:
        raise ValueError("Cannot cluster: 0 content sentences found")
    k = min(k, len(sentences))

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

nlp = spacy.load("en_core_web_sm") # load spacy one time
def get_candidate_phrases(sentences: list[str]) -> list[str]:
    '''Extracts noun labels from sentences as candidates for topic labels'''
    nouns = []
    for sentence in sentences:
        proc_sentence = nlp(sentence)
        for word in proc_sentence.noun_chunks:
            nouns.append(word.text)
    return list(set(nouns)) #remove duplicates

def get_topic_labels(sentences: list[str], sent_transformer: SentenceTransformer) -> str:
    '''Finds the most representative phrase of a group of sentences'''
    embeddings = sent_transformer.encode(sentences)
    
    #Centroid: mean of embeddings along rows, reshap for cosine similarity
    mean_embedding = np.mean(embeddings, axis=0).reshape(1,-1) #1D -> 2D

    candidate_nouns = get_candidate_phrases(sentences)
    if not candidate_nouns: #Edge case - no nouns identified
        similarities = cosine_similarity(embeddings, mean_embedding).flatten() # -> 1D
        closest_index = np.argmax(similarities)
        return sentences[closest_index][:80] #print sentence, cap at 80 characters

    #Compare candidate embeddings
    candidate_embeddings = sent_transformer.encode(candidate_nouns)
    candidate_scores = cosine_similarity(candidate_embeddings, mean_embedding).flatten()
    best_index = np.argmax(candidate_scores)
    return candidate_nouns[best_index]



if __name__ == '__main__':
    file_path = Path("data/sample/lecture_2.txt")
    
    transcript = load_transcript(file_path)
    filtered = filter_transcript(transcript)
    sentences = split_into_sentences(filtered)

    #load sentence transformer
    sent_transformer = SentenceTransformer('all-MiniLM-L6-v2')


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
        (model, cluster_labels) = sentence_embedding_cluster_train(sentences, sent_transformer, k)
        groups = group_by_cluster(sentences, cluster_labels)
        print(f'K = {k}')
        for cluster_number, sentences_grouped in groups.items():
            print(f'Cluster {cluster_number}:')
            for sentence in sentences_grouped:
                print(f' - {sentence}')
        print('\n--------------------------\n')
