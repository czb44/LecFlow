from collections import defaultdict
from pathlib import Path

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from ..transcript import filter_transcript, load_transcript, split_into_sentences


def train_clusters(sentences: list[str], k: int) -> tuple[TfidfVectorizer, KMeans, list[int]]:
    """Vectorize sentences with TF-IDF classifier and cluster into K clusters with KMeans"""
    # Vectorize: learn vocab and TF-IDF weights
    vectorizer = TfidfVectorizer(stop_words="english")  # remove stopwords
    sentence_vectors = vectorizer.fit_transform(sentences)  # fit on training

    model = KMeans(n_clusters=k, random_state=2)
    model.fit(sentence_vectors)
    cluster_labels = model.labels_.tolist()  # pull out cluster assignments, convert to list

    return vectorizer, model, cluster_labels


def sentence_embedding_cluster_train(
    sentences: list[str],
    sent_transformer: SentenceTransformer,
    k: int | None = None,
) -> tuple[KMeans, list[int]]:
    """Vectorize sentences with sentence embeddings and cluster into K clusters with KMeans"""
    # Edge case handling: 0 or <k content sentences
    if not sentences:
        raise ValueError("Cannot cluster: 0 content sentences found")

    embeddings = sent_transformer.encode(sentences)

    if k is None:
        k = find_best_k(embeddings)

    model = KMeans(n_clusters=k, random_state=2)
    model.fit(embeddings)
    cluster_labels = model.labels_.tolist()  # pull out cluster assignments, convert to list

    return model, cluster_labels


def group_by_cluster(sentences: list[str], cluster_labels: list[int]) -> dict[int, list[str]]:
    """Group all sentences by their assigned cluster number, regardeless of poisition in transcript.
    Note: doesn't preserve origical sentence order; can merge non-adjacent references of topic."""
    groups = defaultdict(list)
    for sent, label in zip(sentences, cluster_labels):
        groups[label].append(sent)
    return groups


def group_adjacent(sentences: list[str], cluster_labels: list[int]) -> list[list[str]]:
    """Groups consecutive sentences of the same cluster into ordered blocks. Maintains
    sentence order; topic mentioned in two different spots becomes two different blocks."""
    if not sentences:  # avoid error if empty
        return []

    if len(sentences) != len(cluster_labels):
        raise ValueError("sentences and cluster_labels unequal length")

    grouped_blocks = []
    current_block, current_cluster = [sentences[0]], cluster_labels[0]
    for sentence, label in zip(sentences[1:], cluster_labels[1:]):
        if label == current_cluster:
            current_block.append(sentence)
        else:
            grouped_blocks.append(current_block)
            current_cluster = label
            current_block = [sentence]
    grouped_blocks.append(current_block)
    return grouped_blocks


nlp = spacy.load("en_core_web_sm")  # load spacy one time


def get_candidate_phrases(sentences: list[str]) -> list[str]:
    """Extracts noun labels from sentences as candidates for topic labels"""
    nouns = []
    for sentence in sentences:
        proc_sentence = nlp(sentence)
        for word in proc_sentence.noun_chunks:
            nouns.append(word.text)
    return list(set(nouns))  # remove duplicates


def get_topic_labels(sentences: list[str], sent_transformer: SentenceTransformer) -> str:
    """Finds the most representative phrase of a group of sentences"""
    embeddings = sent_transformer.encode(sentences)

    # Centroid: mean of embeddings along rows, reshap for cosine similarity
    mean_embedding = np.mean(embeddings, axis=0).reshape(1, -1)  # 1D -> 2D

    candidate_nouns = get_candidate_phrases(sentences)
    if not candidate_nouns:  # Edge case - no nouns identified
        similarities = cosine_similarity(embeddings, mean_embedding).flatten()  # -> 1D
        closest_index = np.argmax(similarities)
        return sentences[closest_index][:80]  # print sentence, cap at 80 characters

    # Compare candidate embeddings
    candidate_embeddings = sent_transformer.encode(candidate_nouns)
    candidate_scores = cosine_similarity(candidate_embeddings, mean_embedding).flatten()
    best_index = np.argmax(candidate_scores)
    return candidate_nouns[best_index]


def find_best_k(embeddings, min_k: int = 2, max_k: int = 20) -> int:
    """Determine k, the number of clusters, using silhouette score."""
    # Clusters cannot outnumber embeddings - 1 for silhouette
    max_k = min(
        max_k, max(2, len(embeddings) // 10), len(embeddings) - 1
    )  # ~ 1 cluster per 10 sentences allowed

    if max_k < min_k:
        return 1

    scores = []
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=2)
        labels = model.fit_predict(embeddings)
        if len(set(labels)) < 2:  # silhouette needs >= 2 unique clusters
            continue

        score = silhouette_score(embeddings, labels)
        scores.append((k, score))

    if not scores:
        return 1

    # Near-best / tolerance: avoid overclustering long lectures
    best_score = max(score for _, score in scores)
    tol_level = 0.005
    for k, score in scores:
        if score >= best_score - tol_level:
            return k

    return scores[-1][0]


def split_large_blocks(
    blocks: list[list[str]],
    sent_transformer: SentenceTransformer,
    max_block_size: int = 50,
    window_size: int = 3,
) -> list[list[str]]:
    """Split oversized blocks at best boundary with recursion"""
    final_blocks = []  # fit size parameters ~ small enouch

    for block in blocks:
        if len(block) <= max_block_size:
            final_blocks.append(block)  # block small enouch
            continue

        # if not small enough, repeat boundary logic
        embeddings = sent_transformer.encode(block)

        similarities = []

        for i in range(window_size, len(block) - window_size):
            left = embeddings[i - window_size : i]
            right = embeddings[i : i + window_size]

            left_mean = np.mean(left, axis=0).reshape(1, -1)
            right_mean = np.mean(right, axis=0).reshape(1, -1)

            score = cosine_similarity(left_mean, right_mean)[0][0]
            similarities.append((i, score))

        if not similarities:
            # if block is too short to compare
            split_idx = len(block) // 2
        else:
            # lowest similarity = strongest topic change
            split_idx = min(similarities, key=lambda x: x[1])[0]

        left_block = block[:split_idx]
        right_block = block[split_idx:]

        # recursively split again if either side is still too large
        final_blocks.extend(
            split_large_blocks(
                [left_block, right_block],
                sent_transformer,
                max_block_size,
                window_size,
            )
        )

    return final_blocks


def segment_by_similarity(
    sentences: list[str],
    sent_transformer: SentenceTransformer,
    window_size: int = 3,
    min_block_size: int = 50,
    percentile: float = 5,
) -> list[list[str]]:
    """Split ordered lecture sentences into topic blocks by
    similarity between neighboring windows of sentences.
    Compares natural topic boundaries in lecture."""

    # Edge case 1: no sentences
    if not sentences:
        return []

    # Edge case 2: few sentences
    if len(sentences) <= min_block_size:
        return [sentences]

    embeddings = sent_transformer.encode(sentences)

    similarities = []  # possible boundaries

    # Compare mean of window embeddings before and after boundary i
    for i in range(window_size, len(sentences) - window_size):
        left_window = embeddings[i - window_size : i]
        right_window = embeddings[i : i + window_size]

        left_mean = np.mean(left_window, axis=0).reshape(1, -1)
        right_mean = np.mean(right_window, axis=0).reshape(1, -1)

        similarity = cosine_similarity(left_mean, right_mean)[0][0]
        similarities.append((i, similarity))

    if not similarities:
        return [sentences]

    scores = [score for _, score in similarities]
    # Get cutiff for lowest percentile of similaries
    threshold = np.percentile(scores, percentile)

    # Keep if below similarity threshold
    candidate_boundaries = [idx for idx, score in similarities if score <= threshold]

    boundaries = []
    last_boundary = 0  # last accepted boundary

    for boundary in candidate_boundaries:
        # Enforce minimum block size
        if boundary - last_boundary >= min_block_size:
            boundaries.append(boundary)
            last_boundary = boundary

    # Build actual sentence,ordered blocks
    blocks = []
    start = 0

    for boundary in boundaries:
        blocks.append(sentences[start:boundary])
        start = boundary

    if start < len(sentences):
        blocks.append(sentences[start:])

    # split if larger than max allowed size
    blocks = split_large_blocks(
        blocks, sent_transformer, max_block_size=50, window_size=window_size
    )

    return blocks


if __name__ == "__main__":
    file_path = Path("data/sample/mit_18_650_full_lecture_1.txt")

    transcript = load_transcript(file_path)
    filtered = filter_transcript(transcript)
    sentences = split_into_sentences(filtered)

    # load sentence transformer
    sent_transformer = SentenceTransformer("all-MiniLM-L6-v2")

    (model, cluster_labels) = sentence_embedding_cluster_train(sentences, sent_transformer)

    print(f"K Chosen: {model.n_clusters}")

    groups = group_by_cluster(sentences, cluster_labels)

    for cluster_number, sentences_grouped in groups.items():
        print(f"Cluster {cluster_number}:")
        for ct, sentence in enumerate(sentences_grouped):
            print(f" - {sentence}")
            if ct >= 3:
                break
