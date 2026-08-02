from pathlib import Path
from .transcript import load_transcript, filter_transcript, split_into_sentences
from .classification.data import label_key, load_model, load_data, split_sentences_labels
from .classification.predict import predict_labels
from .classification.train import train_classifier_embeddings
from .notes import generate_notes, save_notes
from .chunking import group_adjacent
from .topics.cluster import train_clusters, group_by_cluster, sentence_embedding_cluster_train





def main() -> None:
    file_path = Path("data/sample/lecture_2.txt")
    
    transcript = load_transcript(file_path)
    filtered = filter_transcript(transcript)
    sentences = split_into_sentences(filtered)


    # Test embeddings for housekeeping classification
    df2 = load_data(Path('data/labeled/lecflow_contentvhousekeeping_data.csv'))
    sentences2, labels2 = split_sentences_labels(df2)
    embedding_model, embedding_transformer, (x_test2, y_test2) = train_classifier_embeddings(sentences2, labels2)

    # Apply to real transcript sentences (already loaded as `sentences`)
    sentences_embedded = embedding_transformer.encode(sentences)
    embed_predictions = embedding_model.predict(sentences_embedded).tolist()

    model, vectorizer = load_model(Path('models'))
    predictions = predict_labels(model, vectorizer, sentences)
    
    # Compare against existing TF-IDF predictions
    for sentence, tfidf_pred, embed_pred in zip(sentences, predictions, embed_predictions):
        print(f'TF-IDF: {label_key[tfidf_pred]} | Embeddings: {label_key[embed_pred]} | {sentence}')


    content_sentences = [sentence for sentence, prediction in zip(sentences,predictions) if prediction == 0]
    housekeeping_sentences = [sentence for sentence, prediction in zip(sentences,predictions) if prediction == 1]
    content_text = ' '.join(content_sentences)

    
    for k in range(4, 5):
        (cluster_model, cluster_labels) = sentence_embedding_cluster_train(content_sentences, k)
        blocks = group_adjacent(content_sentences, cluster_labels)

        # print(f'K = {k}')
        # for i, block in enumerate(blocks):
        #     print(f'Block {i}:')
        #     for sentence in block:
        #         print(f' - {sentence}')
        # print('\n--------------------------\n')


    notes = generate_notes(blocks, housekeeping_sentences)
    output_path = Path("outputs/notes/lecture_2_notes.md")
    save_notes(notes, output_path) 

    # print(f'Notes saved to: {output_path}')


if __name__ == '__main__':
    main()
