from pathlib import Path
from .transcript import load_transcript, filter_transcript, split_into_sentences
from .classification.data import label_key, load_model, load_embed_model, load_data, split_sentences_labels
from .classification.predict import predict_labels, predict_labels_embed
from .notes import generate_notes, save_notes
from .topics.cluster import train_clusters, group_by_cluster, sentence_embedding_cluster_train


def main() -> None:
    file_path = Path("data/sample/lecture_2.txt")
    
    transcript = load_transcript(file_path)
    filtered = filter_transcript(transcript)
    sentences = split_into_sentences(filtered)

    # #TF-IDF Housekeeping model
    # model, vectorizer = load_model(Path('models'))
    # predictions = predict_labels(model, vectorizer, sentences)

    #Sentence-transformer embedding model
    embed_model = load_embed_model(Path('models'))
    embed_predictions = predict_labels_embed(embed_model, sentences)

    
    # # Compare sentence-transformer embeddings against TF-IDF
    # for sentence, tfidf_pred, embed_pred in zip(sentences, predictions, embed_predictions):
    #     print(f'TF-IDF: {label_key[tfidf_pred]} | Embeddings: {label_key[embed_pred]} | {sentence}')

    #Use embedding model for classification of housekeeping vs content
    content_sentences = [sentence for sentence, prediction in zip(sentences,embed_predictions) if prediction == 0]
    housekeeping_sentences = [sentence for sentence, prediction in zip(sentences,embed_predictions) if prediction == 1]
    
    
    for k in range(4, 5): #loop to test different k-values
        (cluster_model, cluster_labels) = sentence_embedding_cluster_train(content_sentences, k)
        blocks = group_by_cluster(content_sentences, cluster_labels)

        # print(f'K = {k}')
        # for i, block in enumerate(blocks):
        #     print(f'Block {i}:')
        #     for sentence in block:
        #         print(f' - {sentence}')
        # print('\n--------------------------\n')


    notes = generate_notes(blocks, housekeeping_sentences)
    output_path = Path("outputs/notes/lecture_2_notes.md")
    save_notes(notes, output_path) 

    print(f'Notes saved to: {output_path}')


if __name__ == '__main__':
    main()
