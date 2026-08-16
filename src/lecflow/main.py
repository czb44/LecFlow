from pathlib import Path
from .transcript import load_transcript, filter_transcript, split_into_sentences
from .data import load_data, split_sentences_labels
from .classification.data import label_key, load_model, load_embed_model
from .unit_type.rules import classify_unit_type
from .classification.predict import predict_labels, predict_labels_embed
from .notes import generate_notes, save_notes
from .topics.cluster import train_clusters, group_by_cluster, sentence_embedding_cluster_train, get_topic_labels



def main() -> None:
    file_path = Path("data/sample/lecture_2.txt")
    
    #Load and clean transcript
    transcript = load_transcript(file_path)
    filtered = filter_transcript(transcript)
    sentences = split_into_sentences(filtered)

    #Load sentence transformer
    sent_transformer = SentenceTransformer('all-MiniLM-L6-v2')

    #Use pre-trained LR classifier (embed) for housekeeping 
    embed_model = load_embed_model(Path('models'))
    embed_predictions = predict_labels_embed(embed_model, sentences, sent_transformer)

    content_sentences = [sentence for sentence, prediction in zip(sentences,embed_predictions) if prediction == 0]
    housekeeping_sentences = [sentence for sentence, prediction in zip(sentences,embed_predictions) if prediction == 1]

    (cluster_model, cluster_labels) = sentence_embedding_cluster_train(content_sentences, sent_transformer, k=4)
    blocks = group_by_cluster(content_sentences, cluster_labels)

    topic_labels = {}
    for cluster_num, group_of_sents in blocks.items():
        topic_labels[cluster_num] = get_topic_labels(group_of_sents, sent_transformer)

    notes = generate_notes(blocks, topic_labels, housekeeping_sentences)
    output_path = Path("outputs/notes/lecture_2_notes_v2_3.md")
    save_notes(notes, output_path) 

    print(f'Notes saved to: {output_path}')


if __name__ == '__main__':
    main()
