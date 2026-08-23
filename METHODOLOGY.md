# Methodology

## Overview
- Baseline-first approach: start with simple foundation, test on lecture data, and add complexity when results justify it.
- Core processing: ML, NLP, and rule-based approaches rather than using an LLM for every task. 

## Housekeeping Classification

- Baseline: **TF-IDF + Logistic Regression**
    - 85.4% accuracy on a 96-sentence benchmark.
    - Errors occurred mostly on housekeeping sentences without obvious logistical terminology.
- Comparison: **Sentence Embeddings + Logistic Regression**
    - Replacing TF-IDF with `all-MiniLM-L6-v2` embeddings increased accuracy to 97.9% and reduced errors (14 to 2).
    - Retrained with reviewed examples from college lecture transcripts.
    - Final results on hold-out set: 191 sentences from three unseen college lectures.
        - TF-IDF: 90.1% accuracy, 36% housekeeping recall, 46% housekeeping F1.
        - Embeddings: 92.7% accuracy, 77% housekeeping recall, 71% housekeeping F1.

*Production decision: sentence embeddings + logistic regression.*

## Topic Segmentation

### K-Means
- Baseline: **TF-IDF + K-Means**
    - Results: ~18 blocks from ~35 content sentences; often separated related sentences.
- Second approach: **Sentence Embeddings + K-Means**
    - Results: ~16 blocks and better semantic grouping on same test set.
    - Experiment: fixed `k`, silhouette-based adaptive `k`, global grouping, and adjacent grouping.
        - Adaptive `k` improved handling for various lecture lengths, but K-Means still ignored lecture order.
        - On a full college probability lecture (~1.5hrs) testing adaptive K-Means with chronological grouping resulted in `k=14` and 619 fragmented blocks.

### Sequential Segmentation
- Replacement: **Neighboring-Window Cosine Similarity**
    - Compares embeddings of adjacent sentence windows to detect topic changes.
    - Minimum block sizes reduce fragmentation; oversized blocks are split at their strongest local boundary.
    - Preserved chronology, creating a reasonable number of blocks for a lecture.

*Production decision: neighboring-window cosine similarity.*

## Topic Labeling

- Baseline: **NLP + Embedding Centroid Matching**
    - Extract noun-phrase candidates (spaCy).
    - Compare candidate embeddings versus mean embedding for each block.
    - Choose closest candidate as the topic label.
    - Duplicate labels are numbered to keep headings and table-of-contents entries unique.

*Production decision: embedding centroid + noun-phrase matching.*

## Unit Type Classification

- Baseline: **Rule-Based Classification**
    - Regex and phrase matching for definitions, examples, and questions; unmatched sentences default to explanations.
- Comparison: **Sentence Embeddings + Logistic Regression**
    - First model: ~97% held-out accuracy. Testing showed over-prediction of questions.
    - Second model: 75% held-out accuracy and 44% agreement with rules; more varied training data resulted in decreased question prediction & over-prediction of examples.
    - Review of disagreements generally favored the rule-based approach.

*Production decision: retain rule-based classifier.*

## Local LLM Refinement

- Determining whether academic content or housekeeping is useful requires more context than previously used classifiers capture.
- Qwen run locally through Ollama, keeping processing local and free
    - Testing showed more relevant notes, with increased latency and occasional prompt-adherence errors.

*Production decision: retain targeted LLM refinement*
