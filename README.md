# LecFlow
A self-contained lecture-to-study pipeline

## Description
LecFlow is a local-first Python app that converts lecture transcripts or recordings into organized notes, extracting searchable concepts, generating strategic review questions, and equipping students with the tools necessary for success. LecFlow does this all completely free and privacy-friendly - no API calls, no charges. 

## Pipeline Overview
lecture recording/transcript → organized notes → searchable concepts → review questions → better academic performance


## Objectives
1. Accept a lecture transcript or recording.
2. Clean and process the lecture.
3. Break it into meaningful chunks.
4. Classify chunks into concepts, examples, housekeeping, definitions, questions, etc.
5. Generate structured Markdown study notes.
6. Store lectures, transcripts, chunks, and notes locally.
7. Search across lectures.
8. Eventually transcribe audio locally.
9. Generate basic review questions.
10. Evaluate search/classification quality.
11. Present everything in a Streamlit UI.

## Project Plan
MVP: Transcript --> Notes

Gather Info:
- transcript.txt first
- later audio files
- later optional slides/PDFs


Process: 
- Clean tanscript
- Split into chunks
- Identify topics
- Group as concept,example, question etc.
- store transcript and chunks

Study:
- generate structured notes
- generate review questions
- search previous lectures
- track weak concepts

Later Versions:
- Audio
- Be able to accept different file types / encodings
- Optional Slides/PDFs
- Chatbot (maybe)


## Implementation Plan
1. Transcript Notes Generator
2. Chunk and classify transcript (concept, examples, etc.)
3. Streamlit UI (file upload, view notes)
4. SQLite Storage (save lectures, transcripts, notes, chunks)
5. Search across lectures
6. Audio transcription (local) - upload audio and transcribe locally
7. Evaluation (measure usefullness)


## Status: WIP
