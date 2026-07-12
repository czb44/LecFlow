from pathlib import Path
#import transcript processing founctions from transcript.py
from .transcript import load_transcript, filter_transcript
from .notes import generate_notes, save_notes



def main() -> None:
    file_path = Path("data/sample/lecture_1.txt")
    output_path = Path("outputs/notes/lecture_1_notes.md")
    
    transcript = load_transcript(file_path)
    filtered = filter_transcript(transcript)
    notes = generate_notes(filtered)
    save_notes(notes, output_path) 

    print(f'Notes saved to: {output_path}')


if __name__ == '__main__':
    main()
