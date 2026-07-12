from pathlib import Path

def generate_notes(filtered_transcript: str) -> str:
    '''Converts cleaned transcript to a Markdown-formatted string'''
    placeholder_summary = 'This is a placeholder summary'
    heading = '# Lecture Notes\n\n'
    summary = f'## Raw Summary\n\n{placeholder_summary}\n\n'
    body = f'## Main Transcript\n\n{filtered_transcript}'
    return heading + summary + body

def save_notes(notes: str, output_path: Path) -> None:
    '''Ensures output folder exists and writes notes (Markdown string)
    to output Markdown File'''
    #create parent directories if neccessary, write notes to file
    output_path.parent.mkdir(parents=True, exist_ok=True)  
    output_path.write_text(notes, encoding='utf-8')
    