from pathlib import Path

def load_transcript(file_path: Path) -> str:
    ''' Loads transcript and extracts its contents'''
    if file_path.exists():
        return file_path.read_text(encoding='utf-8') #read all the same
    else:
        raise FileNotFoundError('File cannot be found. Verify path is correct')

def filter_transcript(transcript: str) -> str:
    '''Formats extra whitespace in transcript with consistency'''
    trans_list = transcript.split() 
    return ' '.join(trans_list)
