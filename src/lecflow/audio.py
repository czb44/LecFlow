from pathlib import Path
from faster_whisper import WhisperModel
import subprocess

def load_audio_model(model_size: str = 'base') -> WhisperModel:
    '''Load Whisper transcription model'''
    return WhisperModel(model_size, device='cpu', compute_type='int8') #int8 for speed

def get_audio_transcription(audio_path: Path, audio_model: WhisperModel) -> str:
    '''Transcribes audio file'''
    if not isinstance(audio_path, Path):
        raise TypeError("audio_path must be of type 'Path'")
    if not audio_path.is_file():
        raise FileNotFoundError("Audio file not found")
    
    allowed_types = {".wav", ".mp3", ".m4a"}
    if audio_path.suffix.lower() not in allowed_types:
        raise ValueError(f"Audio type '{audio_path.suffix}' is unsupported. Please convert to a supported type: '.wav', '.mp3', '.m4a'")
    
    segments, info = audio_model.transcribe(audio_path)

    cleaned_segments = [segment.text.strip() for segment in segments]
    text = ' '.join(cleaned_segments)

    if not text:
        raise ValueError('No speech detected in audio file. Inspect audio file then try again')

    return text

def video_to_audio(video_path: Path, audio_output_path: Path) -> None:
    '''Extract audio from video file with ffmeg'''
    if not isinstance(video_path, Path):
        raise TypeError("video_path must be of type 'Path'")
    if not video_path.is_file():
        raise FileNotFoundError("Video file not found")
    
    allowed_types = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    if video_path.suffix.lower() not in allowed_types:
        raise ValueError(f"Video type '{video_path.suffix}' is unsupported. Please convert to a supported type: '.mp4', '.mov', '.mkv', '.avi', '.webm'")
    
    audio_output_path.parent.mkdir(parents=True, exist_ok=True) #create missing directories

    subprocess.run(['ffmpeg', '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', str(audio_output_path), '-y'], check=True, capture_output=True) #overwrite, raise exception w/ error

if __name__ == '__main__':
    video_path = Path('data/sample/short_test_video.MOV')
    extracted_audio_path = Path("data/sample/short_test_video_audio.wav")

    audio_model = load_audio_model()
    video_to_audio(video_path, extracted_audio_path)
    text = get_audio_transcription(extracted_audio_path, audio_model)
    
    print(text)
