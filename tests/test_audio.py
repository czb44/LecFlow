from pathlib import Path

from lecflow.audio import get_audio_transcription, video_to_audio


def test_video_to_audio(tmp_path):
    video_path = Path("tests/assets/short_test_video.MOV")
    extracted_audio_path = tmp_path / "test_audio_extract.wav"
    video_to_audio(video_path, extracted_audio_path)
    assert extracted_audio_path.exists()


# Helper function to normalize whisper outputs
def normalize(text):
    return (
        text.lower()
        .replace(".", "")
        .replace("?", "")
        .replace(",", "")
        .replace("!", "")
        .replace("'", "")
        .replace("`", "")
        .strip()
    )


def test_get_audio_transcription(audio_model):
    text = "Today's lecture will be going over the theorem probabilities."
    audio_path = Path("tests/assets/short_test_audio.m4a")
    result = get_audio_transcription(audio_path, audio_model)
    assert normalize(result) == normalize(text)
