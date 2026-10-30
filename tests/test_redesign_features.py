import json
from pathlib import Path
from app import (
    estimate_transcription_time,
    format_audio_duration,
    format_display_transcript,
    format_file_size,
    get_audio_format_label,
)
from audio_utils import ffprobe_duration, validate_audio_upload


def test_format_label_mapping():
    assert get_audio_format_label("test.mp3") == "MP3 Audio"
    assert get_audio_format_label("voice.wav") == "WAV Sound"
    assert get_audio_format_label("notes.m4a") == "M4A Audio"
    assert get_audio_format_label("recording.mpeg") == "MPEG Audio"
    assert get_audio_format_label("clip.flac") == "FLAC Audio"
    assert get_audio_format_label("sound.ogg") == "OGG Audio"


def test_format_file_size():
    assert format_file_size(500 * 1024) == "500.0 KB"
    assert format_file_size(5 * 1024 * 1024) == "5.00 MB"


def test_format_audio_duration():
    assert format_audio_duration(None) == "Unknown"
    assert format_audio_duration(0) == "Unknown"
    assert format_audio_duration(45) == "00:45"
    assert format_audio_duration(125) == "02:05"
    assert format_audio_duration(3665) == "01:01:05"


def test_estimate_transcription_time_dynamic():
    # Estimations are calculated dynamically based on duration and speed factor
    assert estimate_transcription_time(None) == "~10-20 sec"
    
    # Fast audio (e.g. 30 sec)
    est_30 = estimate_transcription_time(30)
    assert "~" in est_30 and "sec" in est_30
    
    # Measured RTF test (e.g. 100 sec audio with measured RTF of 0.1 -> 10 sec)
    est_measured = estimate_transcription_time(100, measured_rtf=0.1)
    assert est_measured == "~10 sec"
    
    # Long audio (e.g. 1200 sec audio with RTF of 0.2 -> 240 sec = ~4 min)
    est_long = estimate_transcription_time(1200, measured_rtf=0.2)
    assert est_long == "~4 min"


def test_format_display_transcript_plain_text():
    segments = [
        {"start": 0.0, "end": 2.5, "text": "This recording is copyright."},
        {"start": 4.0, "end": 8.0, "text": "In the IELTS test, you hear some recordings and you have to answer questions on them."},
        {"start": 10.0, "end": 14.0, "text": "You have time to read the instructions and questions."},
    ]
    formatted = format_display_transcript(segments)
    
    # Verify no timestamps or time markers
    assert "[" not in formatted
    assert "]" not in formatted
    assert "00:0" not in formatted
    
    # Verify all actual spoken words are preserved with proper paragraph breaks
    expected = (
        "This recording is copyright.\n\n"
        "In the IELTS test, you hear some recordings and you have to answer questions on them.\n\n"
        "You have time to read the instructions and questions."
    )
    assert formatted == expected


def test_empty_segments_format():
    assert format_display_transcript([]) == ""

