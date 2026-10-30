import pytest
from transcript_manager import format_timestamp, full_text, normalized_segments


def test_format_timestamp():
    assert format_timestamp(65.5, ".") == "00:01:05.500"


def test_full_text_preserves_segment_lines():
    assert full_text([{"text": "One"}, {"text": ""}, {"text": "Two"}]) == "One\nTwo"


def test_normalized_segments_accept_metadata():
    segments = normalized_segments([{"start": 1, "end": 2, "text": " hello "}])
    assert segments == [{"start": 1.0, "end": 2.0, "text": "hello", "confidence": None}]
