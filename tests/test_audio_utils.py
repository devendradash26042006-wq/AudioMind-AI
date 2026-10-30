from audio_utils import validate_audio_upload


def test_valid_mpeg_upload():
    assert validate_audio_upload("voice.mpeg", 1024, 500) == (True, None)


def test_rejects_unknown_extension():
    valid, message = validate_audio_upload("voice.exe", 1024, 500)
    assert not valid and "Unsupported" in message


def test_rejects_empty_file():
    valid, message = validate_audio_upload("voice.wav", 0, 500)
    assert not valid and "empty" in message


def test_rejects_oversized_file():
    valid, message = validate_audio_upload("voice.wav", 501 * 1024 * 1024, 500)
    assert not valid and "limit" in message
