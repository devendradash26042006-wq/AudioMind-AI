from export_utils import export_json, export_srt, export_txt, export_vtt
from transcript_manager import format_timestamp


SEGMENTS = [{"start": 0.0, "end": 2.5, "text": "Hello world"}]


def test_txt_export():
    assert export_txt(SEGMENTS) == "Hello world\n"


def test_srt_export():
    output = export_srt(SEGMENTS)
    assert "00:00:00,000 --> 00:00:02,500" in output


def test_vtt_export():
    output = export_vtt(SEGMENTS)
    assert output
    assert output.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.500" in output


def test_json_export():
    output = export_json(SEGMENTS, {"language": "en"})
    assert '"language": "en"' in output
    assert '"text": "Hello world"' in output
