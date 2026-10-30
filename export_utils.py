import json
from transcript_manager import format_timestamp, full_text


def export_txt(segments) -> str:
    return full_text(segments) + "\n"


def export_srt(segments) -> str:
    lines = []
    for index, segment in enumerate(segments, 1):
        lines.extend([
            str(index),
            f"{format_timestamp(segment['start'], ',')} --> {format_timestamp(segment['end'], ',')}",
            segment["text"].strip(),
            "",
        ])
    return "\n".join(lines)


def export_vtt(segments) -> str:
    lines = ["WEBVTT", ""]
    for segment in segments:
        lines.extend([
            f"{format_timestamp(segment['start'], '.') } --> {format_timestamp(segment['end'], '.')}",
            segment["text"].strip(),
            "",
        ])
    return "\n".join(lines)


def export_json(segments, metadata: dict) -> str:
    payload = {"metadata": metadata, "segments": segments, "text": full_text(segments)}
    return json.dumps(payload, ensure_ascii=False, indent=2)
