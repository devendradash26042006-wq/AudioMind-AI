from typing import Iterable


def format_timestamp(seconds: float, decimal_separator: str = ",") -> str:
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        secs += 1
        millis = 0
    if secs == 60:
        mins, secs = divmod((secs + minutes), 60)
        minutes = mins
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_separator}{millis:03d}"


def full_text(segments: Iterable[dict]) -> str:
    return "\n".join(str(segment.get("text", "")).strip() for segment in segments if str(segment.get("text", "")).strip())


def normalized_segments(raw_segments) -> list[dict]:
    result = []
    for segment in raw_segments:
        if hasattr(segment, "start"):
            text = getattr(segment, "text", "")
            start = float(segment.start)
            end = float(segment.end)
            confidence = None
        else:
            text = segment.get("text", "")
            start = float(segment.get("start", 0))
            end = float(segment.get("end", start))
            confidence = segment.get("confidence")
        result.append({"start": start, "end": end, "text": text.strip(), "confidence": confidence})
    return result
