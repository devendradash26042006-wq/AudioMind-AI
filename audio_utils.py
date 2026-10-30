import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".mpeg", ".mpg", ".mp4", ".flac",
    ".ogg", ".oga", ".opus", ".webm", ".aac", ".wma", ".aiff", ".amr",
}


def validate_audio_upload(filename: str, size_bytes: int, max_mb: int) -> tuple[bool, Optional[str]]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file type '{suffix or '(none)'}'. Supported types include MP3, WAV, M4A, MPEG, MP4, FLAC, and OGG."
    if size_bytes <= 0:
        return False, "The uploaded file is empty."
    if size_bytes > max_mb * 1024 * 1024:
        return False, f"File exceeds the {max_mb} MB upload limit."
    return True, None


def ffprobe_duration(path: str) -> Optional[float]:
    if shutil.which("ffprobe"):
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=20, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass

    # Fallback to PyAV (bundled with faster-whisper)
    try:
        import av
        with av.open(path) as container:
            if container.duration is not None and av.time_base:
                return float(container.duration) / float(av.time_base)
            # Try audio stream duration
            for stream in container.streams.audio:
                if stream.duration and stream.time_base:
                    return float(stream.duration * stream.time_base)
    except Exception:
        pass

    # Fallback to wave for WAV files
    try:
        import wave
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return float(frames) / float(rate)
    except Exception:
        pass

    return None


def save_upload(uploaded_file, destination_dir: Optional[str] = None) -> str:
    directory = destination_dir or tempfile.mkdtemp(prefix="audiomind_")
    os.makedirs(directory, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    destination = os.path.join(directory, safe_name)
    with open(destination, "wb") as output:
        output.write(uploaded_file.getbuffer())
    return destination


def convert_for_browser(path: str) -> str:
    native_browser_exts = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".webm", ".mp4"}
    ext = os.path.splitext(path)[1].lower()

    if shutil.which("ffmpeg"):
        output = os.path.splitext(path)[0] + ".converted.mp3"
        result = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", path, "-vn", "-ar", "16000", "-ac", "1", output],
            capture_output=True, timeout=300, check=False,
        )
        if result.returncode == 0 and os.path.exists(output):
            return output

    # PyAV fallback for formats browsers can't play natively
    if ext not in native_browser_exts:
        try:
            import av
            output = os.path.splitext(path)[0] + ".converted.wav"
            with av.open(path) as in_c:
                if in_c.streams.audio:
                    in_stream = in_c.streams.audio[0]
                    with av.open(output, "w", "wav") as out_c:
                        out_stream = out_c.add_stream("pcm_s16le", rate=16000, layout="mono")
                        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
                        for frame in in_c.decode(in_stream):
                            for resampled in resampler.resample(frame):
                                for packet in out_stream.encode(resampled):
                                    out_c.mux(packet)
                        for packet in out_stream.encode(None):
                            out_c.mux(packet)
                    if os.path.exists(output):
                        return output
        except Exception:
            pass

    return path
