import shutil
from typing import Callable, Optional

from faster_whisper import WhisperModel


def resolve_device(requested: str = "auto") -> tuple[str, str]:
    cuda_available = False
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
    except (ImportError, RuntimeError):
        pass
    device = "cuda" if requested == "cuda" and cuda_available else ("cuda" if requested == "auto" and cuda_available else "cpu")
    compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def load_model(model_size: str, device: str = "auto", compute_type: str = "auto"):
    device, default_compute = resolve_device(device)
    compute_type = default_compute if compute_type == "auto" else compute_type
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_file(path: str, model_size: str = "base", language: Optional[str] = None,
                    task: str = "transcribe", device: str = "auto",
                    compute_type: str = "auto", vad_filter: bool = True,
                    progress_callback: Optional[Callable[[str], None]] = None,
                    model_instance=None):
    resolved_device, resolved_compute = resolve_device(device)
    resolved_compute = resolved_compute if compute_type == "auto" else compute_type
    model = model_instance
    if model is None:
        try:
            model = load_model(model_size, resolved_device, resolved_compute)
        except (RuntimeError, ValueError) as exc:
            if resolved_device == "cuda":
                model = load_model(model_size, "cpu", "int8")
                resolved_device, resolved_compute = "cpu", "int8"
            else:
                raise RuntimeError(f"Could not load transcription model: {exc}") from exc

    segments, info = model.transcribe(
        path, language=None if language == "auto" else language,
        task=task, vad_filter=vad_filter,
        beam_size=5, word_timestamps=False,
    )
    output = []
    for segment in segments:
        output.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
            "confidence": None,
        })
        if progress_callback:
            progress_callback(segment.text.strip())
    if not output:
        raise RuntimeError("No speech segments were detected in the audio.")
    return output, {
        "language": info.language,
        "language_probability": info.language_probability,
        "device": resolved_device,
        "compute_type": resolved_compute,
        "model": model_size,
        "task": task,
    }
