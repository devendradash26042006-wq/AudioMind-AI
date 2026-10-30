# AudioMind AI — Transcribe · Understand · Learn

AudioMind AI is a modern SaaS audio transcription dashboard built with Streamlit and powered locally by `faster-whisper`. It delivers fast, accurate speech-to-text with segment timestamps, an interactive audio player, client-side copy & share tools, inline transcript editing, and text export capabilities.

---

## Features

- **Modern SaaS Dashboard**: Clean, responsive light theme with soft blue & purple gradients, rounded cards, and professional typography.
- **Drag-and-Drop Audio Upload**: Support for MP3, WAV, M4A, MPEG, MP4, FLAC, OGG, Opus, WebM, AAC, WMA, and AIFF (up to 500 MB).
- **Dynamic File Metrics**: Live extraction of filename, audio format, file size, audio duration, and calculated estimated transcription time.
- **Embedded Audio Player**: Full-featured player with custom play/pause, seek progress bar with gradient fill, elapsed & total time display, and volume slider with mute toggle.
- **Faster-Whisper AI Engine**: High-performance local Whisper transcription with automatic device detection (CPU/CUDA) and VAD filtering.
- **Timestamped Transcripts**: Segment-by-segment display with formatted timestamps (`[MM:SS]`).
- **Interactive Actions**:
  - **Copy Text**: One-click clipboard copy with animated notification toast.
  - **Share Text**: Native browser Web Share API integration with automatic clipboard fallback.
  - **Edit Text**: In-place transcript editor with Save Changes and Cancel controls.
  - **Download Text File**: UTF-8 `.txt` export containing the latest transcript and timestamps, dynamically named from the audio file.

---

## Project Structure

```text
AudioMind-AI/
├── .streamlit/
│   └── config.toml           # Streamlit theme and upload size configuration
├── tests/
│   ├── test_audio_utils.py
│   ├── test_core_workflow.py
│   ├── test_export_utils.py
│   ├── test_redesign_features.py
│   └── test_transcript_manager.py
├── app.py                    # Main Streamlit dashboard application
├── audio_utils.py            # Audio validation, saving, duration & browser conversion
├── chat_engine.py            # Optional Q&A engine
├── concept_analyzer.py       # Concept extraction utilities
├── config.py                 # Application configuration & settings
├── export_utils.py           # TXT, SRT, VTT, and JSON exporters
├── llm_provider.py           # Optional LLM integration
├── requirements.txt          # Python dependencies
├── study_tools.py            # Flashcard and study tool generators
├── transcription.py          # Faster-Whisper model loading and transcription
└── transcript_manager.py     # Timestamp formatting and segment helpers
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- (Optional) NVIDIA GPU with CUDA for accelerated inference.

### 2. Installation

Clone the repository:
```powershell
git clone https://github.com/devendradash26042006-wq/AudioMind-AI.git
cd AudioMind-AI
```

Create a virtual environment and install dependencies:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the Application

```powershell
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

---

## Running Tests

Run the full pytest test suite (21 unit tests):
```powershell
python -m pytest -v
```

---

## License

MIT License.
