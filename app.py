import base64
import json
import os
import time
from pathlib import Path
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

try:
    import torch
except ImportError:
    torch = None

from audio_utils import convert_for_browser, ffprobe_duration, save_upload, validate_audio_upload
from config import SETTINGS
from export_utils import export_json, export_srt, export_txt, export_vtt
from transcript_manager import format_timestamp, full_text

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="AudioMind AI — Transcribe · Understand · Learn",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# Session State Initialization
# ==============================================================================
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None
if "uploaded_size" not in st.session_state:
    st.session_state.uploaded_size = 0
if "segments" not in st.session_state:
    st.session_state.segments = []
if "metadata" not in st.session_state:
    st.session_state.metadata = {}
if "edited_transcript" not in st.session_state:
    st.session_state.edited_transcript = None
if "is_editing" not in st.session_state:
    st.session_state.is_editing = False
if "measured_rtf" not in st.session_state:
    st.session_state.measured_rtf = None


def reset_workspace():
    """Resets all session state variables when a new file is uploaded or cleared."""
    st.session_state.audio_path = None
    st.session_state.uploaded_name = None
    st.session_state.uploaded_size = 0
    st.session_state.segments = []
    st.session_state.metadata = {}
    st.session_state.edited_transcript = None
    st.session_state.is_editing = False


# ==============================================================================
# Helper Functions
# ==============================================================================
@st.cache_resource
def load_transcription_model(model_size: str, device: str, compute_type: str):
    from transcription import load_model
    return load_model(model_size, device, compute_type)


def get_audio_format_label(filename: str) -> str:
    suffix = Path(filename).suffix.lower().lstrip(".")
    mapping = {
        "mp3": "MP3 Audio",
        "wav": "WAV Sound",
        "m4a": "M4A Audio",
        "mpeg": "MPEG Audio",
        "mpg": "MPEG Audio",
        "mp4": "MP4 Audio",
        "flac": "FLAC Audio",
        "ogg": "OGG Audio",
        "oga": "OGG Audio",
        "opus": "Opus Audio",
        "webm": "WebM Audio",
        "aac": "AAC Audio",
        "wma": "WMA Audio",
        "aiff": "AIFF Audio",
        "amr": "AMR Audio",
    }
    return mapping.get(suffix, f"{suffix.upper()} Audio" if suffix else "Audio File")


def format_audio_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds <= 0:
        return "Unknown"
    total_sec = int(round(seconds))
    hours, remainder = divmod(total_sec, 3600)
    mins, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def estimate_transcription_time(duration_sec: Optional[float], measured_rtf: Optional[float] = None) -> str:
    if duration_sec is None or duration_sec <= 0:
        return "~10-20 sec"
    if measured_rtf is not None and measured_rtf > 0:
        rtf = measured_rtf
    else:
        has_cuda = bool(torch and torch.cuda.is_available())
        rtf = 0.08 if has_cuda else 0.22
    est_sec = max(2, round(duration_sec * rtf))
    if est_sec < 60:
        return f"~{est_sec} sec"
    mins, secs = divmod(est_sec, 60)
    return f"~{mins}m {secs}s" if secs > 0 else f"~{mins} min"


def format_display_transcript(segments: list[dict]) -> str:
    """Formats segments into clean, continuous, readable paragraphs without timestamps or time markers."""
    if not segments:
        return ""
    paragraphs = []
    for s in segments:
        text = str(s.get("text", "")).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def get_active_transcript() -> str:
    """Returns the current transcript text (latest edited or formatted clean segments) without timestamps."""
    if st.session_state.edited_transcript is not None:
        return st.session_state.edited_transcript.strip()
    if st.session_state.segments:
        return format_display_transcript(st.session_state.segments)
    return ""


# ==============================================================================
# CSS & Design System Injection
# ==============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }

    /* Hide default Streamlit chrome */
    header[data-testid="stHeader"] { display: none !important; }
    #MainMenu, footer { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }

    /* Main Container Spacing */
    .block-container {
        max-width: 1200px !important;
        padding-top: 1.25rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        margin: 0 auto !important;
    }

    /* Header Bar */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 16px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04);
    }
    .header-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .logo-badge {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.28);
    }
    .brand-title {
        font-size: 22px;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .brand-subtitle {
        font-size: 13px;
        font-weight: 500;
        color: #64748B;
        letter-spacing: -0.01em;
    }
    .header-actions {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .theme-pill {
        display: flex;
        align-items: center;
        gap: 6px;
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        color: #334155;
    }
    .icon-btn {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #475569;
        transition: all 0.2s ease;
    }
    .icon-btn:hover {
        background: #EEF2FF;
        border-color: #C7D2FE;
        color: #4F46E5;
    }

    /* Common Card Styling */
    .dashboard-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .card-header-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
    }
    .card-icon-container {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: #EEF2FF;
        border: 1px solid #E0E7FF;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #4F46E5;
    }
    .card-heading {
        font-size: 18px;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.01em;
        margin: 0;
    }
    .card-subheading {
        font-size: 13px;
        color: #64748B;
        margin: 0;
    }

    /* Upload Area Styling */
    .upload-zone-wrapper {
        border: 2px dashed #CBD5E1;
        border-radius: 14px;
        background: #FAFAFC;
        padding: 24px 20px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .upload-zone-wrapper:hover {
        border-color: #6366F1;
        background: #F5F7FF;
    }
    .format-pill {
        display: inline-block;
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        color: #475569;
        margin-top: 8px;
    }

    /* Style Streamlit File Uploader to integrate seamlessly */
    [data-testid="stFileUploader"] {
        margin-top: 8px;
    }
    [data-testid="stFileUploader"] > section {
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
    }
    [data-testid="stFileUploader"] button {
        background: #FFFFFF !important;
        color: #4F46E5 !important;
        border: 1.5px solid #C7D2FE !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 6px 18px !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background: #EEF2FF !important;
        border-color: #4F46E5 !important;
    }

    /* File Details Card */
    .detail-item-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 11px 14px;
        border-radius: 10px;
        background: #F8FAFC;
        border: 1px solid #F1F5F9;
        margin-bottom: 9px;
    }
    .detail-label-group {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #475569;
        font-size: 13px;
        font-weight: 600;
    }
    .detail-val-group {
        font-size: 13px;
        font-weight: 700;
        color: #0F172A;
    }
    .format-badge {
        background: #EEF2FF;
        color: #4F46E5;
        border: 1px solid #C7D2FE;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
    }

    /* Button Styling */
    div.stButton > button[kind="primary"], .btn-transcribe {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
        transition: all 0.2s ease !important;
        height: 48px !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4338CA 0%, #6D28D9 100%) !important;
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.42) !important;
        transform: translateY(-1px) !important;
    }

    /* Green Download Button */
    div[data-testid="stDownloadButton"] > button {
        background: #ECFDF5 !important;
        color: #047857 !important;
        border: 1.5px solid #A7F3D0 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        padding: 12px 24px !important;
        height: 48px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stDownloadButton"] > button:hover:not(:disabled) {
        background: #D1FAE5 !important;
        border-color: #6EE7B7 !important;
        color: #065F46 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2) !important;
    }
    div[data-testid="stDownloadButton"] > button:disabled {
        background: #F8FAFC !important;
        color: #94A3B8 !important;
        border-color: #E2E8F0 !important;
        cursor: not-allowed !important;
        opacity: 0.65 !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* Secondary Action Buttons */
    div.stButton > button[kind="secondary"] {
        background: #FFFFFF !important;
        color: #334155 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 7px 16px !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: #6366F1 !important;
        color: #4F46E5 !important;
        background: #F8FAFC !important;
    }

    /* Full-width Cards */
    .full-width-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
        margin-top: 24px;
    }

    /* Empty States */
    .empty-state-box {
        text-align: center;
        padding: 48px 24px;
        color: #64748B;
    }
    .empty-icon-circle {
        width: 58px;
        height: 58px;
        border-radius: 50%;
        background: #F1F5F9;
        color: #94A3B8;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 14px auto;
    }
    .empty-title {
        font-size: 16px;
        font-weight: 700;
        color: #334155;
        margin-bottom: 6px;
    }
    .empty-desc {
        font-size: 13px;
        color: #94A3B8;
        max-width: 440px;
        margin: 0 auto;
    }

    /* Transcript Box — Clean, continuous, readable paragraphs */
    .transcript-container {
        max-height: 480px;
        overflow-y: auto;
        padding: 20px 24px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        margin-top: 14px;
        scrollbar-width: thin;
        scrollbar-color: #CBD5E1 #F1F5F9;
    }
    .transcript-container::-webkit-scrollbar {
        width: 6px;
    }
    .transcript-container::-webkit-scrollbar-track {
        background: #F1F5F9;
        border-radius: 6px;
    }
    .transcript-container::-webkit-scrollbar-thumb {
        background: #CBD5E1;
        border-radius: 6px;
    }
    .transcript-container::-webkit-scrollbar-thumb:hover {
        background: #94A3B8;
    }
    .transcript-paragraph {
        font-size: 15px;
        line-height: 1.75;
        color: #1E293B;
        margin-bottom: 16px;
        letter-spacing: -0.01em;
    }
    .transcript-paragraph:last-child {
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# SECTION 3: HEADER SECTION
# ==============================================================================
st.markdown(
    """
    <div class="app-header">
        <div class="header-brand">
            <div class="logo-badge">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 2v20M17 5v14M7 5v14M22 10v4M2 10v4"/>
                </svg>
            </div>
            <div>
                <div class="brand-title">AudioMind AI</div>
                <div class="brand-subtitle">Transcribe · Understand · Learn</div>
            </div>
        </div>
        <div class="header-actions">
            <div class="theme-pill" title="Active Theme">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="5"/>
                    <line x1="12" y1="1" x2="12" y2="3"/>
                    <line x1="12" y1="21" x2="12" y2="23"/>
                    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
                    <line x1="1" y1="12" x2="3" y2="12"/>
                    <line x1="21" y1="12" x2="23" y2="12"/>
                    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
                </svg>
                <span>Light</span>
            </div>
            <div class="icon-btn" title="Supported: MP3, WAV, M4A, MPEG, MP4, FLAC, OGG. Faster-Whisper base model.">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
            </div>
            <div class="icon-btn" title="User Profile">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                    <circle cx="12" cy="7" r="4"/>
                </svg>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# SECTION 4: MAIN DASHBOARD LAYOUT (UPPER TWO COLUMNS)
# ==============================================================================
col_upload, col_details = st.columns([1, 1], gap="large")

# ------------------------------------------------------------------------------
# LEFT COLUMN: AUDIO UPLOAD
# ------------------------------------------------------------------------------
with col_upload:
    st.markdown(
        """
        <div class="dashboard-card">
            <div class="card-header-row">
                <div class="card-icon-container">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="17 8 12 3 7 8"/>
                        <line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                </div>
                <div>
                    <h2 class="card-heading">Upload Your Audio File</h2>
                    <p class="card-subheading">Drag & drop an audio file here, or click to browse</p>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload audio file",
        type=[
            "mp3", "wav", "m4a", "mpeg", "mpg", "mp4", "flac", "ogg",
            "oga", "opus", "webm", "aac", "wma", "aiff", "amr",
        ],
        label_visibility="collapsed",
        key="main_audio_uploader",
    )

    if uploaded_file is not None:
        valid, error_msg = validate_audio_upload(
            uploaded_file.name,
            uploaded_file.size,
            SETTINGS.max_upload_mb,
        )
        if not valid:
            st.error(error_msg, icon="⚠️")
        else:
            if st.session_state.uploaded_name != uploaded_file.name:
                reset_workspace()
                st.session_state.audio_path = save_upload(uploaded_file)
                st.session_state.uploaded_name = uploaded_file.name
                st.session_state.uploaded_size = uploaded_file.size
                st.rerun()

            st.markdown(
                f"""
                <div style="margin-top: 12px; padding: 12px 14px; background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 10px; display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 10px; overflow: hidden;">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#4F46E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M9 18V5l12-2v13"/>
                            <circle cx="6" cy="18" r="3"/>
                            <circle cx="18" cy="16" r="3"/>
                        </svg>
                        <span style="font-size: 13px; font-weight: 700; color: #1E293B; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                            {uploaded_file.name}
                        </span>
                    </div>
                    <span style="font-size: 12px; font-weight: 600; color: #4F46E5; background: #FFFFFF; padding: 2px 8px; border-radius: 6px; border: 1px solid #C7D2FE;">
                        Ready
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("✕ Remove / Replace File", key="btn_remove_file", use_container_width=True):
                reset_workspace()
                st.rerun()

    st.markdown(
        """
            <div style="margin-top: auto; padding-top: 14px; text-align: center;">
                <span class="format-pill">Supported: MP3, WAV, M4A, MPEG, MP4, FLAC, OGG & more</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------------------
# RIGHT COLUMN: UPLOADED FILE DETAILS
# ------------------------------------------------------------------------------
with col_details:
    st.markdown(
        """
        <div class="dashboard-card">
            <div class="card-header-row">
                <div class="card-icon-container" style="background: #F5F3FF; border-color: #EDE9FE; color: #7C3AED;">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                        <line x1="16" y1="13" x2="8" y2="13"/>
                        <line x1="16" y1="17" x2="8" y2="17"/>
                        <polyline points="10 9 9 9 8 9"/>
                    </svg>
                </div>
                <div>
                    <h2 class="card-heading">Uploaded File Details</h2>
                    <p class="card-subheading">Dynamic audio metrics and processing estimate</p>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
        duration_sec = ffprobe_duration(st.session_state.audio_path)
        format_label = get_audio_format_label(st.session_state.uploaded_name)
        size_label = format_file_size(st.session_state.uploaded_size)
        duration_label = format_audio_duration(duration_sec)
        est_time_label = estimate_transcription_time(duration_sec, st.session_state.measured_rtf)

        st.markdown(
            f"""
            <div class="detail-item-row">
                <div class="detail-label-group">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#4F46E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>
                    </svg>
                    <span>File Name</span>
                </div>
                <div class="detail-val-group" style="max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{st.session_state.uploaded_name}">
                    {st.session_state.uploaded_name}
                </div>
            </div>

            <div class="detail-item-row">
                <div class="detail-label-group">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#7C3AED" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="12 2 2 7 12 12 22 7 12 2"/>
                        <polyline points="2 17 12 22 22 17"/>
                        <polyline points="2 12 12 17 22 12"/>
                    </svg>
                    <span>File Type</span>
                </div>
                <div class="detail-val-group">
                    <span class="format-badge">{format_label}</span>
                </div>
            </div>

            <div class="detail-item-row">
                <div class="detail-label-group">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#0284C7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
                    </svg>
                    <span>File Size</span>
                </div>
                <div class="detail-val-group">{size_label}</div>
            </div>

            <div class="detail-item-row">
                <div class="detail-label-group">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#E11D48" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
                    </svg>
                    <span>Audio Duration</span>
                </div>
                <div class="detail-val-group">{duration_label}</div>
            </div>

            <div class="detail-item-row">
                <div class="detail-label-group">
                    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#D97706" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
                    </svg>
                    <span>Estimated Transcription Time</span>
                </div>
                <div class="detail-val-group" style="color: #4F46E5;">{est_time_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

        col_btn_transcribe, col_btn_download = st.columns([1, 1], gap="small")

        with col_btn_transcribe:
            if st.button("▶ Transcribe Audio", type="primary", use_container_width=True, key="btn_transcribe_audio"):
                with st.spinner("Transcribing audio with Whisper AI..."):
                    started = time.perf_counter()
                    try:
                        model = load_transcription_model(
                            SETTINGS.transcription_model,
                            SETTINGS.transcription_device,
                            SETTINGS.transcription_compute_type,
                        )
                        from transcription import transcribe_file

                        segments, metadata = transcribe_file(
                            st.session_state.audio_path,
                            model_size=SETTINGS.transcription_model,
                            language=None,
                            task="transcribe",
                            device=SETTINGS.transcription_device,
                            compute_type=SETTINGS.transcription_compute_type,
                            vad_filter=True,
                            model_instance=model,
                        )
                        elapsed = time.perf_counter() - started
                        st.session_state.segments = segments
                        st.session_state.metadata = metadata
                        st.session_state.metadata["processing_seconds"] = round(elapsed, 2)
                        st.session_state.metadata["audio_duration_seconds"] = duration_sec
                        if duration_sec and duration_sec > 0:
                            st.session_state.measured_rtf = round(elapsed / duration_sec, 3)
                        st.session_state.edited_transcript = None
                        st.session_state.is_editing = False
                        st.toast("Transcription completed successfully!", icon="🎉")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Transcription failed: {type(exc).__name__}: {exc}", icon="❌")

        with col_btn_download:
            active_txt = get_active_transcript()
            has_transcript = bool(active_txt.strip())
            file_stem = Path(st.session_state.uploaded_name).stem if st.session_state.uploaded_name else "audio"
            dl_filename = f"{file_stem}_transcript.txt"

            st.download_button(
                label="⬇ Download Text File",
                data=active_txt.encode("utf-8") if has_transcript else b"",
                file_name=dl_filename,
                mime="text/plain; charset=utf-8",
                disabled=not has_transcript,
                key="btn_download_txt",
                use_container_width=True,
            )

    else:
        st.markdown(
            """
            <div class="empty-state-box" style="padding: 32px 16px;">
                <div class="empty-icon-circle">
                    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>
                        <polyline points="13 2 13 9 20 9"/>
                    </svg>
                </div>
                <div class="empty-title">No Audio File Selected</div>
                <div class="empty-desc">Upload an audio recording on the left to view file metadata and start transcription.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_btn_transcribe, col_btn_download = st.columns([1, 1], gap="small")
        with col_btn_transcribe:
            st.button("▶ Transcribe Audio", type="primary", use_container_width=True, disabled=True, key="btn_transcribe_disabled")
        with col_btn_download:
            st.download_button(
                label="⬇ Download Text File",
                data="",
                file_name="transcript.txt",
                disabled=True,
                use_container_width=True,
                key="btn_download_disabled",
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# SECTION 5: AUDIO PLAYER SECTION
# ==============================================================================
st.markdown(
    """
    <div class="full-width-card" style="margin-top: 24px;">
        <div class="card-header-row" style="margin-bottom: 14px;">
            <div class="card-icon-container" style="background: #F0FDF4; border-color: #DCFCE7; color: #16A34A;">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                    <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/>
                </svg>
            </div>
            <div>
                <h2 class="card-heading">Audio Player</h2>
                <p class="card-subheading">Preview, seek, and verify your uploaded recording</p>
            </div>
        </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
    playable = convert_for_browser(st.session_state.audio_path)
    with open(playable, "rb") as af:
        audio_bytes = af.read()
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")
    suffix = Path(playable).suffix.lower()
    mime_type = "audio/mpeg" if suffix in (".mp3", ".mpeg") else f"audio/{suffix.lstrip('.')}"
    duration_sec = ffprobe_duration(st.session_state.audio_path) or 0
    duration_label = format_audio_duration(duration_sec)
    format_label = get_audio_format_label(st.session_state.uploaded_name)
    size_label = format_file_size(st.session_state.uploaded_size)

    player_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: transparent;
            color: #0F172A;
            overflow: hidden;
        }}
        .player-container {{
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 16px 20px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        .player-meta-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 13px;
        }}
        .meta-left {{
            display: flex;
            align-items: center;
            gap: 10px;
            overflow: hidden;
        }}
        .meta-filename {{
            font-weight: 700;
            color: #1E293B;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 320px;
        }}
        .meta-badges {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .badge {{
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            color: #475569;
        }}
        .controls-row {{
            display: flex;
            align-items: center;
            gap: 14px;
        }}
        .play-btn {{
            width: 42px;
            height: 42px;
            border-radius: 50%;
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
            border: none;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 4px 10px rgba(79, 70, 229, 0.32);
            transition: all 0.2s ease;
            flex-shrink: 0;
        }}
        .play-btn:hover {{
            transform: scale(1.05);
            box-shadow: 0 6px 14px rgba(79, 70, 229, 0.45);
        }}
        .time-text {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
            font-size: 12px;
            font-weight: 600;
            color: #64748B;
            min-width: 44px;
            text-align: center;
            user-select: none;
        }}
        .slider-wrapper {{
            flex: 1;
            position: relative;
            display: flex;
            align-items: center;
        }}
        .seek-slider {{
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: #E2E8F0;
            outline: none;
            cursor: pointer;
        }}
        .seek-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: #4F46E5;
            cursor: pointer;
            box-shadow: 0 1px 4px rgba(0,0,0,0.2);
            transition: transform 0.1s ease;
        }}
        .seek-slider::-webkit-slider-thumb:hover {{
            transform: scale(1.2);
        }}
        .volume-group {{
            display: flex;
            align-items: center;
            gap: 8px;
            flex-shrink: 0;
        }}
        .mute-btn {{
            background: transparent;
            border: none;
            color: #64748B;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 4px;
        }}
        .mute-btn:hover {{
            color: #4F46E5;
        }}
        .volume-slider {{
            -webkit-appearance: none;
            width: 70px;
            height: 4px;
            border-radius: 2px;
            background: #CBD5E1;
            outline: none;
            cursor: pointer;
        }}
        .volume-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 11px;
            height: 11px;
            border-radius: 50%;
            background: #475569;
            cursor: pointer;
        }}
    </style>
    </head>
    <body>
        <div class="player-container">
            <div class="player-meta-row">
                <div class="meta-left">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#4F46E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>
                    </svg>
                    <span class="meta-filename" title="{st.session_state.uploaded_name}">{st.session_state.uploaded_name}</span>
                </div>
                <div class="meta-badges">
                    <span class="badge">{format_label}</span>
                    <span class="badge">{size_label}</span>
                    <span class="badge">{duration_label}</span>
                </div>
            </div>

            <div class="controls-row">
                <button id="btnPlay" class="play-btn" onclick="togglePlay()">
                    <svg id="iconPlay" width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                        <polygon points="5 3 19 12 5 21 5 3"/>
                    </svg>
                    <svg id="iconPause" width="18" height="18" viewBox="0 0 24 24" fill="currentColor" style="display:none;">
                        <rect x="6" y="4" width="4" height="16"/>
                        <rect x="14" y="4" width="4" height="16"/>
                    </svg>
                </button>

                <span id="txtCurrent" class="time-text">00:00</span>

                <div class="slider-wrapper">
                    <input type="range" id="seekSlider" class="seek-slider" min="0" max="100" value="0" step="0.1" oninput="onSeek(this.value)">
                </div>

                <span id="txtDuration" class="time-text">{duration_label}</span>

                <div class="volume-group">
                    <button id="btnMute" class="mute-btn" onclick="toggleMute()">
                        <svg id="iconVolume" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                            <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
                        </svg>
                        <svg id="iconMuted" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none;">
                            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                            <line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/>
                        </svg>
                    </button>
                    <input type="range" id="volSlider" class="volume-slider" min="0" max="1" step="0.05" value="1" oninput="onVolume(this.value)">
                </div>
            </div>

            <audio id="audioEl" preload="metadata" src="data:{mime_type};base64,{encoded_audio}"></audio>
        </div>

        <script>
            const audio = document.getElementById('audioEl');
            const btnPlay = document.getElementById('btnPlay');
            const iconPlay = document.getElementById('iconPlay');
            const iconPause = document.getElementById('iconPause');
            const txtCurrent = document.getElementById('txtCurrent');
            const txtDuration = document.getElementById('txtDuration');
            const seekSlider = document.getElementById('seekSlider');
            const btnMute = document.getElementById('btnMute');
            const iconVolume = document.getElementById('iconVolume');
            const iconMuted = document.getElementById('iconMuted');
            const volSlider = document.getElementById('volSlider');

            function formatTime(s) {{
                if (isNaN(s) || s < 0) return "00:00";
                const total = Math.floor(s);
                const h = Math.floor(total / 3600);
                const m = Math.floor((total % 3600) / 60);
                const sec = total % 60;
                if (h > 0) {{
                    return String(h).padStart(2, '0') + ":" + String(m).padStart(2, '0') + ":" + String(sec).padStart(2, '0');
                }}
                return String(m).padStart(2, '0') + ":" + String(sec).padStart(2, '0');
            }}

            function togglePlay() {{
                if (audio.paused) {{
                    audio.play();
                }} else {{
                    audio.pause();
                }}
            }}

            audio.addEventListener('play', () => {{
                iconPlay.style.display = 'none';
                iconPause.style.display = 'block';
            }});

            audio.addEventListener('pause', () => {{
                iconPlay.style.display = 'block';
                iconPause.style.display = 'none';
            }});

            audio.addEventListener('ended', () => {{
                iconPlay.style.display = 'block';
                iconPause.style.display = 'none';
                seekSlider.value = 0;
                txtCurrent.innerText = "00:00";
            }});

            audio.addEventListener('timeupdate', () => {{
                txtCurrent.innerText = formatTime(audio.currentTime);
                if (audio.duration) {{
                    seekSlider.value = (audio.currentTime / audio.duration) * 100;
                }}
            }});

            audio.addEventListener('loadedmetadata', () => {{
                if (audio.duration && !isNaN(audio.duration)) {{
                    txtDuration.innerText = formatTime(audio.duration);
                }}
            }});

            function onSeek(val) {{
                if (audio.duration) {{
                    audio.currentTime = (val / 100) * audio.duration;
                }}
            }}

            function onVolume(val) {{
                audio.volume = val;
                audio.muted = (val == 0);
                updateMuteIcons();
            }}

            function toggleMute() {{
                audio.muted = !audio.muted;
                updateMuteIcons();
            }}

            function updateMuteIcons() {{
                if (audio.muted || audio.volume === 0) {{
                    iconVolume.style.display = 'none';
                    iconMuted.style.display = 'block';
                }} else {{
                    iconVolume.style.display = 'block';
                    iconMuted.style.display = 'none';
                }}
            }}
        </script>
    </body>
    </html>
    """
    components.html(player_html, height=130)

else:
    st.markdown(
        """
        <div class="empty-state-box" style="padding: 24px;">
            <div class="empty-icon-circle" style="width: 44px; height: 44px; margin-bottom: 8px;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                    <line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/>
                </svg>
            </div>
            <div class="empty-title" style="font-size: 14px;">No Audio Loaded</div>
            <div class="empty-desc" style="font-size: 12px;">Upload an audio file above to enable interactive playback and waveform controls.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# SECTION 6: GENERATED TEXT / TRANSCRIPT SECTION
# ==============================================================================
active_transcript_text = get_active_transcript()
has_active_transcript = bool(active_transcript_text.strip())

st.markdown(
    """
    <div class="full-width-card" style="margin-top: 24px;">
    """,
    unsafe_allow_html=True,
)

# Header Row: Title on Left, Actions on Right
col_gen_title, col_gen_actions = st.columns([1, 1], gap="medium")

with col_gen_title:
    badge_label = "Edited" if st.session_state.edited_transcript else ""
    badge_html = f'<span style="background: #EEF2FF; color: #4F46E5; border: 1px solid #C7D2FE; padding: 2px 8px; border-radius: 6px; font-size: 12px; font-weight: 700; margin-left: 10px;">{badge_label}</span>' if badge_label else ""

    st.markdown(
        f"""
        <div class="card-header-row" style="margin-bottom: 0;">
            <div class="card-icon-container" style="background: #EFF6FF; border-color: #DBEAFE; color: #2563EB;">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                    <line x1="16" y1="13" x2="8" y2="13"/>
                    <line x1="16" y1="17" x2="8" y2="17"/>
                </svg>
            </div>
            <div style="display: flex; align-items: center;">
                <h2 class="card-heading">Generated Text</h2>
                {badge_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_gen_actions:
    if has_active_transcript:
        col_copy_share, col_edit_btn = st.columns([2, 1], gap="small")

        with col_copy_share:
            # Interactive Copy and Share toolbar with client-side user gestures & notifications
            escaped_text = json.dumps(active_transcript_text)
            escaped_title = json.dumps(f"AudioMind AI Transcript - {st.session_state.uploaded_name or 'Audio'}")

            actions_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
                    display: flex;
                    justify-content: flex-end;
                    align-items: center;
                    gap: 8px;
                    overflow: hidden;
                    height: 40px;
                }}
                .action-btn {{
                    display: inline-flex;
                    align-items: center;
                    gap: 6px;
                    background: #FFFFFF;
                    color: #334155;
                    border: 1px solid #CBD5E1;
                    border-radius: 8px;
                    padding: 7px 12px;
                    font-size: 13px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.2s ease;
                }}
                .action-btn:hover {{
                    border-color: #6366F1;
                    color: #4F46E5;
                    background: #F8FAFC;
                }}
                .action-btn.success {{
                    border-color: #10B981;
                    color: #059669;
                    background: #ECFDF5;
                }}
                #toast {{
                    position: fixed;
                    bottom: 4px;
                    right: 4px;
                    background: #1E293B;
                    color: #FFFFFF;
                    font-size: 11px;
                    font-weight: 600;
                    padding: 4px 10px;
                    border-radius: 6px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    opacity: 0;
                    transition: opacity 0.25s ease;
                    pointer-events: none;
                }}
                #toast.show {{
                    opacity: 1;
                }}
            </style>
            </head>
            <body>
                <button id="btnCopy" class="action-btn" onclick="copyText()">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                    </svg>
                    <span id="copyLabel">Copy Text</span>
                </button>

                <button id="btnShare" class="action-btn" onclick="shareText()">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/>
                        <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/>
                        <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>
                    </svg>
                    <span id="shareLabel">Share Text</span>
                </button>

                <div id="toast">Copied to clipboard</div>

                <script>
                    const transcriptData = {escaped_text};
                    const transcriptTitle = {escaped_title};
                    const toast = document.getElementById('toast');

                    function showToast(msg) {{
                        toast.innerText = msg;
                        toast.classList.add('show');
                        setTimeout(() => {{
                            toast.classList.remove('show');
                        }}, 2200);
                    }}

                    function copyFallback(text, onSuccessMsg) {{
                        const ta = document.createElement('textarea');
                        ta.value = text;
                        ta.style.position = 'fixed';
                        ta.style.opacity = '0';
                        document.body.appendChild(ta);
                        ta.select();
                        try {{
                            document.execCommand('copy');
                            const btn = document.getElementById('btnCopy');
                            btn.classList.add('success');
                            document.getElementById('copyLabel').innerText = 'Copied!';
                            showToast(onSuccessMsg || 'Copied transcript to clipboard!');
                            setTimeout(() => {{
                                btn.classList.remove('success');
                                document.getElementById('copyLabel').innerText = 'Copy Text';
                            }}, 2000);
                        }} catch (e) {{
                            showToast('Failed to copy text.');
                        }}
                        document.body.removeChild(ta);
                    }}

                    function copyText() {{
                        if (navigator.clipboard && navigator.clipboard.writeText) {{
                            navigator.clipboard.writeText(transcriptData).then(() => {{
                                const btn = document.getElementById('btnCopy');
                                btn.classList.add('success');
                                document.getElementById('copyLabel').innerText = 'Copied!';
                                showToast('Copied transcript to clipboard!');
                                setTimeout(() => {{
                                    btn.classList.remove('success');
                                    document.getElementById('copyLabel').innerText = 'Copy Text';
                                }}, 2000);
                            }}).catch(() => {{
                                copyFallback(transcriptData);
                            }});
                        }} else {{
                            copyFallback(transcriptData);
                        }}
                    }}

                    function shareText() {{
                        if (navigator.share) {{
                            navigator.share({{
                                title: transcriptTitle,
                                text: transcriptData
                            }}).then(() => {{
                                showToast('Shared transcript successfully!');
                            }}).catch((err) => {{
                                if (err.name !== 'AbortError') {{
                                    copyFallback(transcriptData, 'Sharing unavailable — copied to clipboard instead!');
                                }}
                            }});
                        }} else {{
                            copyFallback(transcriptData, 'Web Share not supported — copied to clipboard instead!');
                        }}
                    }}
                </script>
            </body>
            </html>
            """
            components.html(actions_html, height=44)

        with col_edit_btn:
            if not st.session_state.is_editing:
                if st.button("✏️ Edit Text", key="btn_start_edit", use_container_width=True):
                    st.session_state.is_editing = True
                    st.rerun()

# Card Body: Transcript Display or Edit Area or Empty State
if has_active_transcript:
    if st.session_state.is_editing:
        st.markdown(
            """
            <div style="margin-top: 14px; margin-bottom: 8px;">
                <span style="font-size: 13px; font-weight: 600; color: #475569;">
                    Edit the transcript below. Changes will be saved as plain text and reflected in text file downloads, copy, and playback views.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        edited_val = st.text_area(
            "Edit Transcript",
            value=active_transcript_text,
            height=380,
            label_visibility="collapsed",
            key="transcript_editor_area",
        )

        col_save, col_cancel, _ = st.columns([1.5, 1.5, 5], gap="small")
        with col_save:
            if st.button("💾 Save Changes", type="primary", use_container_width=True, key="btn_save_changes"):
                st.session_state.edited_transcript = edited_val.strip()
                st.session_state.is_editing = False
                st.toast("Transcript changes saved successfully!", icon="✅")
                st.rerun()
        with col_cancel:
            if st.button("Cancel", use_container_width=True, key="btn_cancel_changes"):
                st.session_state.is_editing = False
                st.rerun()

    else:
        # Standard View Mode: Clean, continuous readable text with proper paragraphs (no timestamps or duration labels)
        st.markdown('<div class="transcript-container">', unsafe_allow_html=True)

        transcript_content = st.session_state.edited_transcript if st.session_state.edited_transcript else format_display_transcript(st.session_state.segments)
        for paragraph in transcript_content.split("\n\n"):
            clean_para = paragraph.strip()
            if clean_para:
                st.markdown(
                    f"""
                    <div class="transcript-paragraph">
                        {clean_para}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown(
        """
        <div class="empty-state-box">
            <div class="empty-icon-circle">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                    <line x1="16" y1="13" x2="8" y2="13"/>
                    <line x1="16" y1="17" x2="8" y2="17"/>
                </svg>
            </div>
            <div class="empty-title">Your generated transcript will appear here.</div>
            <div class="empty-desc">Upload an audio file above and click "Transcribe Audio" to generate the full readable transcript.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
