"""
Speech-to-text for the Q&A app (voice questions).

Uses faster-whisper (an offline Whisper implementation). Nothing is sent
to the internet: the model is downloaded once on first use, then it runs
locally like the other models in this project.

The text it returns is shown to the user for checking/editing first; only
after they press "Send" does it go into the SAME pipeline as a typed
question.

INSTALL (inside the same .venv):
    pip install faster-whisper

COMPARE MODELS ON YOUR OWN VOICE (pick the best speed / accuracy):
    python test_voice.py --compare small,medium,large-v3-turbo
"""
import io
import os
import sys
import wave

import numpy as np

# ---- settings -------------------------------------------------------
# Model size. Bigger = more accurate Tamil, but slower:
#   "small"          ~0.5 GB  fast,   weak Tamil (mis-hears words)
#   "medium"         ~1.5 GB  slower, clearly better Tamil
#   "large-v3-turbo" ~1.6 GB  best accuracy; fast on GPU, slow on CPU
# Use `python test_voice.py --compare ...` to choose on your own voice.
MODEL_SIZE = "small"

# "cpu" always works. "cuda" uses the NVIDIA GPU (much faster) but needs
# the CUDA libraries -- test it first with:
#     python test_voice.py --device cuda
DEVICE = "cpu"

BEAM_SIZE = 5     # 5 = more accurate; 1 = faster (greedy)

# A short hint given to Whisper before decoding. It tells the model that
# this is a school-science question, so it prefers science words
# (e.g. "விசை" instead of a similar-sounding everyday word).
USE_PROMPT = True
PROMPTS = {
    "ta": "இது ஒரு பள்ளி அறிவியல் கேள்வி. விசை, அணு, ஒளி, ஒலி, வெப்பம், "
          "காந்தம், மின்னோட்டம், அமிலம் என்றால் என்ன?",
    "en": "This is a school science question. What is force, atom, light, "
          "sound, heat, magnet, current, acid?",
}

# Language choices shown in the UI -> Whisper language code.
# None = auto detect, but ONLY between Tamil and English.
LANGUAGES = {
    "Auto detect": None,
    "Tamil": "ta",
    "English": "en",
}
ALLOWED = ("ta", "en")   # the only languages our pipeline supports

WHISPER_RATE = 16000     # Whisper expects 16 kHz mono audio

_models = {}             # (size, device) -> loaded model


def _compute_type(device):
    return "int8_float16" if device == "cuda" else "int8"


def _add_torch_cuda_dlls():
    """Windows + GPU: faster-whisper needs cuBLAS / cuDNN DLLs. PyTorch
    (already installed for the other models) ships them in torch/lib, so
    that folder is added to the DLL search path."""
    if os.name != "nt":
        return
    try:
        import torch
        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib):
            os.add_dll_directory(lib)
            os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def load(model_size=None, device=None):
    """Loads a Whisper model once (per size/device) and reuses it."""
    size = model_size or MODEL_SIZE
    dev = device or DEVICE
    key = (size, dev)
    if key not in _models:
        from faster_whisper import WhisperModel
        if dev == "cuda":
            _add_torch_cuda_dlls()
        model = WhisperModel(size, device=dev, compute_type=_compute_type(dev),
                             cpu_threads=os.cpu_count() or 4)
        # Warm-up on 1 s of silence: the first real question is not slowed
        # down by one-time setup, and a broken GPU setup fails here.
        list(model.transcribe(np.zeros(WHISPER_RATE, dtype=np.float32),
                              language="en", beam_size=1)[0])
        _models[key] = model
    return _models[key]


def _wav_to_array(audio):
    """
    Reads a WAV (bytes or file path) into the 16 kHz mono float32 array
    Whisper expects -- with Python's own `wave` module, no ffmpeg / PyAV
    (newer PyAV versions break faster-whisper's own audio decoding).
    Returns None if the input is not a WAV.
    """
    try:
        src = io.BytesIO(audio) if isinstance(audio, (bytes, bytearray)) else audio
        with wave.open(src, "rb") as w:
            rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError):
        return None
    if width == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        return None
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)       # stereo -> mono
    if rate != WHISPER_RATE and len(x) > 1:             # simple resample
        n_out = int(round(len(x) * WHISPER_RATE / rate))
        x = np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x)
    return x.astype(np.float32)


# Auto detect picks English only when Whisper is clearly sure it is
# English. Short Tamil questions were sometimes scored as English and then
# came out as unrelated English words ("We say in Robin Hood."). Real
# English speech is normally detected with high confidence, so anything
# less certain is treated as Tamil (Tanglish speech also sounds Tamil).
ENGLISH_MIN_PROB = 0.7


def _pick_language(model, audio):
    """Auto detect, limited to Tamil / English: one cheap language-ID pass
    (no full transcription). English only if clearly English, else Tamil."""
    try:
        _lang, _p, all_probs = model.detect_language(audio)
        probs = dict(all_probs)
    except Exception:   # older faster-whisper without detect_language()
        _segs, info = model.transcribe(audio, beam_size=1)
        probs = dict(info.all_language_probs or [])
    return "en" if probs.get("en", 0.0) >= ENGLISH_MIN_PROB else "ta"


def transcribe(audio, language=None, model_size=None, device=None):
    """
    audio:    raw audio bytes (the WAV from the chat-bar mic) or a file path.
    language: "ta", "en" or None (auto detect between Tamil and English).

    Returns (text, language_used).
    """
    model = load(model_size, device)
    source = _wav_to_array(audio)
    if source is None:   # not a WAV (mp3 / m4a file) -> let faster-whisper decode it
        source = io.BytesIO(audio) if isinstance(audio, (bytes, bytearray)) else audio
    if language is None:
        language = _pick_language(model, source)
        if hasattr(source, "seek"):
            source.seek(0)
    segments, _info = model.transcribe(
        source,
        language=language,
        task="transcribe",                 # keep the spoken language, do NOT translate
        beam_size=BEAM_SIZE,
        vad_filter=True,                   # cut silence before/after speech
        initial_prompt=PROMPTS.get(language) if USE_PROMPT else None,
        condition_on_previous_text=False,  # no repeating loops on short clips
        without_timestamps=True,           # we only need the text -> a bit faster
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, language


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python stt.py <audio file> [ta|en]")
        sys.exit(1)
    lang = sys.argv[2] if len(sys.argv) > 2 else None
    text, used = transcribe(sys.argv[1], lang)
    print(f"Language used: {used}")
    print(f"Text: {text}")