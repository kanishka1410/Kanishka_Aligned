"""
Backend test for voice questions -- no Streamlit needed.

Step 1: records from your microphone (or reuses a saved recording).
Step 2: speech-to-text with stt.py and prints what was heard + how long.
Step 3: (optional) sends that text through pipeline_v2, exactly like a
        typed question.

INSTALL (inside the same .venv):
    pip install faster-whisper sounddevice

USAGE:
    python test_voice.py                          record 5 s, auto detect, settings from stt.py
    python test_voice.py --lang ta                record 5 s, Tamil
    python test_voice.py --lang ta --run 8        record, then run pipeline on Std 8

    Choose the best model on YOUR voice (records once, tries each model
    on the same recording, prints text + time for each):
    python test_voice.py --lang ta --compare small,medium,large-v3-turbo

    Re-test the last recording without speaking again:
    python test_voice.py --file test_voice.wav --lang ta --compare small,medium

    Check whether the GPU works (much faster if it does):
    python test_voice.py --lang ta --device cuda
"""
import argparse
import sys
import time
import wave

import stt

SAMPLE_RATE = 16000   # same rate the chat-bar mic uses


def record(seconds, path="test_voice.wav"):
    """Records mono 16 kHz audio from the default microphone into a WAV."""
    import sounddevice as sd
    print(f"Recording for {seconds} s -- speak now...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="int16")
    sd.wait()
    print("Recording finished.")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)          # int16
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.tobytes())
    print(f"Saved to {path} (you can play it to check the mic).")
    return path


def run_one(path, language, size, device):
    print(f"\n--- model={size}  device={device} ---")
    t0 = time.time()
    stt.load(size, device)
    print(f"Load        : {time.time() - t0:.1f} s (one time only; the app pre-loads it)")
    t0 = time.time()
    text, used = stt.transcribe(path, language, size, device)
    secs = time.time() - t0
    print(f"Language    : {used}")
    print(f"Transcribe  : {secs:.1f} s   <- what the student waits for")
    print(f"Heard text  : {text!r}")
    return text, used, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="audio file to use instead of recording")
    ap.add_argument("--seconds", type=float, default=5, help="recording length")
    ap.add_argument("--lang", choices=["ta", "en", "auto"], default="auto")
    ap.add_argument("--model", help=f"model size (default {stt.MODEL_SIZE})")
    ap.add_argument("--device", choices=["cpu", "cuda"],
                    help=f"device (default {stt.DEVICE})")
    ap.add_argument("--compare", help="comma list of model sizes to compare")
    ap.add_argument("--run", choices=["6", "7", "8"],
                    help="also run the pipeline for this standard")
    args = ap.parse_args()

    path = args.file or record(args.seconds)
    language = None if args.lang == "auto" else args.lang
    device = args.device or stt.DEVICE

    if args.compare:
        rows = []
        for size in [s.strip() for s in args.compare.split(",") if s.strip()]:
            try:
                text, used, secs = run_one(path, language, size, device)
                rows.append((size, f"{secs:.1f} s", text))
            except Exception as e:
                print(f"FAILED: {type(e).__name__}: {e}")
                rows.append((size, "failed", ""))
        print("\n" + "=" * 70)
        print("COMPARISON (same recording)")
        print("=" * 70)
        for size, secs, text in rows:
            print(f"{size:<16} {secs:>8}   {text}")
        return

    text, _used, _secs = run_one(path, language, args.model or stt.MODEL_SIZE, device)
    if not text:
        print("\nNo speech recognised. Check the microphone / speak louder.")
        sys.exit(1)

    if args.run:
        import pipeline_v2
        query = f"{args.run}-{text}"
        print(f"\nRunning pipeline with: {query}\n")
        pipeline_v2.process_query(query)


if __name__ == "__main__":
    main()