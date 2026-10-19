"""
Chatbot-style UI for pipeline_v2.py, with conversation history and a LIVE
view of the pipeline stages while a question is running.

- The page draws immediately; the pipeline and models load afterwards with
  a visible status (first start only), then are reused for every question.
- While a question runs, the status header shows the current stage and a
  box below it shows the latest lines of the pipeline's output as they
  are printed. When it finishes, the status collapses, the answer is
  shown, and the full log stays available in an expander.

Uses the optional reranker / nli_verifier / multilingual_verifier
parameters of pipeline_v2.process_query(). The CLI
(`python pipeline_v2.py --query "..."`) is unaffected.

HOW TO RUN (inside the same .venv you already use):

    pip install streamlit
    streamlit run app_ui.py --server.fileWatcherType none

HISTORY: every question is also saved to history.jsonl (next to this
file) -- one JSON line per question: time, question, standard, language,
answer, response time and the full stage log. The sidebar can show any
saved run again and download the whole file. The chat on screen still
resets if you reload the page; the file does not.

VOICE: the chat bar has a mic button next to the send button.
  1. Press the mic and speak; press the check mark to stop recording.
  2. stt.py (faster-whisper, runs offline) turns the speech into text.
  3. The text is shown in an editable preview -- NOTHING is sent yet.
  4. The student edits it if needed and presses "Send question"
     (or "Discard"). Only then does it go through the SAME pipeline as
     a typed question.
The spoken language is always auto-detected (Tamil or English). The speech
model is loaded the first time the mic is used (not at start-up, to save memory).
Needs a recent Streamlit (chat_input with accept_audio) and
`pip install faster-whisper`. On an older Streamlit the mic button is
simply not shown and typing works as before.
"""
import contextlib
import datetime
import inspect
import io
import json
import os
import re
import socket
import sys
import time
from pathlib import Path


def _offline_if_no_internet():
    """No internet -> tell Hugging Face to use ONLY the downloaded models.
    Without this, loading a model that is already on disk still tries to
    contact huggingface.co and the app stops with 'Failed to resolve
    huggingface.co'. Must run before any model library is imported."""
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return
    try:
        socket.create_connection(("huggingface.co", 443), timeout=2).close()
    except OSError:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


_offline_if_no_internet()

import streamlit as st

VALID_STANDARDS = ["6", "7", "8"]
FINAL_MARKER = "STAGE 13 -- FINAL RESULT SUMMARY"
STAGE_RE = re.compile(r"STAGE [0-9A-Za-z\.]+ -- [^\n]+")
LIVE_TAIL_LINES = 40
HISTORY_FILE = Path(__file__).parent / "history.jsonl"
HISTORY_SHOWN = 50   # newest runs listed in the sidebar

st.set_page_config(page_title="Samacheer Kalvi Science Q&A", layout="wide")


@st.cache_resource(show_spinner=False)
def load_backend():
    """Imports the pipeline and builds the three models ONCE per running
    app. Imports are done here (not at the top of the file) so the page
    can draw first and show a loading status, instead of staying blank
    while torch / sentence-transformers / chromadb import."""
    sys.path.insert(0, str(Path(__file__).parent))
    import pipeline_v2 as _pipeline
    import rerank
    import nli_verify
    import nli_verify_multilingual
    import translate_nllb

    translate_nllb.available()  # load NLLB now, not on the first Tamil question
    import tamil_rescue
    tamil_rescue.load()  # rescue reranker, also loaded once at start
    models = (
        rerank.Reranker(),
        nli_verify.NLIVerifier(),
        nli_verify_multilingual.MultilingualNLIVerifier(),
    )
    return _pipeline, models


@st.cache_resource(show_spinner=False)
def load_stt():
    """Loads the speech-to-text model once, only when the first voice
    question arrives (so typed-only use does not pay for it)."""
    sys.path.insert(0, str(Path(__file__).parent))
    import stt
    stt.load()
    return stt


# Mic button inside the chat bar (st.chat_input(accept_audio=True)) only
# exists in newer Streamlit versions -- checked once here.
CHAT_HAS_MIC = "accept_audio" in inspect.signature(st.chat_input).parameters


class LiveLog(io.StringIO):
    """A StringIO that also pushes new output to the page while the
    pipeline is still running: updates the status label with the current
    stage name and redraws the tail of the log (throttled)."""

    def __init__(self, status, box):
        super().__init__()
        self.status = status
        self.box = box
        self._last_draw = 0.0

    def write(self, s):
        n = super().write(s)
        stages = STAGE_RE.findall(s)
        if stages:
            self.status.update(label=stages[-1])
        now = time.time()
        if "\n" in s and now - self._last_draw > 0.25:
            self._last_draw = now
            self.redraw()
        return n

    def redraw(self):
        tail = "\n".join(self.getvalue().splitlines()[-LIVE_TAIL_LINES:])
        self.box.code(tail, language=None)


def save_history(full_query, answer, full_log, seconds):
    """Appends one run to history.jsonl. Never breaks the app."""
    lang = re.search(r"Detected Language:\n(.*)", full_log)
    std = full_query.split("-", 1)[0].strip()
    rec = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": full_query,
        "standard": std,
        "language": lang.group(1).strip() if lang else "",
        "answer": answer,
        "seconds": round(seconds, 2),
        "log": full_log,
    }
    try:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as e:
        st.warning(f"Could not save history: {e}")


def load_history():
    if not HISTORY_FILE.exists():
        return []
    runs = []
    with open(HISTORY_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return runs


with st.sidebar:
    st.header("Settings")
    default_standard = st.selectbox("Default Standard", VALID_STANDARDS, index=1)
    st.caption('Type "6-...", "7-...", or "8-..." in a message to override this for just that message.')
    if CHAT_HAS_MIC:
        voice_lang = "Auto detect"   # Tamil or English is detected from the speech
        st.caption("Press the mic in the chat bar, speak, then press the check mark. "
                   "You will see the text first and can edit it before sending. "
                   "Spoken questions use the Default Standard above "
                   '(or add "8-" in the preview).')
    else:
        voice_lang = None
        st.caption("Voice input needs a newer Streamlit: pip install -U streamlit")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.header("History")
    _runs = load_history()
    if not _runs:
        st.caption("No saved runs yet.")
    else:
        st.caption(f"{len(_runs)} saved runs (newest first).")
        _newest = list(reversed(_runs))[:HISTORY_SHOWN]
        _labels = ["-- select a past run --"] + [
            f"{r['time']} | {r['seconds']}s | {r['query'][:60]}" for r in _newest]
        _pick = st.selectbox("View a past run", range(len(_labels)),
                             format_func=lambda i: _labels[i])
        st.session_state.history_pick = _newest[_pick - 1] if _pick else None
        st.download_button("Download history.jsonl",
                           HISTORY_FILE.read_bytes(),
                           file_name="history.jsonl",
                           mime="application/json")

st.title("Samacheer Kalvi Science Q&A")
st.caption("Ask in Tamil, Tanglish, or English. Stages are shown live while a question runs.")

with st.status("Loading pipeline and models (first start only -- can take a minute or two)...",
               expanded=True) as load_status:
    _t0 = time.time()
    pipeline_v2, (reranker, nli_verifier, multilingual_verifier) = load_backend()
    # The speech model is NOT pre-loaded here: with all pipeline models in
    # memory, also loading Whisper at start-up ran Windows out of memory
    # ("The paging file is too small", os error 1455). It is loaded the
    # first time the mic is used instead (load_stt() is cached).
    load_status.update(label=f"Models ready ({time.time() - _t0:.0f}s)", state="complete", expanded=False)

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str, "log": str | None}

_hist = st.session_state.get("history_pick")
if _hist:
    with st.container():
        st.markdown(f"**Saved run** -- {_hist['time']} -- "
                    f"{_hist['language']} -- {_hist['seconds']} s")
        st.markdown(f"**Question:** {_hist['query']}")
        st.markdown(_hist["answer"])
        with st.expander("Full pipeline log (all stages)"):
            st.text(_hist["log"])


def run_pipeline(full_query: str, status, box) -> tuple[str, str]:
    """Calls pipeline_v2.process_query() in-process with the cached models,
    streaming its printed stages to the page. Returns (answer_text, full_log)."""
    buf = LiveLog(status, box)
    try:
        with contextlib.redirect_stdout(buf):
            pipeline_v2.process_query(
                full_query,
                reranker=reranker,
                nli_verifier=nli_verifier,
                multilingual_verifier=multilingual_verifier,
            )
    except Exception as e:
        buf.write(f"\n[error] {type(e).__name__}: {e}\n")
    buf.redraw()

    full_log = buf.getvalue()
    if FINAL_MARKER in full_log:
        answer = full_log.split(FINAL_MARKER, 1)[1].strip()
    else:
        # Some outcomes stop before Stage 13 (e.g. "No relevant textbook
        # content found in Standard 7." or an [error] line). Show that last
        # message as the answer instead of a generic one.
        lines = [l.strip() for l in full_log.splitlines()
                 if l.strip() and not l.strip().startswith(("=", "-"))]
        answer = lines[-1] if lines else "No answer was produced -- see the log below."
    return answer, full_log


# ---- render existing conversation ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("log"):
            with st.expander("Full pipeline log (all stages)"):
                st.text(msg["log"])

# ---- voice preview: the student checks / edits the text before sending ----
if "pending_voice" not in st.session_state:
    st.session_state.pending_voice = None   # {"id", "text", "lang", "secs"} or None
if "voice_n" not in st.session_state:
    st.session_state.voice_n = 0


def _voice_send(edit_key):
    """'Send question' pressed: hand the (edited) text to the next run."""
    text = st.session_state.get(edit_key, "").strip()
    if text:
        st.session_state.voice_send = text
    st.session_state.pending_voice = None


def _voice_discard():
    st.session_state.pending_voice = None


# ---- new message (typed, or spoken with the mic in the chat bar) ----
if CHAT_HAS_MIC:
    submitted = st.chat_input("Ask a question... (or press the mic)",
                              accept_audio=True, audio_sample_rate=16000)
else:
    submitted = st.chat_input("Ask a question...")

user_text = None
spoken = False
if isinstance(submitted, str):
    user_text = submitted
elif submitted is not None:
    user_text = (submitted.text or "").strip()
    audio = getattr(submitted, "audio", None)
    if not user_text and audio is not None:
        # Recording finished -> transcribe, then SHOW it. Do not send.
        heard = ""
        with st.spinner("Converting speech to text..."):
            try:
                stt = load_stt()
                _t = time.time()
                heard, lang_used = stt.transcribe(audio.getvalue(),
                                                  stt.LANGUAGES[voice_lang])
                secs = time.time() - _t
            except Exception as e:
                st.error(f"Speech-to-text failed: {type(e).__name__}: {e}")
        if heard:
            st.session_state.voice_n += 1
            st.session_state.pending_voice = {"id": st.session_state.voice_n, "text": heard,
                                              "lang": lang_used, "secs": secs}
        else:
            st.warning("No speech was recognised -- please try the mic again.")
        user_text = None

# "Send question" was pressed in the previous run -> send it now.
_voice_send_text = st.session_state.pop("voice_send", None)
if _voice_send_text:
    user_text = _voice_send_text
    spoken = True

_pv = st.session_state.pending_voice
if _pv is not None and not user_text:
    with st.container(border=True):
        st.markdown(f"**🎤 Check your question before sending** "
                    f"({'Tamil' if _pv['lang'] == 'ta' else 'English'}, "
                    f"converted in {_pv['secs']:.1f} s)")
        _edit_key = f"voice_edit_{_pv['id']}"
        st.text_area("Heard text -- edit if a word is wrong", value=_pv["text"],
                     key=_edit_key, height=80)
        _c1, _c2, _ = st.columns([1, 1, 4])
        _c1.button("Send question", type="primary", key=f"voice_send_{_pv['id']}",
                   on_click=_voice_send, args=(_edit_key,))
        _c2.button("Discard", key=f"voice_discard_{_pv['id']}", on_click=_voice_discard)

if user_text:
    user_text = user_text.strip().lstrip("-").strip()  # ignore a stray leading hyphen
    prefix, sep, rest = user_text.partition("-")
    if sep and prefix.strip() in VALID_STANDARDS:
        full_query = user_text
    else:
        full_query = f"{default_standard}-{user_text}"
    display_text = f"🎤 {user_text}" if spoken else user_text

    st.session_state.messages.append({"role": "user", "content": display_text, "log": None})
    with st.chat_message("user"):
        st.markdown(display_text)

    with st.chat_message("assistant"):
        with st.status("Starting pipeline...", expanded=True) as status:
            box = st.empty()
            _t_start = time.time()
            answer, full_log = run_pipeline(full_query, status, box)
            _secs = time.time() - _t_start
            status.update(label=f"Finished ({_secs:.1f} s)", state="complete", expanded=False)
        save_history(full_query, answer, full_log, _secs)
        st.markdown(answer)
        with st.expander("Full pipeline log (all stages)"):
            st.text(full_log)

    st.session_state.messages.append({"role": "assistant", "content": answer, "log": full_log})