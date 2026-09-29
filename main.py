"""
AI Meeting Assistant (v2)
==========================
Upload a meeting video -> get transcript, speaker labels, summary, agenda,
decisions, action items, per-speaker sentiment and a downloadable report.

What's new vs v1
----------------
* Groq calls survive rate limits: retry/backoff that honours Groq's own
  "try again in Xs" hint, optional tokens-per-minute pacing, and automatic
  splitting of any chunk that is "too large" for the model.
* Long meetings work: transcript is chunked (~1,800 words), each chunk is
  analysed ONCE (summary + agenda + decisions + action items in one JSON call),
  then chunk results are merged (map-reduce). Nothing is silently truncated.
* Local fallbacks for everything: Google Translate (translation), BART
  (summary), TF-IDF topic blocks (agenda), spaCy rules (action items).
  With no GROQ_API_KEY at all the app still runs in local-only mode.
* Better speaker detection:
    - "pyannote" engine (real diarization; needs HF_TOKEN), or
    - "lightweight" engine: sliding-window voice embeddings + spectral
      clustering with eigengap speaker-count estimate + merging of near-
      duplicate / tiny clusters + isolated-flip cleanup.
  You can also just tell the app how many people were in the meeting.
* Two-step flow: check/fix speakers and rename them BEFORE spending LLM tokens.
  Transcript + voice embeddings are cached, so re-detecting speakers is instant
  (lightweight engine) and never re-runs Whisper.
* Sentiment: RoBERTa transformer (TextBlob as lightweight option) with
  positive/neutral/negative split per speaker and a sentiment timeline.

Setup
-----
    pip install streamlit openai-whisper resemblyzer librosa scikit-learn scipy \
                groq python-dotenv spacy textblob transformers torch \
                deep-translator indic-transliteration unidecode pandas matplotlib
    python -m spacy download en_core_web_sm
    # optional, best speaker accuracy:
    pip install pyannote.audio

.env file:
    GROQ_API_KEY=gsk_...
    HF_TOKEN=hf_...            # only for the pyannote engine; you must also accept
                               # the terms on huggingface.co for
                               # pyannote/speaker-diarization-3.1 and
                               # pyannote/segmentation-3.0

Run:  streamlit run app.py      (ffmpeg must be installed and on PATH)
"""

import os
import re
import json
import time
import random
import shutil
import tempfile
import subprocess
import importlib.util
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_TOKEN", "")

st.set_page_config(page_title="AI Meeting Assistant", page_icon="🗣️", layout="wide")

LANG_OPTIONS = {
    "Auto-detect": None, "English": "en", "Hindi": "hi", "Marathi": "mr",
    "Urdu": "ur", "Bengali": "bn", "Gujarati": "gu", "Tamil": "ta",
    "Telugu": "te", "Spanish": "es", "French": "fr", "German": "de",
}
SPEAKER_COLORS = ["#7C4DFF", "#18C6C6", "#FF6584", "#FFB300", "#43A047",
                  "#5C6BC0", "#8D6E63", "#EC407A", "#26A69A", "#9CCC65"]


# ----------------------------------------------------------------------------
# Visual theme
# ----------------------------------------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    :root { --accent-1:#7C4DFF; --accent-2:#18C6C6; --accent-3:#FF6584; }
    @keyframes fadeInUp { from {opacity:0; transform:translateY(14px);} to {opacity:1; transform:translateY(0);} }
    @keyframes gradientShift { 0%{background-position:0% 50%;} 50%{background-position:100% 50%;} 100%{background-position:0% 50%;} }
    .main .block-container { animation: fadeInUp 0.6s ease-out; padding-top: 2rem; }
    h1 { font-family:'Poppins',sans-serif !important; font-weight:700 !important;
         background: linear-gradient(90deg,var(--accent-1),var(--accent-2),var(--accent-3),var(--accent-1));
         background-size:300% 300%; -webkit-background-clip:text; -webkit-text-fill-color:transparent;
         background-clip:text; animation: gradientShift 6s ease infinite; }
    div[data-testid="stButton"] button { background: linear-gradient(90deg,var(--accent-1),var(--accent-2));
         color:white; border:none; border-radius:10px; padding:0.6rem 1.4rem; font-weight:600;
         transition: transform .15s ease, box-shadow .15s ease; box-shadow:0 2px 10px rgba(124,77,255,.25); }
    div[data-testid="stButton"] button:hover { transform: translateY(-2px) scale(1.02);
         box-shadow:0 6px 18px rgba(124,77,255,.4); }
    div[data-testid="stDownloadButton"] button { background: linear-gradient(90deg,var(--accent-2),var(--accent-1));
         color:white; border:none; border-radius:10px; font-weight:600; }
    button[role="tab"] { border-radius:10px 10px 0 0 !important; font-weight:600; }
    button[aria-selected="true"] { background: linear-gradient(90deg,rgba(124,77,255,.15),rgba(24,198,198,.15)) !important;
         border-bottom:3px solid var(--accent-1) !important; }
    .stAlert { border-radius:12px; animation: fadeInUp .5s ease-out; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg,rgba(124,77,255,.06),rgba(24,198,198,.04));
         border-right:1px solid rgba(124,77,255,.15); }
    div[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }
    details { border-radius:10px !important; }
    </style>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Small helpers + cached model loaders
# ----------------------------------------------------------------------------
def fmt_time(sec: float) -> str:
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str):
    import whisper
    return whisper.load_model(model_size)


@st.cache_resource(show_spinner=False)
def load_spacy_model():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


@st.cache_resource(show_spinner=False)
def load_voice_encoder():
    from resemblyzer import VoiceEncoder
    return VoiceEncoder()


@st.cache_resource(show_spinner=False)
def load_bart_summarizer():
    from transformers import pipeline
    return pipeline("summarization", model="facebook/bart-large-cnn")


@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    from transformers import pipeline
    return pipeline("text-classification",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    top_k=None, truncation=True, max_length=256)


def check_ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------
# Step 1: audio + transcription
# ----------------------------------------------------------------------------
def extract_audio(video_path: str, audio_path: str):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", "-vn", audio_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.decode(errors='ignore')}")


def transcribe_audio(model, audio_path: str, language=None):
    # condition_on_previous_text=False reduces Whisper's repetition loops /
    # hallucination cascades on long recordings.
    kwargs = {"task": "transcribe", "verbose": False, "condition_on_previous_text": False}
    if language:
        kwargs["language"] = language
    return model.transcribe(audio_path, **kwargs)


def clean_segments(segments, no_speech_threshold=0.6, avg_logprob_threshold=-1.0):
    """Drop empty segments and ones Whisper likely hallucinated on non-speech
    (high no_speech_prob AND low confidence). Returns light-weight dicts."""
    out = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        if (seg.get("no_speech_prob", 0.0) > no_speech_threshold
                and seg.get("avg_logprob", 0.0) < avg_logprob_threshold):
            continue
        out.append({"start": float(seg["start"]), "end": float(seg["end"]), "text": text})
    return out


def get_language_name(lang_code: str) -> str:
    try:
        import whisper
        return whisper.tokenizer.LANGUAGES.get(lang_code, lang_code).title()
    except Exception:
        return lang_code


def has_non_latin_chars(text: str) -> bool:
    return any(c.isalpha() and ord(c) > 0x02AF for c in text)


def detect_script(text: str) -> str:
    if any("\u0900" <= c <= "\u097F" for c in text):
        return "devanagari"
    if any("\u0600" <= c <= "\u06FF" for c in text):
        return "arabic"
    return "other"


def romanize_text(text: str) -> str:
    if not text or not text.strip():
        return text
    if detect_script(text) == "devanagari":
        try:
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate
            return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
        except Exception:
            pass
    try:
        from unidecode import unidecode
        return unidecode(text)
    except Exception:
        return text


# ----------------------------------------------------------------------------
# Step 2: speaker diarization
#   Engine A: pyannote.audio (real diarization: VAD + speaker-change + overlap)
#   Engine B: lightweight - sliding-window resemblyzer embeddings, spectral
#             clustering + eigengap, merging of similar/tiny clusters.
#   Both end by mapping speaker turns/windows onto Whisper segments.
# ----------------------------------------------------------------------------
WIN_SEC = 1.6          # embedding window (resemblyzer's native partial length)
MAX_FIT_WINDOWS = 1500  # spectral clustering is fit on at most this many windows


def pyannote_available() -> bool:
    try:
        return importlib.util.find_spec("pyannote.audio") is not None
    except Exception:
        return False


def resolve_engine(choice: str) -> str:
    if choice.startswith("Lightweight"):
        return "lightweight"
    if choice.startswith("pyannote"):
        return "pyannote"
    return "pyannote" if (HF_TOKEN and pyannote_available()) else "lightweight"


@st.cache_resource(show_spinner=False)
def load_pyannote_pipeline(hf_token: str):
    from pyannote.audio import Pipeline
    name = "pyannote/speaker-diarization-3.1"
    try:
        pipe = Pipeline.from_pretrained(name, use_auth_token=hf_token)
    except TypeError:  # newer pyannote versions renamed the argument
        pipe = Pipeline.from_pretrained(name, token=hf_token)
    if pipe is None:
        raise RuntimeError("Could not load pyannote pipeline - check HF_TOKEN and that "
                           "you accepted the model terms on huggingface.co.")
    try:
        import torch
        if torch.cuda.is_available():
            pipe.to(torch.device("cuda"))
    except Exception:
        pass
    return pipe


def run_pyannote(audio_path: str, expected: int, max_speakers: int, hf_token: str):
    import torch
    import librosa
    pipe = load_pyannote_pipeline(hf_token)
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio = {"waveform": torch.from_numpy(y).unsqueeze(0), "sample_rate": sr}
    kwargs = {"num_speakers": expected} if expected > 0 else {"min_speakers": 1, "max_speakers": max_speakers}
    out = pipe(audio, **kwargs)
    annotation = getattr(out, "speaker_diarization", out)
    turns = [(float(t.start), float(t.end), str(lab))
             for t, _, lab in annotation.itertracks(yield_label=True)]
    if not turns:
        raise RuntimeError("pyannote returned no speech turns")
    return turns


def labels_from_turns(segs, turns):
    names = sorted({t[2] for t in turns})
    idmap = {n: i for i, n in enumerate(names)}
    out = []
    for s in segs:
        overlap = {}
        for a, b, lab in turns:
            o = min(b, s["end"]) - max(a, s["start"])
            if o > 0:
                overlap[lab] = overlap.get(lab, 0.0) + o
        if overlap:
            out.append(idmap[max(overlap, key=overlap.get)])
        else:  # segment fell in a gap: take the nearest turn in time
            mid = (s["start"] + s["end"]) / 2
            best = min(turns, key=lambda t: 0 if t[0] <= mid <= t[1] else min(abs(t[0] - mid), abs(t[1] - mid)))
            out.append(idmap[best[2]])
    return out


# ---- lightweight engine ------------------------------------------------------
def compute_window_embeddings(audio_path: str, segments):
    """Embed sliding 1.6 s windows INSIDE each Whisper segment (short segments
    use their exact span). Many windows per segment -> a robust majority vote,
    and a much better signal than one blended embedding per segment."""
    import librosa
    encoder = load_voice_encoder()
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    hop = 0.8 if len(y) / sr < 2400 else 1.6
    win = int(WIN_SEC * sr)
    global_rms = float(np.sqrt(np.mean(y ** 2))) + 1e-9
    embs, seg_ids = [], []

    for i, seg in enumerate(segments):
        s, e = seg["start"], seg["end"]
        dur = e - s
        if dur < 0.8:
            continue  # too short to embed reliably; inherits a neighbour later
        if dur <= WIN_SEC:
            spans = [(int(s * sr), int(e * sr))]
        else:
            starts = list(np.arange(s, e - WIN_SEC + 1e-6, hop))
            if (e - WIN_SEC) - starts[-1] > hop / 2:
                starts.append(e - WIN_SEC)
            spans = [(int(t * sr), int(t * sr) + win) for t in starts]
        for a, b in spans:
            clip = y[a:b]
            if len(clip) < 0.8 * sr:
                continue
            if np.sqrt(np.mean(clip ** 2)) < 0.3 * global_rms:
                continue  # near-silent window
            try:
                emb = encoder.embed_utterance(clip.astype(np.float32))
            except Exception:
                continue
            embs.append(emb)
            seg_ids.append(i)

    return {"emb": np.array(embs, dtype=np.float32), "seg_ids": np.array(seg_ids, dtype=int)}


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _normalize_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _affinity(X, prune_frac=0.06):
    """Cosine affinity, keep only each row's strongest neighbours, symmetrize.
    (Standard refinement that makes spectral clustering of voice embeddings
    far less sensitive to noise.)"""
    n = len(X)
    sim = np.clip(X @ X.T, 0, 1)
    p = min(n, max(3, int(prune_frac * n)))
    idx = np.argpartition(-sim, kth=p - 1, axis=1)[:, :p]
    rows = np.arange(n)[:, None]
    A = np.zeros_like(sim)
    A[rows, idx] = sim[rows, idx]
    return 0.5 * (A + A.T)


def estimate_num_speakers(X, max_k):
    """Eigengap heuristic on the normalized graph Laplacian."""
    n = len(X)
    max_k = max(1, min(max_k, n - 1))
    A = _affinity(X)
    d = A.sum(axis=1) + 1e-9
    dm = 1.0 / np.sqrt(d)
    L = np.eye(n) - dm[:, None] * A * dm[None, :]
    eig = np.linalg.eigvalsh(L)[:max_k + 1]
    gaps = np.diff(eig)
    return max(1, int(np.argmax(gaps)) + 1)


def _merge_clusters(emb, labels, merge_thr, min_share):
    """Post-process an over-segmented clustering: absorb tiny clusters into
    their nearest neighbour, then merge clusters whose centroids are closer
    than `merge_thr` (cosine distance)."""
    labels = labels.copy()
    while True:
        ids = np.unique(labels)
        if len(ids) <= 1:
            break
        cents = np.array([_unit(emb[labels == c].mean(axis=0)) for c in ids])
        shares = np.array([(labels == c).mean() for c in ids])
        dist = 1.0 - cents @ cents.T
        np.fill_diagonal(dist, np.inf)
        small = np.where(shares < min_share)[0]
        if len(small):
            i = small[np.argmin(shares[small])]
            j = int(np.argmin(dist[i]))
            labels[labels == ids[i]] = ids[j]
            continue
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        if dist[i, j] < merge_thr:
            labels[labels == ids[i]] = ids[j]
            continue
        break
    return labels


def cluster_windows(emb, expected=0, max_k=8, merge_thr=0.12, min_share=0.03):
    """Returns (labels per window, info dict)."""
    n = len(emb)
    info = {"k_initial": 1, "closest_pair": None}
    if n < 6 or expected == 1:
        return np.zeros(n, dtype=int), info

    emb = _normalize_rows(emb)
    step = int(np.ceil(n / MAX_FIT_WINDOWS))
    fit_idx = np.arange(0, n, step)
    X = emb[fit_idx]

    k = min(expected, len(X)) if expected > 0 else estimate_num_speakers(X, min(max_k, len(X) - 1))
    info["k_initial"] = k
    if k <= 1:
        return np.zeros(n, dtype=int), info

    from sklearn.cluster import SpectralClustering
    fit_labels = SpectralClustering(n_clusters=k, affinity="precomputed",
                                    random_state=0).fit_predict(_affinity(X))
    ids = np.unique(fit_labels)
    cents = np.array([_unit(X[fit_labels == c].mean(axis=0)) for c in ids])
    labels = ids[np.argmax(emb @ cents.T, axis=1)]  # assign ALL windows to nearest centroid

    if expected == 0:
        labels = _merge_clusters(emb, labels, merge_thr, min_share)

    ids = np.unique(labels)
    if len(ids) > 1:
        cents = np.array([_unit(emb[labels == c].mean(axis=0)) for c in ids])
        dist = 1.0 - cents @ cents.T
        np.fill_diagonal(dist, np.inf)
        info["closest_pair"] = float(dist.min())
    return labels, info


def assign_from_windows(n_segments, seg_ids, labels):
    """Majority vote of window labels within each segment; segments with no
    usable window inherit the nearest labelled segment."""
    seg_label = [None] * n_segments
    for i in np.unique(seg_ids):
        seg_label[int(i)] = int(np.bincount(labels[seg_ids == i]).argmax())
    have = [i for i, v in enumerate(seg_label) if v is not None]
    if not have:
        return [0] * n_segments
    for i in range(n_segments):
        if seg_label[i] is None:
            seg_label[i] = seg_label[min(have, key=lambda j: abs(j - i))]
    return seg_label


def fix_isolated_flips(segs, seg_label, max_dur=2.5):
    """A short segment sandwiched between two segments of the SAME other
    speaker is almost always an embedding glitch, not a real turn."""
    out = list(seg_label)
    for i in range(1, len(segs) - 1):
        if (out[i - 1] == out[i + 1] != out[i]
                and segs[i]["end"] - segs[i]["start"] < max_dur):
            out[i] = out[i - 1]
    return out


def apply_speaker_labels(segs, seg_label):
    """Rename to 'Speaker N' by order of first appearance."""
    order = []
    for v in seg_label:
        if v not in order:
            order.append(v)
    remap = {old: f"Speaker {i + 1}" for i, old in enumerate(order)}
    for s, v in zip(segs, seg_label):
        s["speaker"] = remap[v]
    return len(order)


def talk_shares(segs):
    tot = defaultdict(float)
    for s in segs:
        tot[s["speaker"]] += max(0.0, s["end"] - s["start"])
    total = sum(tot.values()) or 1.0
    return {k: round(100 * v / total, 1) for k, v in sorted(tot.items())}


def diarize_segments(audio_path, segments, engine, expected, max_speakers,
                     merge_thr, min_share, hf_token, cache):
    """Returns (segments_with_speaker, num_speakers, info, cache)."""
    segs = [dict(s) for s in segments]
    info = {"engine": engine, "warning": None, "closest_pair": None}
    if not segs:
        return segs, 1, info, cache

    if engine == "pyannote":
        try:
            turns = run_pyannote(audio_path, expected, max_speakers, hf_token)
            seg_label = fix_isolated_flips(segs, labels_from_turns(segs, turns))
            n = apply_speaker_labels(segs, seg_label)
            info["shares"] = talk_shares(segs)
            return segs, n, info, cache
        except Exception as e:
            info["warning"] = (f"pyannote engine failed ({e}). Fell back to the "
                               f"lightweight engine.")
            info["engine"] = "lightweight"

    if cache is None or "emb" not in cache:
        cache = compute_window_embeddings(audio_path, segs)
    emb, seg_ids = cache["emb"], cache["seg_ids"]
    if len(emb) < 2:
        for s in segs:
            s["speaker"] = "Speaker 1"
        info["shares"] = talk_shares(segs)
        return segs, 1, info, cache

    labels, cinfo = cluster_windows(emb, expected=expected, max_k=max_speakers,
                                    merge_thr=merge_thr, min_share=min_share)
    seg_label = assign_from_windows(len(segs), seg_ids, labels)
    seg_label = fix_isolated_flips(segs, seg_label)
    n = apply_speaker_labels(segs, seg_label)
    info.update(cinfo)
    info["shares"] = talk_shares(segs)
    return segs, n, info, cache


# ----------------------------------------------------------------------------
# Groq helpers: retry/backoff, pacing, JSON mode
# ----------------------------------------------------------------------------
_TPM = {"budget": 0, "log": deque()}


def _throttle(est_tokens: int):
    """Optional proactive pacing so we don't burn the per-minute token budget."""
    budget = _TPM["budget"]
    if not budget:
        return
    est = min(est_tokens, budget)
    log = _TPM["log"]
    while True:
        now = time.time()
        while log and now - log[0][0] > 60:
            log.popleft()
        if sum(t for _, t in log) + est <= budget:
            break
        time.sleep(max(0.5, 60 - (now - log[0][0]) + 0.2))
    log.append((time.time(), est))


def is_groq_limit_error(e: Exception) -> bool:
    status = getattr(e, "status_code", None)
    msg = str(e).lower()
    return (
        status in (413, 429)
        or type(e).__name__ == "RateLimitError"
        or any(k in msg for k in ["rate limit", "rate_limit", "tokens per", "request too large",
                                  "context length", "context_length", "quota", "too large"])
    )


def is_too_large_error(e: Exception) -> bool:
    msg = str(e).lower()
    return getattr(e, "status_code", None) == 413 or "too large" in msg or "context length" in msg \
        or "context_length" in msg


def _retry_after_seconds(e: Exception, default: float) -> float:
    try:
        hdr = e.response.headers.get("retry-after")
        if hdr:
            return float(hdr)
    except Exception:
        pass
    m = re.search(r"try again in\s*(?:(\d+)m)?\s*([\d.]+)s", str(e), re.I)
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2))
    return default


def _groq_chat(api_key, model, prompt, max_tokens=400, json_mode=False,
               max_retries=5, max_wait=90.0) -> str:
    """429 -> wait as long as Groq asks and retry. 'Too large' / long waits
    (e.g. daily quota) -> raise immediately so the caller can adapt."""
    from groq import Groq
    client = Groq(api_key=api_key, max_retries=0)
    is_reasoning = "gpt-oss" in model
    # Reasoning models spend part of max_tokens on hidden thinking; give headroom.
    budget_tokens = max_tokens + (1200 if is_reasoning else 0)
    kwargs = {}
    if is_reasoning:
        kwargs["extra_body"] = {"reasoning_effort": "low"}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    est = int(len(prompt) / 3) + budget_tokens

    for attempt in range(max_retries + 1):
        _throttle(est)
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=budget_tokens, **kwargs)
            try:
                if _TPM["budget"] and _TPM["log"]:
                    _TPM["log"][-1] = (_TPM["log"][-1][0], int(resp.usage.total_tokens))
            except Exception:
                pass
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if "reasoning_effort" in str(e).lower() and "extra_body" in kwargs:
                kwargs.pop("extra_body")  # this model doesn't support it; retry without
                continue
            if is_too_large_error(e) or not is_groq_limit_error(e) or attempt == max_retries:
                raise
            wait = _retry_after_seconds(e, default=2 ** attempt * 2)
            if wait > max_wait:
                raise
            time.sleep(wait + random.uniform(0.5, 1.5))


def _parse_json(raw: str):
    t = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object in model output")
    return json.loads(t[a:b + 1])


def _groq_json(api_key, model, prompt, max_tokens):
    """JSON mode first; if the model/API rejects it or the output isn't valid
    JSON, retry once in plain mode with lenient parsing. Limit errors propagate."""
    last = None
    for json_mode in (True, False):
        try:
            return _parse_json(_groq_chat(api_key, model, prompt, max_tokens, json_mode=json_mode))
        except Exception as e:
            if is_groq_limit_error(e):
                raise
            last = e
    raise last


@st.cache_data(ttl=600, show_spinner=False)
def fetch_groq_models(api_key: str):
    from groq import Groq
    resp = Groq(api_key=api_key).models.list()
    ids = [m.id for m in resp.data]
    exclude = ["whisper", "tts", "guard", "moderation", "distil"]
    chat = sorted(i for i in ids if not any(k in i.lower() for k in exclude))
    return chat or sorted(ids)


def pick_default(available, preferred):
    for p in preferred:
        if p in available:
            return p
    return available[0] if available else None


# ----------------------------------------------------------------------------
# Translation to English: Groq (batched) with free Google Translate fallback
# ----------------------------------------------------------------------------
def google_translate_to_english(text: str) -> str:
    from deep_translator import GoogleTranslator
    text = text.strip()
    if not text:
        return text
    try:
        return GoogleTranslator(source="auto", target="en").translate(text[:4900]) or text
    except Exception:
        return text


def _google_many(texts):
    with ThreadPoolExecutor(max_workers=4) as ex:
        return list(ex.map(google_translate_to_english, texts))


def translate_texts(api_key, model, texts, use_groq=True, batch_size=15, progress=None):
    """Returns (english_texts, stats). Once Groq fails in a way retries can't
    fix, the rest of the meeting switches to Google Translate."""
    out = list(texts)
    stats = {"groq": 0, "google": 0}
    groq_ok = bool(api_key) and use_groq
    total = max(1, len(texts))

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        done = False
        if groq_ok:
            numbered = "\n".join(f"{j + 1}. {t.strip()}" for j, t in enumerate(batch))
            prompt = (
                "Translate each numbered line below into natural, fluent English.\n"
                "Rules:\n- Return ONLY a numbered list, one translation per line.\n"
                f"- Keep the EXACT SAME numbering (1 to {len(batch)}). Do not skip, merge or add lines.\n"
                "- Keep each translation on a single line.\n- No preamble or commentary.\n\n" + numbered)
            try:
                resp = _groq_chat(api_key, model, prompt, max_tokens=max(500, 120 * len(batch)))
                parsed = {}
                for line in resp.split("\n"):
                    m = re.match(r"\s*(\d+)[\.\)]\s*(.*)", line)
                    if m:
                        idx, val = int(m.group(1)) - 1, m.group(2).strip()
                        if 0 <= idx < len(batch) and val:
                            parsed[idx] = val
                missing = [j for j in range(len(batch)) if j not in parsed]
                fixed = dict(zip(missing, _google_many([batch[j] for j in missing]))) if missing else {}
                for j in range(len(batch)):
                    out[start + j] = parsed.get(j) or fixed[j]
                stats["groq"] += len(parsed)
                stats["google"] += len(fixed)
                done = True
            except Exception:
                groq_ok = False
        if not done:
            for j, t in enumerate(_google_many(batch)):
                out[start + j] = t
            stats["google"] += len(batch)
        if progress:
            progress(min(1.0, (start + len(batch)) / total), "Translating to English...")
    return out, stats


# ----------------------------------------------------------------------------
# Meeting intelligence: chunked map-reduce, one JSON call per chunk
# ----------------------------------------------------------------------------
CHUNK_PROMPT = """You are an expert meeting analyst. Below is part <<I>> of <<N>> of a timestamped English meeting transcript.
Return ONLY a valid JSON object (no markdown, no commentary) with exactly these keys:
{
  "summary": "3-6 sentence summary of THIS part: topics, decisions, outcome",
  "agenda": [{"title": "3-6 word title", "start_seconds": 0, "description": "one sentence"}],
  "decisions": ["each clear decision that was made"],
  "action_items": [{"owner": "person name or 'Unassigned'", "task": "what must be done", "deadline": "deadline or 'Not specified'"}]
}
Rules:
- Agenda items in chronological order, using the [123s] timestamps from the transcript.
- Only list REAL commitments or assignments as action items, not general statements.
- Use empty lists when nothing applies. Refer to speakers by the names shown.

Transcript:
<<TEXT>>"""


def _line(s):
    return f"[{int(s['start'])}s] {s.get('speaker', 'Speaker')}: {s['text'].strip()}"


def split_segments_by_words(segs, max_words):
    chunks, cur, count = [], [], 0
    for s in segs:
        w = len(s["text"].split()) + 6
        if cur and count + w > max_words:
            chunks.append(cur)
            cur, count = [], 0
        cur.append(s)
        count += w
    if cur:
        chunks.append(cur)
    return chunks


def _norm_chunk(d):
    agenda = []
    for it in d.get("agenda") or []:
        if isinstance(it, dict):
            try:
                start = float(it.get("start_seconds", 0))
            except (TypeError, ValueError):
                start = 0.0
            title = str(it.get("title", "")).strip()
            if title:
                agenda.append({"title": title, "start": start,
                               "description": str(it.get("description", "")).strip()})
    decisions = [str(x).strip() for x in (d.get("decisions") or []) if str(x).strip()]
    actions = []
    for it in d.get("action_items") or []:
        if isinstance(it, dict) and str(it.get("task", "")).strip():
            actions.append({"owner": str(it.get("owner", "Unassigned")).strip() or "Unassigned",
                            "task": str(it["task"]).strip(),
                            "deadline": str(it.get("deadline", "Not specified")).strip() or "Not specified"})
    return {"summary": str(d.get("summary", "")).strip(), "agenda": agenda,
            "decisions": decisions, "action_items": actions}


def _merge_chunk_data(a, b):
    return {"summary": (a["summary"] + " " + b["summary"]).strip(),
            "agenda": a["agenda"] + b["agenda"],
            "decisions": a["decisions"] + b["decisions"],
            "action_items": a["action_items"] + b["action_items"]}


def analyze_chunk_groq(api_key, model, chunk, i, n, depth=0):
    text = "\n".join(_line(s) for s in chunk)
    prompt = CHUNK_PROMPT.replace("<<I>>", str(i)).replace("<<N>>", str(n)).replace("<<TEXT>>", text)
    try:
        return _norm_chunk(_groq_json(api_key, model, prompt, max_tokens=1100))
    except Exception as e:
        # Request bigger than the model allows: split in half and merge results.
        if is_too_large_error(e) and depth < 3 and len(chunk) >= 4:
            mid = len(chunk) // 2
            return _merge_chunk_data(
                analyze_chunk_groq(api_key, model, chunk[:mid], i, n, depth + 1),
                analyze_chunk_groq(api_key, model, chunk[mid:], i, n, depth + 1))
        raise


# ---- local fallbacks ---------------------------------------------------------
def chunk_text(text: str, max_words: int = 650):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def summarize_with_bart(text: str, chunk_words: int = 600) -> str:
    if not text.strip():
        return "No speech detected to summarize."
    summarizer = load_bart_summarizer()

    def _sum(t, max_len=150, min_len=40):
        n = len(t.split())
        max_len = min(max_len, max(20, int(n * 0.6)))
        min_len = min(min_len, max(5, max_len // 2))
        return summarizer(t, max_length=max_len, min_length=min_len,
                          do_sample=False, truncation=True)[0]["summary_text"]

    while True:
        chunks = chunk_text(text, max_words=chunk_words)
        if len(chunks) == 1:
            return _sum(chunks[0], max_len=180, min_len=60)
        text = " ".join(_sum(c) for c in chunks)


def tfidf_agenda(segs):
    """Keyword-based topic blocks: split the meeting evenly in time and title
    each block with its most distinctive words (TF-IDF)."""
    if not segs:
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    t0, t1 = segs[0]["start"], segs[-1]["end"]
    n_blocks = int(np.clip(round((t1 - t0) / 300), 1, 8))
    n_blocks = max(1, min(n_blocks, len(segs)))
    edges = np.linspace(t0, t1 + 1e-6, n_blocks + 1)
    blocks = [[] for _ in range(n_blocks)]
    for s in segs:
        mid = (s["start"] + s["end"]) / 2
        blocks[min(n_blocks - 1, int(np.searchsorted(edges, mid, side="right") - 1))].append(s["text"])
    docs = [" ".join(b) for b in blocks]
    items = []
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=2000)
        X = vec.fit_transform(docs)
        terms = vec.get_feature_names_out()
        for b, doc in enumerate(docs):
            if not doc.strip():
                continue
            row = X[b].toarray()[0]
            top = [terms[j] for j in row.argsort()[::-1][:3] if row[j] > 0]
            items.append({"title": ", ".join(top).title() or "Discussion", "start": float(edges[b]),
                          "description": "Keyword-based topic block (LLM unavailable)."})
    except ValueError:
        items.append({"title": "Discussion", "start": float(t0), "description": ""})
    return items


ACTION_CUES = [
    r"\bwill\b", r"\bneed(s)? to\b", r"\bshould\b", r"\bmust\b", r"\bhave to\b",
    r"\baction item\b", r"\bto( |-)do\b", r"\bfollow(ing)? up\b", r"\bassign(ed)?\b",
    r"\bby (tomorrow|next week|monday|tuesday|wednesday|thursday|friday|end of day|eod)\b",
    r"\blet'?s\b.*\b(do|start|finish|send|prepare|schedule|review)\b",
    r"\bplease\b.*\b(send|share|prepare|review|update|check)\b",
]
ACTION_PATTERN = re.compile("|".join(ACTION_CUES), re.IGNORECASE)


def rule_based_action_items(segs):
    """Fallback only: cue-phrase matching + spaCy entities for deadlines."""
    try:
        nlp = load_spacy_model()
    except Exception:
        nlp = None
    items = []
    for seg in segs:
        sentences = [seg["text"]]
        ents_by_sent = {}
        if nlp:
            doc = nlp(seg["text"])
            sentences = [s.text.strip() for s in doc.sents]
            ents_by_sent = {s.text.strip(): [e.text for e in s.ents if e.label_ in ("DATE", "TIME")]
                            for s in doc.sents}
        for sent in sentences:
            if len(sent.split()) >= 4 and ACTION_PATTERN.search(sent):
                dl = ", ".join(ents_by_sent.get(sent, [])) or "Not specified"
                items.append({"owner": seg.get("speaker", "Unassigned"), "task": sent, "deadline": dl})
    return items


def local_chunk_fallback(chunk):
    text = " ".join(s["text"].strip() for s in chunk)
    try:
        summary = summarize_with_bart(text)
    except Exception:
        summary = " ".join(re.split(r"(?<=[.!?])\s+", text)[:4])
    return {"summary": summary, "agenda": tfidf_agenda(chunk), "decisions": [],
            "action_items": rule_based_action_items(chunk)}


# ---- reduce steps ------------------------------------------------------------
def reduce_summaries(api_key, model, summaries):
    parts = [s for s in summaries if s]
    if len(parts) <= 1 or not api_key:
        return " ".join(parts)
    try:
        while len(parts) > 1:
            groups = [parts[i:i + 8] for i in range(0, len(parts), 8)]
            merged = []
            for g in groups:
                if len(g) == 1:
                    merged.append(g[0])
                    continue
                prompt = ("Combine these partial summaries of consecutive parts of ONE meeting into a "
                          "single coherent summary of 5-8 sentences (main topics, key decisions, overall "
                          "outcome). Remove repetition. Plain text only.\n\n" + "\n\n".join(g))
                merged.append(_groq_chat(api_key, model, prompt, max_tokens=600))
            parts = merged
        return parts[0]
    except Exception:
        return " ".join(summaries)


def consolidate_agenda(api_key, model, items):
    if len(items) <= 12 or not api_key:
        return items
    lines = "\n".join(f"{int(it['start'])}s | {it['title']} | {it['description']}" for it in items)
    prompt = ("Below are agenda items extracted from consecutive parts of one meeting (topics may be split "
              "across parts). Merge duplicates/near-duplicates and return AT MOST 10 items in chronological "
              "order as JSON: {\"agenda\": [{\"title\": \"...\", \"start_seconds\": 0, \"description\": \"...\"}]}\n\n"
              + lines)
    try:
        return _norm_chunk({"agenda": _groq_json(api_key, model, prompt, 900).get("agenda", [])})["agenda"] or items
    except Exception:
        return items


def analyze_meeting(en_segs, api_key, model, chunk_words, progress=None):
    chunks = split_segments_by_words(en_segs, chunk_words)
    parts, notes, engines = [], [], set()
    groq_alive = bool(api_key)
    for i, ch in enumerate(chunks, 1):
        if progress:
            progress((i - 1) / max(1, len(chunks)), f"Analyzing part {i} of {len(chunks)}...")
        data = None
        if groq_alive:
            try:
                data = analyze_chunk_groq(api_key, model, ch, i, len(chunks))
                engines.add("groq")
            except Exception as e:
                notes.append(f"Part {i}: Groq unavailable ({str(e)[:120]}); used local fallback.")
                if is_groq_limit_error(e):
                    groq_alive = False  # don't waste more time on a throttled/exhausted API
        if data is None:
            data = local_chunk_fallback(ch)
            engines.add("local")
        parts.append(data)

    summary = reduce_summaries(api_key if groq_alive else None, model, [p["summary"] for p in parts])
    agenda = sorted((it for p in parts for it in p["agenda"]), key=lambda x: x["start"])
    seen, dedup = set(), []
    for it in agenda:
        key = it["title"].lower()
        if key not in seen:
            seen.add(key)
            dedup.append(it)
    agenda = consolidate_agenda(api_key if groq_alive else None, model, dedup)

    decisions, seen = [], set()
    for p in parts:
        for d in p["decisions"]:
            if d.lower() not in seen:
                seen.add(d.lower())
                decisions.append(d)
    actions, seen = [], set()
    for p in parts:
        for a in p["action_items"]:
            if a["task"].lower() not in seen:
                seen.add(a["task"].lower())
                actions.append(a)

    engine = "groq" if engines == {"groq"} else "local" if engines == {"local"} else "mixed"
    return {"summary": summary or "No speech detected to summarize.", "agenda": agenda,
            "decisions": decisions, "action_items": actions, "engine": engine, "notes": notes}


# ----------------------------------------------------------------------------
# Sentiment + speaker insights
# ----------------------------------------------------------------------------
def score_sentiment(texts, engine_choice):
    """Returns (list of (score in [-1,1], label), engine_used)."""
    if engine_choice.startswith("Transformer"):
        try:
            pipe = load_sentiment_model()
            outs = pipe([t if t.strip() else "." for t in texts], batch_size=16)
            res = []
            for o in outs:
                d = {x["label"].lower(): x["score"] for x in o}
                score = d.get("positive", 0.0) - d.get("negative", 0.0)
                res.append((float(score), max(d, key=d.get)))
            return res, "RoBERTa"
        except Exception:
            pass  # fall through to TextBlob
    from textblob import TextBlob
    res = []
    for t in texts:
        p = float(TextBlob(t).sentiment.polarity)
        res.append((p, "positive" if p > 0.1 else "negative" if p < -0.1 else "neutral"))
    return res, "TextBlob"


def sentiment_label(score: float) -> str:
    if score > 0.15:
        return "🙂 Positive"
    if score < -0.15:
        return "🙁 Negative"
    return "😐 Neutral"


def compute_speaker_insights(segs, en_texts, sentiments):
    stats = defaultdict(lambda: {"talk": 0.0, "turns": 0, "words": 0, "scores": [], "labels": []})
    prev = None
    for s, t, (score, label) in zip(segs, en_texts, sentiments):
        spk = s["speaker"]
        st_ = stats[spk]
        st_["talk"] += max(0.0, s["end"] - s["start"])
        if spk != prev:
            st_["turns"] += 1  # consecutive segments by one speaker = one turn
        prev = spk
        w = len(t.split())
        st_["words"] += w
        if w >= 4:  # ignore "yeah", "okay" etc. for sentiment
            st_["scores"].append(score)
            st_["labels"].append(label)

    total = sum(v["talk"] for v in stats.values()) or 1.0
    rows = []
    for spk, v in stats.items():
        n = len(v["labels"]) or 1
        pct = lambda lab: round(100 * v["labels"].count(lab) / n, 1)
        rows.append({
            "Speaker": spk,
            "Talk Time (s)": round(v["talk"], 1),
            "Talk Time (%)": round(100 * v["talk"] / total, 1),
            "Turns": v["turns"], "Words": v["words"],
            "Avg Sentiment": round(float(np.mean(v["scores"])) if v["scores"] else 0.0, 3),
            "Positive %": pct("positive"), "Neutral %": pct("neutral"), "Negative %": pct("negative"),
        })
    return pd.DataFrame(rows).sort_values("Talk Time (%)", ascending=False).reset_index(drop=True)


def sentiment_timeline_fig(timeline):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    speakers = sorted({p["spk"] for p in timeline})
    for i, spk in enumerate(speakers):
        pts = [p for p in timeline if p["spk"] == spk]
        ax.scatter([p["t"] / 60 for p in pts], [p["score"] for p in pts], s=18, alpha=0.7,
                   color=SPEAKER_COLORS[i % len(SPEAKER_COLORS)], label=spk)
    if len(timeline) >= 5:
        sc = np.array([p["score"] for p in timeline])
        k = min(7, len(sc))
        roll = np.convolve(sc, np.ones(k) / k, mode="same")
        ax.plot([p["t"] / 60 for p in timeline], roll, color="black", linewidth=1.5, label="Overall trend")
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_xlabel("Minutes into meeting")
    ax.set_ylabel("Sentiment (-1 to 1)")
    ax.set_title("Sentiment over time")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(lines)


def agenda_to_markdown(items):
    if not items:
        return "_No agenda could be generated._"
    return "\n".join(
        f"{i}. **{it['title']}** (~{fmt_time(it['start'])})" + (f" — {it['description']}" if it["description"] else "")
        for i, it in enumerate(items, 1))


def build_report(res) -> str:
    L = ["# Meeting Report\n"]
    L.append(f"_Spoken language: **{res['lang_name']}** · **{res['num_speakers']}** speaker(s) · "
             f"analysis engine: **{res['engine']}** · sentiment: **{res['sent_engine']}**. "
             f"Transcript is in English._\n")
    L.append("## Summary\n\n" + res["summary"] + "\n")
    L.append("## Key Decisions\n")
    L.append("\n".join(f"- {d}" for d in res["decisions"]) if res["decisions"] else "_No explicit decisions detected._")
    L.append("\n## Agenda\n\n" + agenda_to_markdown(res["agenda"]) + "\n")
    L.append("## Action Items\n")
    if res["action_items"]:
        L.append(df_to_markdown(pd.DataFrame(res["action_items"]).rename(
            columns={"owner": "Owner", "task": "Task", "deadline": "Deadline"})))
    else:
        L.append("_No clear action items detected._")
    L.append("\n## Speaker Insights\n\n" + df_to_markdown(res["insights_df"]) + "\n")
    L.append("## Full Transcript (English)\n\n" + res["english_transcript"])
    return "\n".join(L)


# ----------------------------------------------------------------------------
# Pipeline steps (called from the UI)
# ----------------------------------------------------------------------------
def make_progress(bar):
    return lambda f, t: bar.progress(float(min(max(f, 0.0), 1.0)), text=t)


def run_step1(uploaded_video, cfg):
    ss = st.session_state
    workdir = tempfile.mkdtemp(prefix="meeting_")
    ss["workdir"] = workdir
    video_path = os.path.join(workdir, "input_" + os.path.basename(uploaded_video.name))
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    audio_path = os.path.join(workdir, "audio.wav")

    with st.spinner("Extracting audio..."):
        extract_audio(video_path, audio_path)
        os.remove(video_path)

    with st.spinner(f"Transcribing with Whisper ({cfg['model_size']})... this can take a while"):
        model = load_whisper_model(cfg["model_size"])
        result = transcribe_audio(model, audio_path, cfg["language"])
    lang_code = result.get("language", "en")
    segments = clean_segments(result.get("segments", []))
    if not segments:
        st.error("No usable speech was detected (silence/noise/music?). Try a clearer recording.")
        return False

    ss["asr"] = {
        "audio_path": audio_path, "segments": segments, "lang_code": lang_code,
        "lang_name": get_language_name(lang_code),
        "needs_translation": lang_code != "en" or any(has_non_latin_chars(s["text"]) for s in segments),
    }
    redetect_speakers(cfg)
    return True


def redetect_speakers(cfg):
    ss = st.session_state
    asr = ss["asr"]
    engine = resolve_engine(cfg["engine_choice"])
    with st.spinner(f"Identifying speakers ({engine} engine)..."):
        segs, n, info, cache = diarize_segments(
            asr["audio_path"], asr["segments"], engine, cfg["expected_speakers"],
            cfg["max_speakers"], cfg["merge_thr"], cfg["min_share"], HF_TOKEN, ss.get("win_cache"))
    ss["win_cache"] = cache
    ss["diar"] = {"segments": segs, "num_speakers": n, "info": info}
    ss.pop("results", None)


def run_step2(cfg):
    ss = st.session_state
    asr, diar = ss["asr"], ss["diar"]
    names = {spk: (ss.get(f"name_{spk}", "").strip() or spk)
             for spk in sorted({s["speaker"] for s in diar["segments"]})}
    segs = [dict(s, speaker=names[s["speaker"]]) for s in diar["segments"]]
    native_texts = [s["text"].strip() for s in segs]
    bar = st.progress(0.0, text="Starting...")
    prog = make_progress(bar)

    # 1) English text (cached across re-runs)
    tstats = {"groq": 0, "google": 0}
    if "en_texts" not in asr:
        if asr["needs_translation"]:
            en, tstats = translate_texts(
                GROQ_API_KEY, cfg["translation_model"], native_texts,
                use_groq=not cfg["google_only"],
                progress=lambda f, t: prog(0.3 * f, t))
        else:
            en = native_texts
        asr["en_texts"] = en
    en_texts = asr["en_texts"]
    if tstats["google"] and GROQ_API_KEY and not cfg["google_only"]:
        st.warning(f"⚠️ Groq was throttled or unavailable — {tstats['google']} segment(s) were "
                   f"translated with Google Translate instead.")

    # 2) sentiment (cached)
    prog(0.32, "Scoring sentiment...")
    if "sent" not in asr or asr.get("sent_choice") != cfg["sentiment_choice"]:
        asr["sent"], asr["sent_engine"] = score_sentiment(en_texts, cfg["sentiment_choice"])
        asr["sent_choice"] = cfg["sentiment_choice"]
    sentiments = asr["sent"]

    # 3) summary / agenda / decisions / action items
    en_segs = [dict(s, text=t) for s, t in zip(segs, en_texts)]
    intel = analyze_meeting(
        en_segs, GROQ_API_KEY or None, cfg["analysis_model"], cfg["chunk_words"],
        progress=lambda f, t: prog(0.4 + 0.6 * f, t))
    bar.progress(1.0, text="Done")

    insights_df = compute_speaker_insights(segs, en_texts, sentiments)
    timeline = [{"t": (s["start"] + s["end"]) / 2, "spk": s["speaker"], "score": sc[0]}
                for s, t, sc in zip(segs, en_texts, sentiments) if len(t.split()) >= 4]

    fmt = lambda s, t: f"**[{s['speaker']} | {fmt_time(s['start'])}]:** {t}"
    ss["results"] = {
        **intel,
        "insights_df": insights_df, "timeline": timeline,
        "english_transcript": "\n\n".join(fmt(s, t.strip()) for s, t in zip(segs, en_texts)),
        "romanized_transcript": "\n\n".join(fmt(s, romanize_text(s["text"].strip())) for s in segs),
        "original_script_transcript": "\n\n".join(fmt(s, s["text"].strip()) for s in segs),
        "lang_name": asr["lang_name"], "num_speakers": diar["num_speakers"],
        "sent_engine": asr["sent_engine"],
    }


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def render_sidebar():
    cfg = {}
    with st.sidebar:
        st.header("⚙️ Settings")
        cfg["model_size"] = st.selectbox(
            "Whisper model size", ["tiny", "base", "small", "medium"], index=2,
            help="Larger = more accurate but slower. 'small' or above is recommended for Hindi/Hinglish.")
        lang_label = st.selectbox("Spoken language", list(LANG_OPTIONS), index=0,
                                  help="Auto-detect looks only at the start of the audio. If it guesses "
                                       "wrong, pick the language here.")
        cfg["language"] = LANG_OPTIONS[lang_label]

        st.markdown("---")
        st.subheader("🧑‍🤝‍🧑 Speakers")
        cfg["engine_choice"] = st.selectbox(
            "Diarization engine",
            ["Auto (pyannote if HF_TOKEN set)", "pyannote (best accuracy)", "Lightweight (resemblyzer)"],
            help="pyannote needs `pip install pyannote.audio` and HF_TOKEN in .env.")
        cfg["expected_speakers"] = int(st.number_input(
            "Expected number of speakers (0 = auto-detect)", 0, 12, 0,
            help="If you know how many people were in the meeting, set it. This is the most reliable way "
                 "to get speakers right."))
        with st.expander("Advanced diarization"):
            cfg["max_speakers"] = st.slider("Max speakers (auto mode)", 2, 12, 8)
            cfg["merge_thr"] = st.slider(
                "Merge similar voices (lightweight engine)", 0.02, 0.40, 0.12, 0.02,
                help="Speakers whose voice centroids are closer than this are merged. Higher = fewer "
                     "speakers. After detection, the app shows the closest pair's distance to help tune this.")
            cfg["min_share"] = st.slider("Ignore speakers with < X% of speech", 0, 10, 3) / 100

        st.markdown("---")
        st.subheader("🤖 Groq LLM")
        cfg["analysis_model"] = cfg["translation_model"] = None
        cfg["chunk_words"], cfg["google_only"] = 1800, False
        if not GROQ_API_KEY:
            st.warning("No `GROQ_API_KEY` found — running in **local-only mode** (Google Translate, BART "
                       "summary, keyword agenda, rule-based action items). Add a key to `.env` for full quality.")
        else:
            st.success("Groq API key loaded ✅")
            try:
                models = fetch_groq_models(GROQ_API_KEY)
                a_def = pick_default(models, ["openai/gpt-oss-120b", "llama-3.3-70b-versatile",
                                              "openai/gpt-oss-20b", "qwen/qwen3.6-27b"])
                t_def = pick_default(models, ["llama-3.1-8b-instant", "openai/gpt-oss-20b", a_def])
                cfg["analysis_model"] = st.selectbox(
                    "Analysis model (summary/agenda/actions)", models,
                    index=models.index(a_def) if a_def in models else 0)
                cfg["translation_model"] = st.selectbox(
                    "Translation model", models,
                    index=models.index(t_def) if t_def in models else 0,
                    help="A small, fast model has higher rate limits and is enough for translation.")
            except Exception as e:
                st.warning(f"Couldn't fetch model list ({e}). Enter model IDs manually.")
                cfg["analysis_model"] = st.text_input("Analysis model ID", "openai/gpt-oss-120b")
                cfg["translation_model"] = st.text_input("Translation model ID", "llama-3.1-8b-instant")
            with st.expander("Advanced LLM options"):
                cfg["chunk_words"] = st.slider("Chunk size (words per LLM call)", 600, 3000, 1800, 100,
                                               help="Smaller = safer on free-tier limits, more calls.")
                _TPM["budget"] = int(st.number_input(
                    "Tokens-per-minute budget (0 = off)", 0, 200000, 0, 500,
                    help="Set to your model's TPM limit (see the Groq console) to pace requests "
                         "proactively instead of relying only on retries."))
                cfg["google_only"] = st.checkbox("Translate with Google only (saves Groq tokens)", False)

        st.markdown("---")
        cfg["sentiment_choice"] = st.selectbox(
            "Sentiment engine",
            ["Transformer (RoBERTa) — better, ~500 MB download on first use", "TextBlob — lightweight"])
        st.caption("🛟 Automatic fallbacks: Google Translate, BART, keyword agenda, rule-based actions.")
    return cfg


def render_speaker_panel(cfg):
    ss = st.session_state
    diar = ss["diar"]
    info = diar["info"]
    st.markdown("### 🧑‍🤝‍🧑 Step 1 result: speakers")
    c1, c2, c3 = st.columns(3)
    c1.metric("Speakers detected", diar["num_speakers"])
    c2.metric("Engine used", info["engine"])
    c3.metric("Language", ss["asr"]["lang_name"])
    if info.get("warning"):
        st.warning(info["warning"])
    if info.get("closest_pair") is not None:
        st.caption(f"Closest pair of detected voices is at cosine distance **{info['closest_pair']:.2f}**. "
                   f"If two 'speakers' are really one person, raise *Merge similar voices* above that value; "
                   f"if two real people got merged, lower it. Then click **Re-detect speakers**.")
    shares = info.get("shares", {})
    if shares:
        st.dataframe(pd.DataFrame({"Speaker": list(shares), "Talk share (%)": list(shares.values())}),
                     width="stretch", hide_index=True)

    with st.expander("Preview transcript with speaker labels"):
        for s in diar["segments"][:20]:
            st.markdown(f"**{s['speaker']}** `{fmt_time(s['start'])}` — {s['text']}")

    if st.button("🔄 Re-detect speakers (uses sidebar settings)"):
        redetect_speakers(cfg)
        st.rerun()

    st.markdown("**Rename speakers** (optional — names are used in the summary, action items and report):")
    cols = st.columns(min(4, max(1, diar["num_speakers"])))
    for i, spk in enumerate(sorted({s["speaker"] for s in diar["segments"]})):
        with cols[i % len(cols)]:
            st.text_input(spk, value=spk, key=f"name_{spk}")


def render_results():
    res = st.session_state["results"]
    tabs = st.tabs(["📝 Transcript", "🗒️ Agenda", "📋 Summary", "✅ Action Items",
                    "👥 Speaker Insights", "⬇️ Report"])
    if res["engine"] != "groq":
        st.caption("ℹ️ Some or all analysis used local fallbacks. " + " ".join(res["notes"]))

    with tabs[0]:
        st.subheader("Speaker-Labeled Transcript (English)")
        st.caption(f"Spoken language: {res['lang_name']} · {res['num_speakers']} speaker(s)")
        st.markdown(res["english_transcript"])
        with st.expander("Romanized (Hinglish-style) transcript"):
            st.markdown(res["romanized_transcript"])
        with st.expander("Original-script transcript"):
            st.markdown(res["original_script_transcript"])

    with tabs[1]:
        st.subheader("Meeting Agenda")
        st.markdown(agenda_to_markdown(res["agenda"]))

    with tabs[2]:
        st.subheader("Meeting Summary")
        st.write(res["summary"])
        st.subheader("Key Decisions")
        if res["decisions"]:
            for d in res["decisions"]:
                st.markdown(f"- {d}")
        else:
            st.caption("No explicit decisions detected.")

    with tabs[3]:
        st.subheader("Action Items")
        if res["action_items"]:
            st.dataframe(pd.DataFrame(res["action_items"]).rename(
                columns={"owner": "Owner", "task": "Task", "deadline": "Deadline"}),
                width="stretch", hide_index=True)
        else:
            st.info("No clear action items were detected.")

    with tabs[4]:
        st.subheader("Speaker Talk-Time & Sentiment")
        st.caption(f"Sentiment engine: {res['sent_engine']} (segments under 4 words are ignored).")
        df = res["insights_df"]
        st.dataframe(df, width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            ax.bar(df["Speaker"], df["Talk Time (%)"], color=SPEAKER_COLORS[:len(df)])
            ax.set_ylabel("Talk Time (%)")
            ax.set_title("Talk Time Share")
            st.pyplot(fig)
        with c2:
            fig2, ax2 = plt.subplots()
            ax2.bar(df["Speaker"], df["Positive %"], color="#18C6C6", label="Positive")
            ax2.bar(df["Speaker"], df["Neutral %"], bottom=df["Positive %"], color="#9E9E9E", label="Neutral")
            ax2.bar(df["Speaker"], df["Negative %"], bottom=df["Positive %"] + df["Neutral %"],
                    color="#FF6584", label="Negative")
            ax2.set_ylabel("% of segments")
            ax2.set_title("Sentiment Mix")
            ax2.legend(fontsize=7)
            st.pyplot(fig2)
        if res["timeline"]:
            st.pyplot(sentiment_timeline_fig(res["timeline"]))
        for _, r in df.iterrows():
            st.write(f"**{r['Speaker']}** — {sentiment_label(r['Avg Sentiment'])}, "
                     f"{r['Turns']} turns, {r['Words']} words")

    with tabs[5]:
        st.subheader("Download Full Report")
        report_md = build_report(res)
        st.download_button("⬇️ Download Report (Markdown)", data=report_md,
                           file_name="meeting_report.md", mime="text/markdown")
        st.text_area("Preview", report_md, height=400)


def reset_for_new_file(file_key):
    ss = st.session_state
    if ss.get("file_key") != file_key:
        shutil.rmtree(ss.get("workdir", ""), ignore_errors=True)
        for k in ["asr", "diar", "results", "win_cache", "workdir"]:
            ss.pop(k, None)
        for k in [k for k in ss.keys() if k.startswith("name_")]:
            ss.pop(k, None)
        ss["file_key"] = file_key


def main():
    inject_custom_css()
    st.title("🗣️ AI Meeting Assistant")
    st.caption("Upload a meeting video to get a transcript, summary, agenda, action items and "
               "per-speaker sentiment — powered by NLP & speech AI.")
    cfg = render_sidebar()

    uploaded = st.file_uploader("Upload a meeting video", type=["mp4", "mov", "avi", "mkv", "webm"])
    if uploaded is None:
        st.info("👆 Upload a video file to begin.")
        render_nlp_summary()
        return

    reset_for_new_file(f"{uploaded.name}-{uploaded.size}")
    st.video(uploaded)
    ss = st.session_state

    if st.button("🎙️ Step 1 — Transcribe & identify speakers", type="primary"):
        if not check_ffmpeg_available():
            st.error("ffmpeg was not found. Install it and make sure it is on your PATH.")
            return
        try:
            for k in ["asr", "diar", "results", "win_cache"]:
                ss.pop(k, None)
            run_step1(uploaded, cfg)
        except Exception as e:
            st.error(f"Step 1 failed: {e}")
            return

    if "diar" in ss:
        render_speaker_panel(cfg)
        if st.button("✨ Step 2 — Generate summary, agenda, action items & report", type="primary"):
            try:
                run_step2(cfg)
                st.success("Done! Explore the results below.")
            except Exception as e:
                st.error(f"Step 2 failed: {e}")
                return

    if "results" in ss:
        render_results()

    st.markdown("---")
    render_nlp_summary()


def render_nlp_summary():
    with st.expander("📚 NLP & AI Techniques Used in This Project", expanded=False):
        st.markdown("""
| Stage | Technique | What it does |
|---|---|---|
| **Speech-to-Text** | OpenAI **Whisper** (encoder-decoder Transformer ASR) | Converts audio to timestamped text in the original language/script |
| **ASR Confidence Filtering** | Whisper `no_speech_prob` / `avg_logprob` | Drops segments hallucinated on silence, music or noise |
| **Speaker Diarization (best)** | **pyannote.audio** neural pipeline (voice activity detection, speaker-change segmentation, embedding, clustering, overlap handling) | Answers "who spoke when", then maps speaker turns onto Whisper segments by time overlap |
| **Speaker Diarization (lightweight)** | **d-vector embeddings** (resemblyzer) over sliding 1.6 s windows + **spectral clustering** with the **eigengap heuristic** for speaker count + cluster merging + isolated-flip cleanup | Local, no-account alternative; many windows per segment give a robust majority vote |
| **Script Romanization** | `indic_transliteration` + `unidecode` | Latin/"Hinglish-style" view of non-Latin transcripts |
| **Translation** | Groq LLM (batched numbered-list prompts) with **Google Translate** fallback | English text for analysis; automatically switches when Groq is throttled |
| **Chunking & Map-Reduce** | Word-bounded chunks, one JSON call each, hierarchical merge | Handles meetings of any length within LLM token limits; oversize chunks are split automatically |
| **Summary / Agenda / Decisions / Action Items** | **Groq-hosted LLM** with structured JSON output | One call per chunk extracts everything at once, with rate-limit retry/backoff and optional token pacing |
| **Local fallbacks** | **BART** summarization, **TF-IDF** topic blocks, **spaCy** + regex action cues | Everything still works if Groq is unavailable or no key is set |
| **Sentiment Analysis** | **RoBERTa** transformer (`cardiffnlp/twitter-roberta-base-sentiment-latest`) or TextBlob lexicon | Per-segment positive/neutral/negative scoring, per-speaker mix and a sentiment timeline |
| **Speaker Analytics** | Aggregation (talk-time %, turns, words) | Quantifies participation |

**Pipeline:** video → audio (ffmpeg) → Whisper transcript → hallucination filter → diarization (pyannote or
embedding clustering) → *[review & rename speakers]* → English translation (Groq/Google) → sentiment →
chunked LLM analysis (summary, agenda, decisions, actions) → speaker analytics → Markdown report.
        """)


if __name__ == "__main__":
    main()