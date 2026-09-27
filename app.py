"""
AI Meeting Assistant
=====================
A Streamlit app that takes an uploaded meeting video and produces:
  1. Full transcript (via OpenAI Whisper - automatic speech recognition)
  2. Speaker-labeled transcript (via unsupervised acoustic clustering /
     diarization, with the number of speakers auto-detected)
  3. Abstractive meeting summary (via a Groq-hosted LLM)
  4. Extracted action items (via rule-based + linguistic pattern matching
     with spaCy, on an English translation produced by the same LLM)
  5. Speaker insights (talk-time share, turn counts, sentiment per speaker)
  6. A downloadable Markdown report
  7. A final section explaining every NLP technique used in the pipeline

Setup: put your Groq API key in a `.env` file (see `.env.example`) or export
it as an environment variable: GROQ_API_KEY=gsk_...

Run with:  streamlit run app.py
See README.md for full setup instructions (ffmpeg, spaCy model, etc.)
"""

import os
import re
import tempfile
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Load GROQ_API_KEY (and anything else) from a local .env file, if present.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ----------------------------------------------------------------------------
# Page config + visual theme
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Meeting Assistant", page_icon="🗣️", layout="wide")


def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    :root {
        --accent-1: #7C4DFF;
        --accent-2: #18C6C6;
        --accent-3: #FF6584;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .main .block-container {
        animation: fadeInUp 0.6s ease-out;
        padding-top: 2rem;
    }

    /* Gradient animated title */
    h1 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, var(--accent-1), var(--accent-2), var(--accent-3), var(--accent-1));
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 6s ease infinite;
    }

    /* Buttons */
    div[data-testid="stButton"] button {
        background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 2px 10px rgba(124, 77, 255, 0.25);
    }
    div[data-testid="stButton"] button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 6px 18px rgba(124, 77, 255, 0.4);
    }
    div[data-testid="stDownloadButton"] button {
        background: linear-gradient(90deg, var(--accent-2), var(--accent-1));
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        transition: transform 0.15s ease;
    }
    div[data-testid="stDownloadButton"] button:hover { transform: translateY(-2px); }

    /* Tabs */
    button[role="tab"] {
        border-radius: 10px 10px 0 0 !important;
        font-weight: 600;
        transition: background 0.2s ease;
    }
    button[aria-selected="true"] {
        background: linear-gradient(90deg, rgba(124,77,255,0.15), rgba(24,198,198,0.15)) !important;
        border-bottom: 3px solid var(--accent-1) !important;
    }

    /* Alerts / info-success-error boxes */
    .stAlert {
        border-radius: 12px;
        animation: fadeInUp 0.5s ease-out;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(124,77,255,0.06), rgba(24,198,198,0.04));
        border-right: 1px solid rgba(124,77,255,0.15);
    }

    /* Dataframe container */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        animation: fadeInUp 0.6s ease-out;
    }

    /* Expander */
    details {
        border-radius: 10px !important;
        transition: box-shadow 0.2s ease;
    }
    details:hover { box-shadow: 0 2px 12px rgba(124,77,255,0.15); }
    </style>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Lazy / cached imports of heavy libraries
# (imported inside functions so the app gives a friendly error if a package
#  is missing, instead of crashing on startup)
# ----------------------------------------------------------------------------
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


def check_ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------
# Step 1: Extract audio from the uploaded video
# ----------------------------------------------------------------------------
def extract_audio(video_path: str, audio_path: str):
    """Extract mono 16kHz WAV audio from a video file using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1", "-vn", audio_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.decode(errors='ignore')}")


# ----------------------------------------------------------------------------
# Step 2: Transcription (ASR) with Whisper - native language/script
# ----------------------------------------------------------------------------
def transcribe_audio(model, audio_path: str):
    """Returns whisper result dict containing 'text' and 'segments'
    (each segment has start, end, text), in the ORIGINAL spoken language/script."""
    result = model.transcribe(audio_path, task="transcribe", verbose=False)
    return result


def get_language_name(lang_code: str) -> str:
    try:
        import whisper
        return whisper.tokenizer.LANGUAGES.get(lang_code, lang_code).title()
    except Exception:
        return lang_code


# ----------------------------------------------------------------------------
# Romanization: convert whatever script Whisper produced (Devanagari, Arabic/
# Urdu, Cyrillic, CJK, etc.) into Latin-script "Hinglish-style" text.
#
# Note: Whisper's language ID for Hindi is sometimes correct while the actual
# decoded script drifts into Urdu (Perso-Arabic) - Hindi and Urdu are the same
# spoken language (Hindustani) with different scripts, and Whisper's training
# data mixes both under similar language tags. Rather than fighting which
# script Whisper decides to emit, we detect the script actually produced and
# romanize it - so the displayed transcript is Hinglish-style either way.
# ----------------------------------------------------------------------------
def detect_script(text: str) -> str:
    if any("\u0900" <= c <= "\u097F" for c in text):
        return "devanagari"
    if any("\u0600" <= c <= "\u06FF" for c in text):
        return "arabic"
    return "other"


def romanize_text(text: str, lang_code: str = "") -> str:
    if not text or not text.strip():
        return text

    script = detect_script(text)

    if script == "devanagari":
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
# Step 3: Speaker diarization via speaker-embedding clustering
#   - For each ASR segment, extract a speaker embedding (d-vector) using a
#     pretrained voice-encoder model (resemblyzer) - trained via metric
#     learning specifically to separate WHO is speaking, unlike raw MFCC
#     statistics which mostly describe WHAT is being said (phonetic content).
#     This is a much closer approximation to real diarization systems than
#     generic acoustic features.
#   - The number of speakers is AUTO-DETECTED with silhouette-score analysis
#     over a range of candidate cluster counts (no manual input needed)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_voice_encoder():
    from resemblyzer import VoiceEncoder
    return VoiceEncoder()


def extract_speaker_embeddings(audio_path: str, segments, sr_target=16000):
    from resemblyzer import preprocess_wav
    import librosa

    encoder = load_voice_encoder()
    y, sr = librosa.load(audio_path, sr=sr_target, mono=True)
    embeddings = []
    valid_idx = []

    for i, seg in enumerate(segments):
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        clip = y[start_sample:end_sample]

        if len(clip) < sr * 0.3:  # too short for a reliable voice embedding
            continue

        try:
            processed = preprocess_wav(clip, source_sr=sr)
            if len(processed) < sr * 0.15:  # VAD trimmed almost everything away
                continue
            emb = encoder.embed_utterance(processed)
        except Exception:
            continue

        embeddings.append(emb)
        valid_idx.append(i)

    return np.array(embeddings), valid_idx


def estimate_num_speakers(embeddings: np.ndarray, max_speakers: int = 8) -> int:
    """Automatically pick the number of speakers by trying several cluster
    counts (cosine-distance agglomerative clustering, matching how speaker
    embeddings are normally compared) and choosing the one with the best
    silhouette score. Falls back to 1 speaker if no candidate produces a
    meaningful separation."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    n = len(embeddings)
    if n < 4:
        return 1

    max_k = min(max_speakers, n - 1)
    best_k, best_score = 1, -1.0

    for k in range(2, max_k + 1):
        try:
            clustering = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
            labels = clustering.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels, metric="cosine")
        except Exception:
            continue
        if score > best_score:
            best_score, best_k = score, k

    # Speaker embeddings separate more cleanly than raw MFCCs when there
    # really are multiple speakers, so a higher bar here means fewer false
    # splits of a single speaker into several "speakers".
    return best_k if best_score >= 0.15 else 1


def cluster_speakers(embeddings: np.ndarray, num_speakers: int):
    from sklearn.cluster import AgglomerativeClustering

    if len(embeddings) == 0:
        return np.array([])

    n_clusters = max(1, min(num_speakers, len(embeddings)))
    if n_clusters == 1:
        return np.zeros(len(embeddings), dtype=int)

    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    return clustering.fit_predict(embeddings)


def assign_speakers_auto(segments, audio_path: str, max_speakers: int = 8, manual_override: int = 0):
    """Attach a 'speaker' field to every segment, auto-detecting how many
    speakers there are (unless manual_override > 0 is given as an escape
    hatch). Returns (segments, num_speakers_detected)."""
    if len(segments) == 0:
        return segments, 1

    embeddings, valid_idx = extract_speaker_embeddings(audio_path, segments)

    if len(embeddings) < 4:
        for seg in segments:
            seg["speaker"] = "Speaker 1"
        return segments, 1

    num_speakers = manual_override if manual_override > 0 else estimate_num_speakers(embeddings, max_speakers)

    if num_speakers <= 1:
        for seg in segments:
            seg["speaker"] = "Speaker 1"
        return segments, 1

    labels = cluster_speakers(embeddings, num_speakers)

    # Re-order cluster ids by first time they appear, so "Speaker 1" is
    # whoever talks first, etc. (purely cosmetic, easier to read)
    first_seen_order = []
    idx_to_label = dict(zip(valid_idx, labels))
    for i in range(len(segments)):
        if i in idx_to_label and idx_to_label[i] not in first_seen_order:
            first_seen_order.append(idx_to_label[i])
    remap = {old: f"Speaker {new+1}" for new, old in enumerate(first_seen_order)}

    for i, seg in enumerate(segments):
        seg["speaker"] = remap.get(idx_to_label.get(i), "Speaker 1")

    return segments, num_speakers


# ----------------------------------------------------------------------------
# Groq LLM helpers - used for BOTH summarization and English translation
# ----------------------------------------------------------------------------
def fetch_groq_models(api_key: str):
    """Query Groq's live model catalog instead of hardcoding IDs, since Groq
    frequently deprecates/renames models. Filters out obviously non-chat
    models (audio/TTS/moderation) so the dropdown only shows usable options."""
    from groq import Groq
    client = Groq(api_key=api_key)
    resp = client.models.list()
    all_ids = [m.id for m in resp.data]
    exclude_kw = ["whisper", "tts", "guard", "moderation", "distil"]
    chat_ids = sorted(i for i in all_ids if not any(k in i.lower() for k in exclude_kw))
    return chat_ids or sorted(all_ids)


def pick_default_groq_model(available_models):
    preferred = ["openai/gpt-oss-120b", "qwen/qwen3.6-27b",
                 "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]
    for p in preferred:
        if p in available_models:
            return p
    return available_models[0] if available_models else None


def _groq_chat(api_key: str, model: str, prompt: str, max_tokens: int = 400) -> str:
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ----------------------------------------------------------------------------
# Step 4: Translation to English via Groq LLM (only runs for non-English audio)
#   - Batches segment texts into numbered-list prompts to keep API calls low
#   - Speaker labels are already attached to these exact segments, so no
#     timestamp-based re-alignment is needed (translation is 1:1 per segment)
# ----------------------------------------------------------------------------
def chunk_text(text: str, max_words: int = 650):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def translate_segments_with_groq(api_key: str, model: str, segments, batch_size: int = 40):
    if not segments:
        return segments

    translated = [dict(seg) for seg in segments]

    for start in range(0, len(segments), batch_size):
        batch = segments[start:start + batch_size]
        numbered = "\n".join(f"{j+1}. {seg['text'].strip()}" for j, seg in enumerate(batch))
        prompt = (
            "Translate each numbered line below into natural, fluent English. "
            "Return ONLY a numbered list with the SAME numbering, one translation "
            "per line, no extra commentary or explanation.\n\n" + numbered
        )
        try:
            response_text = _groq_chat(api_key, model, prompt, max_tokens=max(300, 60 * len(batch)))
            parsed = {}
            for line in response_text.strip().split("\n"):
                m = re.match(r"\s*(\d+)[\.\)]\s*(.*)", line)
                if m:
                    parsed[int(m.group(1)) - 1] = m.group(2).strip()
            for j, seg in enumerate(batch):
                if j in parsed and parsed[j]:
                    translated[start + j]["text"] = parsed[j]
                # else: keep original text as a safe fallback
        except Exception:
            pass  # keep originals for this batch on failure

    return translated


# ----------------------------------------------------------------------------
# Step 5: Summarization via Groq LLM (map-reduce for long transcripts)
# ----------------------------------------------------------------------------
def summarize_meeting_groq(api_key: str, model: str, full_text: str) -> str:
    if not full_text.strip():
        return "No speech detected to summarize."

    base_prompt = (
        "You are an assistant that writes clear, concise meeting summaries. "
        "Summarize the following meeting transcript in 4-8 sentences, covering "
        "the main topics discussed, key decisions, and overall outcome. Write "
        "in plain English.\n\nTranscript:\n{text}"
    )

    words = full_text.split()
    if len(words) <= 6000:
        return _groq_chat(api_key, model, base_prompt.format(text=full_text))

    chunks = chunk_text(full_text, max_words=6000)
    chunk_summaries = [_groq_chat(api_key, model, base_prompt.format(text=c)) for c in chunks]
    combined = " ".join(chunk_summaries)
    reduce_prompt = (
        "Combine these partial meeting summaries into one coherent overall "
        "summary (4-8 sentences), removing repetition:\n\n" + combined
    )
    return _groq_chat(api_key, model, reduce_prompt)


# ----------------------------------------------------------------------------
# Step 6: Action item extraction
#   - Sentence segmentation via spaCy
#   - Rule-based cue-phrase matching (modal verbs, imperative patterns,
#     commitment phrases) - a classic lightweight approach to task mining
#   - spaCy NER pulls out DATE/PERSON/ORG entities as owners / deadlines
# ----------------------------------------------------------------------------
ACTION_CUES = [
    r"\bwill\b", r"\bneed(s)? to\b", r"\bshould\b", r"\bmust\b", r"\bhave to\b",
    r"\baction item\b", r"\bto( |-)do\b", r"\bfollow(ing)? up\b", r"\bassign(ed)?\b",
    r"\bby (tomorrow|next week|monday|tuesday|wednesday|thursday|friday|end of day|eod)\b",
    r"\blet'?s\b.*\b(do|start|finish|send|prepare|schedule|review)\b",
    r"\bplease\b.*\b(send|share|prepare|review|update|check)\b",
]
ACTION_PATTERN = re.compile("|".join(ACTION_CUES), re.IGNORECASE)


def extract_action_items(nlp, segments):
    action_items = []
    for seg in segments:
        doc = nlp(seg["text"])
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text or len(sent_text.split()) < 3:
                continue
            if ACTION_PATTERN.search(sent_text):
                entities = [(ent.text, ent.label_) for ent in sent.ents
                            if ent.label_ in ("PERSON", "DATE", "ORG", "TIME")]
                action_items.append({
                    "speaker": seg.get("speaker", "Speaker 1"),
                    "text": sent_text,
                    "start": seg["start"],
                    "entities": entities,
                })
    return action_items


# ----------------------------------------------------------------------------
# Step 7: Speaker insights - talk time, turn counts, sentiment
# ----------------------------------------------------------------------------
def compute_speaker_insights(original_segments, translated_segments):
    from textblob import TextBlob

    stats = defaultdict(lambda: {"talk_time": 0.0, "turns": 0, "words": 0, "sentiments": []})

    # Talk-time & turn counts are language-agnostic - use original (native
    # audio) segment timestamps, which came straight from acoustic clustering.
    for seg in original_segments:
        spk = seg.get("speaker", "Speaker 1")
        duration = max(0.0, seg["end"] - seg["start"])
        stats[spk]["talk_time"] += duration
        stats[spk]["turns"] += 1

    # Word counts & sentiment are computed on the English translation, so
    # results are consistent and comparable regardless of input language.
    for seg in translated_segments:
        spk = seg.get("speaker", "Speaker 1")
        stats[spk]["words"] += len(seg["text"].split())
        polarity = TextBlob(seg["text"]).sentiment.polarity
        stats[spk]["sentiments"].append(polarity)

    rows = []
    total_time = sum(s["talk_time"] for s in stats.values()) or 1.0
    for spk, s in stats.items():
        avg_sent = float(np.mean(s["sentiments"])) if s["sentiments"] else 0.0
        rows.append({
            "Speaker": spk,
            "Talk Time (s)": round(s["talk_time"], 1),
            "Talk Time (%)": round(100 * s["talk_time"] / total_time, 1),
            "Turns": s["turns"],
            "Words": s["words"],
            "Avg Sentiment": round(avg_sent, 3),
        })
    return pd.DataFrame(rows).sort_values("Talk Time (%)", ascending=False)


def sentiment_label(score: float) -> str:
    if score > 0.15:
        return "🙂 Positive"
    elif score < -0.15:
        return "🙁 Negative"
    return "😐 Neutral"


# ----------------------------------------------------------------------------
# Step 8: Build a downloadable Markdown report
# ----------------------------------------------------------------------------
def build_report(summary, action_items, insights_df, diarized_transcript, lang_name, num_speakers) -> str:
    lines = ["# Meeting Report\n"]
    lines.append(f"_Detected spoken language: **{lang_name}**. Auto-detected "
                  f"**{num_speakers}** speaker(s). Transcript below is romanized "
                  f"(Latin script) for readability; summary, action items, and "
                  f"sentiment were computed from an English translation._\n")
    lines.append("## Summary\n")
    lines.append(summary + "\n")

    lines.append("## Action Items\n")
    if action_items:
        for a in action_items:
            tag = f" ({', '.join(f'{t}: {l}' for t, l in a['entities'])})" if a["entities"] else ""
            lines.append(f"- **[{a['speaker']}]** {a['text']}{tag}")
    else:
        lines.append("_No clear action items detected._")
    lines.append("")

    lines.append("## Speaker Insights\n")
    lines.append(insights_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Full Transcript\n")
    lines.append(diarized_transcript)

    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------
def main():
    inject_custom_css()

    st.title("🗣️ AI Meeting Assistant")
    st.caption("Upload a meeting video to get transcription, summarization, "
               "action items, and speaker insights — powered by NLP & speech AI.")

    with st.sidebar:
        st.header("⚙️ Settings")
        model_size = st.selectbox(
            "Whisper model size", ["tiny", "base", "small", "medium"], index=1,
            help="Larger = more accurate but slower. 'base' is a good default."
        )
        st.caption("🧑‍🤝‍🧑 Number of speakers is auto-detected — no need to set it.")
        with st.expander("Advanced"):
            manual_speaker_override = st.number_input(
                "Override speaker count (0 = auto-detect)", min_value=0, max_value=8, value=0,
                help="Leave at 0 to auto-detect. Set a specific number only if "
                     "auto-detection gets it wrong for this recording."
            )

        st.markdown("---")
        st.subheader("🤖 Groq LLM")

        groq_model = None
        if not GROQ_API_KEY:
            st.error(
                "No `GROQ_API_KEY` found. Add it to a `.env` file (see "
                "`.env.example`) or set it as an environment variable, then "
                "restart the app."
            )
        else:
            st.success("Groq API key loaded from environment ✅")
            try:
                available_models = fetch_groq_models(GROQ_API_KEY)
                default_model = pick_default_groq_model(available_models)
                groq_model = st.selectbox(
                    "Groq model", available_models,
                    index=available_models.index(default_model) if default_model in available_models else 0,
                    help="Fetched live from your Groq account, since Groq's "
                         "model lineup changes frequently."
                )
            except Exception as e:
                st.warning(f"Couldn't fetch model list from Groq ({e}). "
                           f"Enter a model ID manually instead.")
                groq_model = st.text_input("Groq model ID", value="openai/gpt-oss-120b")

        st.markdown("---")
        st.markdown(
            "**Requirements:** ffmpeg installed, and packages in "
            "`requirements.txt`. See README.md for setup."
        )

    uploaded_video = st.file_uploader(
        "Upload a meeting video", type=["mp4", "mov", "avi", "mkv", "webm"]
    )

    if uploaded_video is None:
        st.info("👆 Upload a video file to begin.")
        render_nlp_summary()
        return

    st.video(uploaded_video)

    if st.button("🚀 Process Meeting", type="primary"):
        if not GROQ_API_KEY:
            st.error("Please set GROQ_API_KEY (see sidebar) before processing.")
            return
        if not check_ffmpeg_available():
            st.error(
                "ffmpeg was not found on this system. Please install ffmpeg "
                "and ensure it's on your PATH, then rerun."
            )
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, uploaded_video.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            audio_path = os.path.join(tmpdir, "audio.wav")

            # --- Step 1: audio extraction ---
            with st.spinner("Extracting audio from video..."):
                extract_audio(video_path, audio_path)

            # --- Step 2: transcription (native language/script) ---
            with st.spinner(f"Transcribing with Whisper ({model_size})... this can take a while"):
                whisper_model = load_whisper_model(model_size)
                result = transcribe_audio(whisper_model, audio_path)
                segments = result["segments"]
                detected_lang_code = result.get("language", "en")
                detected_lang_name = get_language_name(detected_lang_code)

            if not segments:
                st.error("No speech was detected in this video.")
                return

            st.info(f"🌐 Detected spoken language: **{detected_lang_name}**")

            # --- Step 3: diarization with AUTO speaker-count detection ---
            with st.spinner("Identifying speakers (voice-embedding clustering)..."):
                segments, num_speakers_detected = assign_speakers_auto(
                    segments, audio_path, manual_override=manual_speaker_override
                )
            st.info(f"🧑‍🤝‍🧑 Auto-detected **{num_speakers_detected}** speaker(s)")

            # --- Step 4: translate to English via Groq (only if needed) ---
            if detected_lang_code == "en":
                translated_segments = [dict(seg) for seg in segments]
            else:
                with st.spinner(f"Translating to English (Groq: {groq_model})..."):
                    try:
                        translated_segments = translate_segments_with_groq(
                            GROQ_API_KEY, groq_model, segments
                        )
                    except Exception as e:
                        st.error(f"Groq translation failed: {e}")
                        return
            translated_full_text = " ".join(seg["text"].strip() for seg in translated_segments)

            # Romanized ("Hinglish-style") transcript for display - whatever
            # script Whisper produced gets transliterated to Latin script.
            diarized_transcript = "\n\n".join(
                f"**[{seg['speaker']} | {seg['start']:.1f}s]:** "
                f"{romanize_text(seg['text'].strip(), detected_lang_code)}"
                for seg in segments
            )
            original_script_transcript = "\n\n".join(
                f"**[{seg['speaker']} | {seg['start']:.1f}s]:** {seg['text'].strip()}"
                for seg in segments
            )

            # --- Step 5: summarization via Groq ---
            with st.spinner(f"Generating summary (Groq: {groq_model})..."):
                try:
                    summary = summarize_meeting_groq(GROQ_API_KEY, groq_model, translated_full_text)
                except Exception as e:
                    st.error(f"Groq summarization failed: {e}")
                    return

            # --- Step 6: action items (always in English) ---
            with st.spinner("Extracting action items (spaCy + rule matching)..."):
                nlp = load_spacy_model()
                action_items = extract_action_items(nlp, translated_segments)

            # --- Step 7: speaker insights (timing native, text English) ---
            with st.spinner("Computing speaker insights..."):
                insights_df = compute_speaker_insights(segments, translated_segments)

            st.session_state["results"] = {
                "summary": summary,
                "action_items": action_items,
                "insights_df": insights_df,
                "diarized_transcript": diarized_transcript,
                "original_script_transcript": original_script_transcript,
                "segments": segments,
                "lang_name": detected_lang_name,
                "num_speakers": num_speakers_detected,
            }
            st.success("Done! Explore the results below.")

    # ---- Display results if available ----
    if "results" in st.session_state:
        res = st.session_state["results"]
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📝 Transcript", "📋 Summary", "✅ Action Items", "👥 Speaker Insights", "⬇️ Report"]
        )

        with tab1:
            st.subheader("Speaker-Labeled Transcript (Romanized)")
            st.caption(f"Detected language: {res['lang_name']} · "
                       f"{res['num_speakers']} speaker(s) auto-detected. Shown in "
                       f"Latin/Hinglish-style script for readability.")
            st.markdown(res["diarized_transcript"])
            with st.expander("Show original-script transcript"):
                st.markdown(res["original_script_transcript"])

        with tab2:
            st.subheader("Meeting Summary")
            st.write(res["summary"])

        with tab3:
            st.subheader("Extracted Action Items")
            if res["action_items"]:
                for a in res["action_items"]:
                    tags = ", ".join(f"{t} ({l})" for t, l in a["entities"])
                    st.markdown(
                        f"- **[{a['speaker']} @ {a['start']:.1f}s]** {a['text']}"
                        + (f"  \n  _Detected entities: {tags}_" if tags else "")
                    )
            else:
                st.info("No clear action items were detected.")

        with tab4:
            st.subheader("Speaker Talk-Time & Sentiment")
            df = res["insights_df"]
            st.dataframe(df, width="stretch")

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                ax.bar(df["Speaker"], df["Talk Time (%)"], color="#7C4DFF")
                ax.set_ylabel("Talk Time (%)")
                ax.set_title("Talk Time Share by Speaker")
                st.pyplot(fig)

            with col2:
                fig2, ax2 = plt.subplots()
                colors = ["#18C6C6" if v > 0.15 else "#FF6584" if v < -0.15 else "#9E9E9E"
                          for v in df["Avg Sentiment"]]
                ax2.bar(df["Speaker"], df["Avg Sentiment"], color=colors)
                ax2.axhline(0, color="black", linewidth=0.5)
                ax2.set_ylabel("Avg Sentiment (-1 to 1)")
                ax2.set_title("Average Sentiment by Speaker")
                st.pyplot(fig2)

            for _, row in df.iterrows():
                st.write(f"**{row['Speaker']}** — {sentiment_label(row['Avg Sentiment'])}, "
                         f"{row['Turns']} turns, {row['Words']} words")

        with tab5:
            st.subheader("Download Full Report")
            report_md = build_report(
                res["summary"], res["action_items"], res["insights_df"],
                res["diarized_transcript"], res["lang_name"], res["num_speakers"]
            )
            st.download_button(
                "⬇️ Download Report (Markdown)",
                data=report_md,
                file_name="meeting_report.md",
                mime="text/markdown",
            )
            st.text_area("Preview", report_md, height=400)

    st.markdown("---")
    render_nlp_summary()


def render_nlp_summary():
    with st.expander("📚 NLP & AI Techniques Used in This Project", expanded=False):
        st.markdown("""
| Stage | Technique | What it does |
|---|---|---|
| **Speech-to-Text** | OpenAI **Whisper** (encoder-decoder Transformer, seq2seq ASR) | Converts raw audio waveforms into text transcripts with timestamps, in the original spoken language/script |
| **Script Romanization** | Devanagari→Latin transliteration (`indic_transliteration`) with a generic Unicode transliteration fallback (`unidecode`) | Converts whatever script Whisper decodes (Devanagari, Urdu/Perso-Arabic, Cyrillic, etc.) into readable Latin/"Hinglish-style" text for display |
| **Speaker Diarization** | **Speaker embeddings (d-vectors)** from a pretrained voice-encoder (`resemblyzer`) + **cosine-distance agglomerative clustering**, with speaker count auto-selected via **silhouette-score analysis** | Groups speech segments by *who* is likely speaking using a model trained via metric learning to separate voice identity from phonetic content — more robust than generic acoustic features (e.g. MFCC statistics), and automatically figures out how many speakers there are |
| **Speech Translation** | Groq-hosted LLM via prompted chat completion (batched, numbered-list translation) | Translates non-English transcript segments into English while preserving per-segment speaker labels exactly, so summary/action items/sentiment stay consistent regardless of input language |
| **Text Chunking** | Word-bounded chunking / map-reduce | Splits long transcripts to fit LLM context-length limits before summarizing |
| **Summarization** | **Groq-hosted LLM** (e.g. Llama 3.x / GPT-OSS) via prompted chat completion | Generates a fluent, human-like meeting summary using in-context instructions rather than a fixed fine-tuned model — a prompt-engineered, general-purpose LLM approach |
| **Sentence Segmentation & NER** | **spaCy** (dependency parsing + statistical NER) | Splits transcript into sentences and detects entities like PERSON, DATE, ORG to tag task owners/deadlines |
| **Action Item Mining** | **Rule-based pattern matching** (regex over modal verbs & commitment phrases: "will", "need to", "follow up", etc.) | A lightweight, explainable form of task/intent mining, common in real meeting-assistant products |
| **Sentiment Analysis** | **TextBlob** (lexicon-based polarity scoring) | Scores each speaker segment as positive/neutral/negative to build a per-speaker sentiment profile |
| **Speaker Analytics** | Aggregation & statistics (talk-time %, turn counts, word counts) | Quantifies participation and engagement per speaker |

**Pipeline in one sentence:** raw video → audio (signal processing) → native-language ASR transcript (Whisper) →
speaker-embedding clustering with auto-detected speaker count (resemblyzer + cosine agglomerative clustering + silhouette score) →
romanization for display + LLM-based English translation (Groq) → LLM-based abstractive summarization (Groq) →
rule-based action item + NER extraction (spaCy) → sentiment/talk-time analytics (TextBlob) → report.
        """)


if __name__ == "__main__":
    main()