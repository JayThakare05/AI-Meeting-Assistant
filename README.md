# AI Meeting Assistant

An NLP mini-project: upload a meeting video and get transcription, speaker
diarization (with auto-detected speaker count), summarization, action items,
and speaker insights — all in one Streamlit app.

## 1. Install ffmpeg (required for audio extraction)

- **Windows:** download from https://ffmpeg.org/download.html and add to PATH
- **Mac:** `brew install ffmpeg`
- **Linux:** `sudo apt-get install ffmpeg`

## 2. Install Python dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> Note: `openai-whisper` and `torch` are large downloads. A GPU is optional
> but speeds up transcription a lot. If you only have CPU, use the `tiny` or
> `base` Whisper model size in the sidebar.

## 3. Set your Groq API key

This app uses [Groq](https://console.groq.com) for summarization and
translation (fast, free-tier LLM inference). Get a free API key at
https://console.groq.com/keys, then either:

**Option A — `.env` file (recommended):**
```bash
cp .env.example .env
# then edit .env and paste your key:
# GROQ_API_KEY=gsk_your_key_here
```

**Option B — environment variable:**
```bash
# Windows (PowerShell)
$env:GROQ_API_KEY="gsk_your_key_here"

# Mac/Linux
export GROQ_API_KEY="gsk_your_key_here"
```

The app reads the key automatically at startup — there's no API key field in
the UI, so it's never accidentally shared or logged.

## 4. Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501),
upload a meeting video (mp4/mov/avi/mkv/webm), and click **Process Meeting**.
The number of speakers is detected automatically — no need to set it.

## What it does

1. **Extracts audio** from the uploaded video with ffmpeg.
2. **Transcribes** the audio with OpenAI Whisper (speech-to-text), in the
   original spoken language/script.
3. **Diarizes speakers** by clustering MFCC acoustic features of each
   transcript segment with KMeans — the number of speakers is chosen
   automatically via silhouette-score analysis, so you never have to specify it.
4. **Translates to English** via a Groq-hosted LLM (only runs when the
   detected language isn't already English), preserving per-segment speaker
   labels exactly.
5. **Summarizes** the meeting with a Groq-hosted LLM (Llama 3.x / GPT-OSS,
   whichever is live on your account), using a chunk-then-reduce prompt
   strategy for long transcripts.
6. **Extracts action items** using spaCy sentence segmentation, NER, and
   rule-based cue-phrase matching (modal verbs, commitment phrases).
7. **Computes speaker insights**: talk-time share, turn counts, word counts,
   and sentiment (via TextBlob) per speaker.
8. Lets you **download a Markdown report** with everything combined.

See the "NLP & AI Techniques Used" expander at the bottom of the app (and in
the project report) for a full breakdown of every technique used, suitable
for your project write-up / viva.

## Auto speaker-count detection

Instead of asking you to guess how many people are in the meeting, the app:
1. Extracts a **speaker embedding** (d-vector) per transcript segment using a
   pretrained voice-encoder model (`resemblyzer`), trained via metric learning
   to separate voice identity from *what* is being said — a much stronger
   signal for "who is speaking" than generic acoustic features like MFCC
   statistics (an earlier version of this app used MFCC + KMeans, which could
   misfire by splitting one speaker into two whenever their tone/pace varied,
   or merging two similar-sounding speakers into one).
2. Tries clustering those embeddings into 2 through 8 groups using
   cosine-distance agglomerative clustering (the standard distance metric for
   comparing speaker embeddings).
3. Picks the group count with the best **silhouette score** (a measure of how
   cleanly separated the clusters are).
4. Falls back to "1 speaker" if no candidate count produces a clean separation.

If auto-detection still gets it wrong for a particular recording (background
noise, overlapping speech, very similar voices, etc.), open **Advanced** in
the sidebar and set an explicit speaker count as an override — it's optional
and defaults to auto.

This is a good thing to call out in your report as an unsupervised
model-selection technique, and the MFCC→embeddings switch is a nice concrete
example of feature-representation choice mattering for a downstream task.

## Groq for summary + translation

Both the meeting summary and the non-English → English translation are done
by a Groq-hosted LLM via prompted chat completion:
- **Translation** is batched: segments are numbered and sent together, and
  the model is asked to return translations with matching numbering, which
  keeps API calls low while preserving exact per-segment speaker attribution.
- **Summarization** uses a map-reduce prompt strategy for long transcripts
  (summarize each chunk, then summarize the summaries).
- The model dropdown is populated **live** from your Groq account (via
  `client.models.list()`), since Groq's model lineup changes frequently —
  no hardcoded model ID to go stale.

## Notes for your report

- Speaker diarization here uses speaker embeddings (resemblyzer) + cosine
  agglomerative clustering + silhouette-based speaker-count selection,
  instead of a full pretrained diarization pipeline (like pyannote.audio),
  which requires a gated HuggingFace token. Worth mentioning as a design
  decision/limitation — no clustering-based approach is as accurate as a
  purpose-built diarization system trained end-to-end on labeled multi-speaker
  audio.
- Whisper model size trades off speed vs. accuracy — good to mention in
  your evaluation section.
- Summarization here is LLM-based (prompt-engineered), which you can contrast
  with a fine-tuned seq2seq model like BART, or an extractive method like
  TextRank, in your report's related-work/comparison section.
- **Known Whisper quirk:** Whisper sometimes correctly identifies the spoken
  language as Hindi but decodes the text in Urdu (Perso-Arabic) script, since
  Hindi and Urdu are the same spoken language (Hindustani) with different
  scripts and Whisper's training data mixes both under similar language tags.
  The romanization step launders this away — whichever script gets produced,
  it's transliterated to Latin script for display.
