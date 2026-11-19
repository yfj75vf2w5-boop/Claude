# B-roll reel pipeline

1. `transcribe.py <16k.wav> <out.json>` — local Whisper large-v3, word timestamps (kk).
2. `cuts.py <16k.wav> <whisper.json> <keep.json> 0.6` — VAD + word union, cuts every pause ≥ 200 ms.
3. `edl.py` — corrected sentences, gold keywords, 1.2x zoom sentences, B-roll animation scenes.
4. `render.py <work_dir> <out.mp4>` — clean cuts only, constant 1.0/1.2 zoom, person-masked hook
   (Inter Black + Playfair Italic), word-by-word subtitles (Whisper time + 0.1 s), white/red
   animation scenes, soft synth pad + soft ticks, loudnorm −14 LUFS. Text stays inside the
   Reels safe zone x 96–984, y 232–1393.

Times are voice-track seconds. When the clean `дауыс.mp3` arrives: cross-correlate it with the
camera audio, re-run steps 1–2 on it, and swap it in as the voice track.
