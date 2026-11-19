import sys, json
from faster_whisper import WhisperModel
import soundfile as sf, numpy as np
src, out = sys.argv[1], sys.argv[2]
audio, sr = sf.read(src, dtype="float32"); assert sr == 16000
if audio.ndim > 1: audio = audio.mean(1)
m = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=4)
segs, info = m.transcribe(audio, word_timestamps=True, beam_size=5, language="kk", vad_filter=False, condition_on_previous_text=False)
res = {"language": info.language, "prob": info.language_probability, "segments": []}
for s in segs:
    res["segments"].append({"start": s.start, "end": s.end, "text": s.text,
        "words": [{"w": w.word, "s": w.start, "e": w.end, "p": w.probability} for w in s.words]})
    print(f"{s.start:6.2f}-{s.end:6.2f} {s.text}", flush=True)
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
print("DONE", info.language, info.language_probability, flush=True)
