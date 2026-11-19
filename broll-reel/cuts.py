import json, sys, numpy as np, soundfile as sf
from faster_whisper.vad import get_speech_timestamps, VadOptions
a, sr = sf.read(sys.argv[1], dtype="float32")
opts = VadOptions(threshold=float(sys.argv[4]), min_silence_duration_ms=200, speech_pad_ms=60, min_speech_duration_ms=120)
ts = get_speech_timestamps(a, opts)
sp = [[t["start"]/sr, t["end"]/sr] for t in ts]
# union with whisper words (protect speech VAD missed)
d = json.load(open(sys.argv[2]))
for s in d["segments"]:
    for w in s["words"]:
        sp.append([max(0, w["s"]-0.03), w["e"]+0.05])
sp.sort(); keep = []
for s, e in sp:
    if keep and s - keep[-1][1] < 0.2: keep[-1][1] = max(keep[-1][1], e)
    else: keep.append([s, e])
dur = len(a)/sr
keep = [[round(max(0,s),3), round(min(dur,e),3)] for s, e in keep if e - s > 0.15]
json.dump(keep, open(sys.argv[3], "w"))
tot = sum(e-s for s, e in keep)
print(len(keep), "segments, kept", round(tot,2), "of", round(dur,2))
for i in range(1, len(keep)): print(f"cut {keep[i-1][1]:.2f}-{keep[i][0]:.2f} ({keep[i][0]-keep[i-1][1]:.2f})")
