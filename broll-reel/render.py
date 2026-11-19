"""B-roll reel renderer.
usage: python3 -I render.py <work_dir> <out.mp4> [--preview SECONDS_START SECONDS_END]
All edit times are voice-track seconds (see edl.py)."""
import sys, os, json, math, subprocess, functools
import numpy as np, cv2, soundfile as sf
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edl import SENT, RAW_RANGES, ZOOM_SENT, BROLL

W, H, FPS = 1080, 1920, 30
SAFE = (96, 232, 984, 1393)
WORK = sys.argv[1]; OUT = sys.argv[2]
SCR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS = os.path.join(SCR, "fonts")
RED = (227, 38, 46); BLACK = (17, 17, 17); WHITE = (255, 255, 255)
GOLD = (255, 200, 61); GREY = (150, 150, 150); LIGHT = (241, 241, 241); WATER = (214, 230, 240)
SUB_DELAY = 0.10

@functools.lru_cache(None)
def inter(size, weight=800):
    f = ImageFont.truetype(os.path.join(FONTS, "Inter.ttf"), size)
    f.set_variation_by_axes([min(32, max(14, size / 3)), weight]); return f
@functools.lru_cache(None)
def play(size, weight=500):
    f = ImageFont.truetype(os.path.join(FONTS, "PlayfairItalic.ttf"), size)
    f.set_variation_by_axes([weight]); return f

# ---------------------------------------------------------------- timeline
keep = json.load(open(os.path.join(WORK, "keep.json")))
segs = []; o = 0
for s, e in keep:
    fs, fe = round(s * FPS), round(e * FPS)
    if fe - fs < 3: continue
    segs.append((fs, fe, o)); o += fe - fs          # src frame start/end, out frame start
NOUT = o
def out_t(src_t):
    f = src_t * FPS
    for fs, fe, of in segs:
        if f < fs: return of / FPS                   # in a cut gap -> next segment start
        if f <= fe: return (of + f - fs) / FPS
    return NOUT / FPS
def src_frame(out_f):
    for fs, fe, of in segs:
        if out_f < of + (fe - fs): return fs + out_f - of
    return segs[-1][1] - 1

# ---------------------------------------------------------------- word timing
def load_words(p):
    d = json.load(open(os.path.join(WORK, p)))
    return [(w["w"].strip(), w["s"], w["e"]) for s in d["segments"] for w in s["words"] if w["w"].strip()]
WC, WR = load_words("clean_whisper.json"), load_words("cam_whisper.json")
def whisper_words(a, b):
    raw = any(lo <= (a + b) / 2 <= hi for lo, hi in RAW_RANGES)
    src = WR if raw else WC
    ws = [w for w in src if w[1] >= a - 0.15 and w[1] < b - 0.05]
    return ws or [("x", a, b)]
SENTS = []
for sid, a, b, text, kws in SENT:
    words = text.split()
    ww = whisper_words(a, b)
    # cumulative char fraction -> time, piecewise linear over whisper word starts
    wl = np.array([len(w[0]) + 1 for w in ww], float); wc = np.concatenate([[0], np.cumsum(wl)]) / wl.sum()
    wt = np.array([w[1] for w in ww] + [ww[-1][2]])
    cl = np.array([len(w) + 1 for w in words], float); cc = np.concatenate([[0], np.cumsum(cl)])[:-1] / cl.sum()
    ts = np.interp(cc, wc, wt); ts[0] = max(ts[0], ww[0][1])
    kwset = {k.lower() for k in kws}
    items = []
    for w, t in zip(words, ts):
        core = w.strip(".,:;?!—").lower()
        items.append(dict(w=w, src=float(t), t=out_t(float(t)) + SUB_DELAY, kw=core in kwset))
    SENTS.append(dict(id=sid, a=a, b=b, oa=out_t(a), ob=out_t(b), words=items))
SBY = {s["id"]: s for s in SENTS}
def sent_start_out(sid): return SBY[sid]["oa"]
def next_start_out(sid):
    nx = [s for s in SENTS if s["id"] > sid]
    return nx[0]["oa"] if nx else NOUT / FPS
def kw_time(sid, word):
    for it in SBY[sid]["words"]:
        if it["w"].strip(".,:;?!—").lower() == word.lower(): return it["t"]
    raise KeyError((sid, word))

# subtitle chunks
CHUNKS = []
for s in SENTS:
    cur = []
    for i, it in enumerate(s["words"]):
        cur.append(it)
        txt = " ".join(x["w"] for x in cur)
        end = it["w"][-1] in ".,:;?!" or len(cur) >= 3 or len(txt) >= 15
        if end or i == len(s["words"]) - 1:
            CHUNKS.append(cur); cur = []
for i, c in enumerate(CHUNKS):
    c0 = c[0]["t"]; nxt = CHUNKS[i + 1][0]["t"] if i + 1 < len(CHUNKS) else 1e9
    sid = next(s["id"] for s in SENTS if c[0] in s["words"])
    CHUNKS[i] = dict(words=c, t0=c0, t1=min(nxt, max(c[-1]["t"] + 0.5, SBY[sid]["ob"] + 0.25)))

# ---------------------------------------------------------------- text helpers
def text_layer():
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))
def draw_sub(img, out_tt):
    ch = [c for c in CHUNKS if c["t0"] <= out_tt < c["t1"]]
    if not ch: return
    c = ch[-1]; d = ImageDraw.Draw(img)
    base, kwsz = 74, 96
    def fnt(it, k=1.0): return inter(int((kwsz if it["kw"] else base) * k), 900 if it["kw"] else 800)
    k = 1.0
    while True:   # layout into lines that fit the safe width
        lines, line, lw = [], [], 0
        sp = fnt({"kw": False}, k).getlength(" ")
        for it in c["words"]:
            wl = fnt(it, k).getlength(it["w"])
            if line and lw + sp + wl > SAFE[2] - SAFE[0] - 20: lines.append(line); line, lw = [], 0
            line.append((it, wl)); lw += (sp if len(line) > 1 else 0) + wl
        lines.append(line)
        if len(lines) <= 2 or k < 0.6: break
        k -= 0.08
    lh = int(kwsz * k * 1.12); ybase = 1300 - lh * len(lines)   # last line bottom ~1300 (< 1393)
    for li, line in enumerate(lines):
        tot = sum(wl for _, wl in line) + sp * (len(line) - 1)
        x = (W - tot) / 2; yb = ybase + (li + 1) * lh
        for it, wl in line:
            if it["t"] <= out_tt:
                f = fnt(it, k)
                d.text((x, yb), it["w"], font=f, fill=GOLD if it["kw"] else WHITE, anchor="ls",
                       stroke_width=7, stroke_fill=(0, 0, 0))
            x += wl + sp

# ---------------------------------------------------------------- hook
HOOK_END = next_start_out(1)
_seg = None
def person_mask(rgb):
    global _seg
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    if _seg is None:
        _seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(SCR, "models", "selfie_multiclass.tflite")),
            output_confidence_masks=True))
    r = _seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    m = 1.0 - r.confidence_masks[0].numpy_view().astype(np.float32)
    m = cv2.resize(m, (W, H)); m = cv2.GaussianBlur(m, (0, 0), 2.0)    # soft matte edge only
    return np.clip((m - 0.35) / 0.3, 0, 1)
@functools.lru_cache(None)
def hook_layers():
    back = text_layer(); d = ImageDraw.Draw(back)
    f = inter(330, 900); word = "СУДЫ"
    while f.getlength(word) > SAFE[2] - SAFE[0]: f = inter(f.size - 6, 900)
    d.text((W / 2, 238), word, font=f, fill=RED, anchor="mt")
    front = text_layer(); d = ImageDraw.Draw(front)
    d.text((W / 2, 1075), "дұрыс ішесіз бе?", font=play(104, 600), fill=WHITE, anchor="ms",
           stroke_width=3, stroke_fill=(20, 20, 30))
    return np.asarray(back, np.float32) / 255, front

# ---------------------------------------------------------------- B-roll animation scenes
def ease(x): x = min(1, max(0, x)); return 1 - (1 - x) ** 3
def ui_frame(idx, label):
    img = np.full((H, W, 3), 255, np.uint8)
    pil = Image.fromarray(img); d = ImageDraw.Draw(pil)
    f = inter(34, 800); tw = f.getlength(label)
    d.rounded_rectangle((SAFE[0], 250, SAFE[0] + tw + 48, 306), radius=28, fill=RED)
    d.text((SAFE[0] + 24, 278), label, font=f, fill=WHITE, anchor="lm")
    d.text((SAFE[2], 278), f"0{idx}/05", font=inter(30, 600), fill=GREY, anchor="rm")
    d.line((SAFE[0], 340, SAFE[2], 340), fill=(225, 225, 225), width=3)
    return pil
def bottle(a, x, y, w, h, color=BLACK, water=None, th=8):
    cv2.rectangle(a, (x + w // 2 - 55, y - 70), (x + w // 2 + 55, y), color, th, cv2.LINE_AA)
    if water is not None:
        lvl = int(y + h - (h - 30) * water)
        cv2.rectangle(a, (x + 10, lvl), (x + w - 10, y + h - 10), WATER, -1, cv2.LINE_AA)
    rr(a, x, y, w, h, 40, color, th)
def rr(a, x, y, w, h, r, color, th):
    for cx, cy, ang in ((x + r, y + r, 180), (x + w - r, y + r, 270), (x + w - r, y + h - r, 0), (x + r, y + h - r, 90)):
        cv2.ellipse(a, (cx, cy), (r, r), ang, 0, 90, color, th, cv2.LINE_AA)
    for p, q in (((x + r, y), (x + w - r, y)), ((x + r, y + h), (x + w - r, y + h)), ((x, y + r), (x, y + h - r)), ((x + w, y + r), (x + w, y + h - r))):
        cv2.line(a, p, q, color, th, cv2.LINE_AA)
def arr(pil): return np.array(pil)

def scene(name, t):
    if name == "filter":
        t_f, t_s = kw_time(6, "фильтрі"), kw_time(7, "сілтілі")
        pil = ui_frame(1, "ФИЛЬТР"); a = arr(pil)
        bottle(a, 170, 470, 300, 640, water=0.85)
        if t >= t_f:
            cv2.rectangle(a, (255, 560), (385, 1050), RED, 8, cv2.LINE_AA)
            for yy in range(600, 1020, 46):
                for xx in (290, 350):
                    cv2.circle(a, (xx, yy), 13, RED, -1, cv2.LINE_AA)
            cv2.line(a, (395, 700), (560, 700), BLACK, 4, cv2.LINE_AA)
        pil = Image.fromarray(a); d = ImageDraw.Draw(pil)
        if t >= t_f:
            d.text((580, 700), "фильтр", font=play(104, 600), fill=BLACK, anchor="lm")
        if t >= t_s:
            d.text((580, 900), "сілтілі", font=play(104, 600), fill=RED, anchor="lm")
            d.text((580, 1010), "СУ", font=inter(96, 900), fill=RED, anchor="lm")
        return arr(pil)
    if name == "ph":
        t_z, t_a = kw_time(10, "закисленіп"), kw_time(11, "аурулар")
        pil = ui_frame(2, "pH"); d = ImageDraw.Draw(pil)
        x0, x1, y0 = 140, 940, 700
        for i in range(x1 - x0):
            u = i / (x1 - x0)
            col = tuple(int(RED[c] * (1 - u) + LIGHT[c] * u) if u < 0.5 else int(LIGHT[c] * (2 - 2 * u) + 120 * (2 * u - 1)) for c in range(3))
            d.line((x0 + i, y0, x0 + i, y0 + 70), fill=col)
        for v in (0, 7, 14):
            d.text((x0 + (x1 - x0) * v / 14, y0 + 110), str(v), font=inter(40, 700), fill=BLACK, anchor="mm")
        v = 7.4 + (4.0 - 7.4) * ease((t - t_z) / 0.8) if t >= t_z else 7.4
        mx = x0 + (x1 - x0) * v / 14
        d.polygon([(mx - 26, y0 - 50), (mx + 26, y0 - 50), (mx, y0 - 8)], fill=BLACK)
        if t >= t_z:
            d.text((W / 2, 520), "закисленген", font=play(116, 600), fill=RED, anchor="mm")
        if t >= t_a:
            d.text((W / 2, 980), "АУРУЛАР ↑", font=inter(120, 900), fill=RED, anchor="mm")
        return arr(pil)
    if name == "cell":
        t_c, t_q = kw_time(13, "клеткаға"), kw_time(14, "қанға")
        pil = ui_frame(3, "КЛЕТКА"); a = arr(pil)
        cells = [(260, 620), (540, 580), (820, 620), (400, 820), (680, 820)]
        if t >= t_c:
            for i, (cx, cy) in enumerate(cells):
                cv2.circle(a, (cx, cy), 95, BLACK, 6, cv2.LINE_AA)
                cv2.circle(a, (cx + 18, cy - 10), 26, GREY, -1, cv2.LINE_AA)
                k = ease((t - t_c - 0.15 * i) / 1.0)
                if k > 0:
                    dy = int(cy - 200 + 190 * k)
                    cv2.circle(a, (cx - 30, dy), 18, (90, 160, 210), -1, cv2.LINE_AA)
        if t >= t_q:
            for i in range(5):
                cx = 230 + i * 155
                cv2.circle(a, (cx, 1030), 58, RED, -1, cv2.LINE_AA)
                cv2.circle(a, (cx, 1030), 24, (240, 120, 125), -1, cv2.LINE_AA)
        pil = Image.fromarray(a); d = ImageDraw.Draw(pil)
        if t >= t_c: d.text((SAFE[2], 430), "клетка", font=play(84, 600), fill=BLACK, anchor="rm")
        if t >= t_q: d.text((SAFE[2], 1140), "қан", font=play(96, 600), fill=RED, anchor="rm")
        return arr(pil)
    if name == "water70":
        t_7 = kw_time(18, "70")
        pil = ui_frame(4, "АҒЗА"); a = arr(pil)
        cx = 250; head = (cx, 470); body = (cx - 140, 590, 280, 520)
        lvl = 0.7 * ease((t - t_7) / 1.2) if t >= t_7 else 0.0
        top, bot = 380, 1110
        fill_y = int(bot - (bot - top) * lvl)
        mask = np.zeros((H, W), np.uint8)
        cv2.circle(mask, head, 90, 255, -1, cv2.LINE_AA)
        cv2.rectangle(mask, (body[0], body[1] + 70), (body[0] + body[2], body[1] + body[3] - 70), 255, -1)
        for (ex, ey) in ((body[0] + 70, body[1] + 70), (body[0] + body[2] - 70, body[1] + 70), (body[0] + 70, body[1] + body[3] - 70), (body[0] + body[2] - 70, body[1] + body[3] - 70)):
            cv2.circle(mask, (ex, ey), 70, 255, -1, cv2.LINE_AA)
        cv2.rectangle(mask, (body[0] + 70, body[1]), (body[0] + body[2] - 70, body[1] + body[3]), 255, -1)
        region = np.zeros((H, W), bool); region[fill_y:bot + 1] = True
        m = (mask > 127)
        a[m] = LIGHT; a[m & region] = WATER
        cnt, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(a, cnt, -1, BLACK, 6, cv2.LINE_AA)
        pil = Image.fromarray(a); d = ImageDraw.Draw(pil)
        if t >= t_7:
            d.text((SAFE[2], 700), "70%", font=inter(200, 900), fill=RED, anchor="rm")
            d.text((SAFE[2], 900), "су", font=play(130, 600), fill=BLACK, anchor="rm")
        return arr(pil)
    if name == "plastic":
        t_p, t_16, t_b, t_u = kw_time(23, "пластик"), kw_time(24, "16"), kw_time(24, "бисфенол"), kw_time(25, "уларды")
        pil = ui_frame(5, "ПЛАСТИК"); a = arr(pil)
        if t >= t_p:
            bottle(a, 150, 560, 260, 480, color=GREY, water=0.8, th=7)
            for yy in (680, 800, 920):
                cv2.line(a, (150, yy), (410, yy), GREY, 4, cv2.LINE_AA)
        if t >= t_16:
            tx = 560
            rr(a, tx - 28, 470, 56, 380, 28, BLACK, 6)
            cv2.circle(a, (tx, 880), 56, RED, -1, cv2.LINE_AA)
            cv2.rectangle(a, (tx - 12, 640), (tx + 12, 860), RED, -1)
        pil = Image.fromarray(a); d = ImageDraw.Draw(pil)
        if t >= t_p: d.text((SAFE[0], 420), "пластик", font=play(96, 600), fill=BLACK, anchor="ls")
        if t >= t_16: d.text((650, 620), "16°C", font=inter(120, 900), fill=BLACK, anchor="lm")
        if t >= t_b:
            d.rounded_rectangle((650, 740, 970, 860), radius=22, fill=RED)
            d.text((810, 800), "BPA", font=inter(84, 900), fill=WHITE, anchor="mm")
            d.text((650, 905), "бисфенол А", font=play(60, 600), fill=RED, anchor="lm")
        if t >= t_u: d.text((810, 1050), "УЛАР", font=inter(130, 900), fill=RED, anchor="mm")
        return arr(pil)
    raise ValueError(name)

SCENES = [(n, sent_start_out(a), next_start_out(b)) for n, a, b in BROLL]
def scene_at(t):
    for n, a, b in SCENES:
        if a <= t < b: return n
    return None
def zoom_at(src_t):
    cur = [s for s in SENT if s[1] <= src_t + 1e-6]
    return bool(cur) and cur[-1][0] in ZOOM_SENT

# ---------------------------------------------------------------- render video
def main():
    pv = "--preview" in sys.argv
    f0, f1 = 0, NOUT
    if pv:
        i = sys.argv.index("--preview"); f0, f1 = int(float(sys.argv[i + 1]) * FPS), int(float(sys.argv[i + 2]) * FPS)
    grade = "eq=contrast=1.05:saturation=1.07:gamma=0.98,colorbalance=rs=0.01:bs=-0.01"
    first_src = src_frame(f0)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{max(0, first_src / FPS - 0.0):.4f}", "-i", os.path.join(SCR, "dl", "camera.mov"),
                            "-vf", grade, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    cur_src = first_src - 1; frame = None
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", OUT + ".video.mp4"], stdin=subprocess.PIPE)
    hb, hf = hook_layers()
    for of in range(f0, f1):
        t = of / FPS; sf_ = src_frame(of)
        while cur_src < sf_:
            buf = dec.stdout.read(W * H * 3)
            if len(buf) < W * H * 3: break
            frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3); cur_src += 1
        sc = scene_at(t)
        if sc:
            img = scene(sc, t)
        else:
            img = frame
            if zoom_at(sf_ / FPS):
                cw, chh = int(W / 1.2), int(H / 1.2); x0 = (W - cw) // 2; y0 = 120
                img = cv2.resize(img[y0:y0 + chh, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LANCZOS4)
            if t < HOOK_END:
                m = person_mask(img)[..., None]
                al = hb[..., 3:4] * (1 - m)
                img = (img * (1 - al) + hb[..., :3] * 255 * al).astype(np.uint8)
        pil = Image.fromarray(np.ascontiguousarray(img)).convert("RGBA")
        if not sc and t < HOOK_END: pil.alpha_composite(hf)
        lay = text_layer(); draw_sub(lay, t); pil.alpha_composite(lay)
        enc.stdin.write(np.asarray(pil.convert("RGB")).tobytes())
        if of % 150 == 0: print(f"frame {of}/{f1}", flush=True)
    enc.stdin.close(); enc.wait(); dec.kill()
    build_audio(f0, f1)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", OUT + ".video.mp4", "-i", OUT + ".audio.wav", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", OUT], check=True)
    os.remove(OUT + ".video.mp4")
    json.dump(dict(segments=segs, scenes=SCENES, chunks=[dict(t0=c["t0"], t1=c["t1"], text=" ".join(w["w"] for w in c["words"]),
               words=[(w["w"], round(w["src"], 3), round(w["t"], 3)) for w in c["words"]]) for c in CHUNKS]),
              open(OUT + ".timeline.json", "w"), ensure_ascii=False, indent=1)
    print("DONE", OUT)

# ---------------------------------------------------------------- audio
def build_audio(f0, f1):
    v, sr = sf.read(os.path.join(WORK, "clean48.wav"), dtype="float32")
    spf = sr // FPS; out = np.zeros(NOUT * spf, np.float32); fd = int(0.006 * sr)
    ramp = np.linspace(0, 1, fd, dtype=np.float32)
    for fs, fe, of in segs:
        piece = v[fs * spf: fe * spf].copy()
        piece[:fd] *= ramp; piece[-fd:] *= ramp[::-1]
        out[of * spf: of * spf + len(piece)] = piece
    n = len(out); t = np.arange(n) / sr
    # soft synth pad (Am - F - C - G), slow attack, very low level
    chords = [[220.0, 261.63, 329.63], [174.61, 220.0, 261.63], [196.0, 261.63, 329.63], [196.0, 246.94, 293.66]]
    mus = np.zeros(n, np.float32); bar = 4.0
    for k in range(int(t[-1] / bar) + 1):
        a0, a1 = int(k * bar * sr), min(n, int((k + 1) * bar * sr) + int(0.6 * sr))
        if a0 >= n: break
        tt = np.arange(a1 - a0) / sr
        env = np.minimum(1, tt / 0.9) * np.minimum(1, np.maximum(0, (bar + 0.6 - tt) / 0.9))
        for fq in chords[k % 4]:
            mus[a0:a1] += (np.sin(2 * np.pi * fq * tt) + 0.25 * np.sin(2 * np.pi * fq * 2 * tt)) * env
    mus = np.convolve(mus, np.ones(32) / 32, mode="same")
    mus *= 10 ** (-30 / 20) / (np.abs(mus).max() + 1e-9)
    fade = int(1.5 * sr); mus[:fade] *= np.linspace(0, 1, fade); mus[-fade:] *= np.linspace(1, 0, fade)
    # soft SFX: gentle tick on each B-roll cut-in and each animated element
    sfx = np.zeros(n, np.float32)
    def tick(at, vol=-34):
        i = int(at * sr); L = int(0.07 * sr)
        if i + L >= n: return
        tt = np.arange(L) / sr
        sfx[i:i + L] += (np.sin(2 * np.pi * 1400 * tt) * np.exp(-tt * 60)) * 10 ** (vol / 20)
    for name, a, b in SCENES:
        tick(a)
    for sid, w in ((6, "фильтрі"), (7, "сілтілі"), (10, "закисленіп"), (11, "аурулар"), (13, "клеткаға"), (14, "қанға"),
                   (18, "70"), (23, "пластик"), (24, "16"), (24, "бисфенол"), (25, "уларды")):
        tick(kw_time(sid, w) - SUB_DELAY, -38)
    vpk = np.abs(out).max() + 1e-9
    mix = out / vpk * 0.7 + mus + sfx
    sf.write(OUT + ".mix.wav", mix[f0 * spf: f1 * spf], sr)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", OUT + ".mix.wav", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-ar", "48000", OUT + ".audio.wav"], check=True)
    os.remove(OUT + ".mix.wav")

if __name__ == "__main__":
    main()
