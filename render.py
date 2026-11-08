#!/usr/bin/env python3
"""Predajné reels video pre Michaelo Cavalli (limewash – matná dekoratívna vápenná farba) podľa style.md.

Použitie:
  pip install edge-tts   # dabing (neurálny slovenský hlas), cachuje sa v assets/vo/
  python3 render.py --src zdroj.mov --out michaelo_cavalli_reel.mp4 --handle @tvoj.handle

Celé video sa renderuje snímka po snímke (PIL + numpy) a posiela do ffmpeg.
Animované zábery (limewash textúra, štetec, kroky, kalkulačka, outro) sú
generované procedurálne, hudba je syntetizovaná v numpy.
"""
import argparse, math, os, subprocess, sys, tempfile, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1080, 1920, 30
SRC_FPS = 30

# ------------------------------------------------------------------ fonts
FD = "/usr/share/fonts"
F_SANS_XB = f"{FD}/opentype/inter/Inter-ExtraBold.otf"
F_SANS_B = f"{FD}/opentype/inter/Inter-Bold.otf"
F_SANS_M = f"{FD}/opentype/inter/Inter-Medium.otf"
F_SERIF = f"{FD}/truetype/crosextra/Caladea-Regular.ttf"
F_SERIF_I = f"{FD}/truetype/crosextra/Caladea-Italic.ttf"
_fc = {}
def font(path, size):
    k = (path, int(size))
    if k not in _fc:
        _fc[k] = ImageFont.truetype(path, int(size))
    return _fc[k]

# ------------------------------------------------------------------ palette (tmavá + teplé zlato/béžová)
GOLD = (212, 175, 106)
GOLD_L = (236, 210, 150)
CREAM = (246, 238, 224)
DARK = (18, 15, 13)
DARK2 = (34, 29, 25)

def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)

def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3

# ------------------------------------------------------------------ source footage
class Source:
    def __init__(self, raw_path):
        n = os.path.getsize(raw_path) // (W * H * 3)
        self.frames = np.memmap(raw_path, dtype=np.uint8, mode="r", shape=(n, H, W, 3))
        self.n = n
        # teplý filmový grade: kontrast + teplý tón
        x = np.arange(256) / 255.0
        s = 0.5 + (x - 0.5) * 1.18
        s = np.clip(s, 0, 1)
        s = s ** 1.08
        r = np.clip(s * 1.06 + 0.01, 0, 1)
        g = np.clip(s * 1.00, 0, 1)
        b = np.clip(s * 0.88, 0, 1)
        self.lut = (np.stack([r, g, b], 1) * 255).astype(np.uint8)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = ((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.75)) ** 2
        self.vig = np.clip(1.0 - 0.55 * d, 0.45, 1.0)[..., None].astype(np.float32)

    def get(self, t, zoom=1.0, cx=0.5, cy=0.5):
        i = int(round(t * SRC_FPS))
        i = min(max(i, 0), self.n - 1)
        a = np.asarray(self.frames[i])
        g = np.empty_like(a)
        for c in range(3):
            g[..., c] = self.lut[a[..., c], c]
        g = (g.astype(np.float32) * self.vig).astype(np.uint8)
        im = Image.fromarray(g)
        if zoom > 1.0001:
            cw, ch = W / zoom, H / zoom
            x0 = min(max(cx * W - cw / 2, 0), W - cw)
            y0 = min(max(cy * H - ch / 2, 0), H - ch)
            im = im.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))
        return im

# ------------------------------------------------------------------ procedural plaster texture
def fractal_noise(w, h, seed, octaves=6, base=4):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gw, gh = base * 2 ** o, int(base * 2 ** o * h / w) + 1
        g = rng.random((gh + 1, gw + 1)).astype(np.float32)
        im = Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        out += np.asarray(im, np.float32) / 255.0 * amp
        tot += amp
        amp *= 0.55
    return out / tot

def make_plaster(w, h, seed=7, tint=(1.0, 0.94, 0.85), dark=0.50, light=0.92):
    n1 = fractal_noise(w, h, seed, 7, 3)
    n2 = fractal_noise(w, h, seed + 1, 6, 5)
    n3 = fractal_noise(w, h, seed + 2, 5, 9)
    # oblakové škvrny a mäkké ťahy štetcom (matný limewash)
    ridge = 1.0 - np.abs(n2 - 0.5) * 2.0
    v = 0.72 * n1 + 0.13 * ridge ** 2 + 0.15 * n3
    v = (v - v.min()) / (v.max() - v.min())
    v = np.clip((v - 0.2) / 0.65, 0, 1) ** 1.1
    v = dark + (light - dark) * v
    rgb = np.stack([v * tint[0], v * tint[1], v * tint[2]], -1)
    # jemné zlaté žilky
    vein = np.zeros_like(ridge)[..., None]
    rgb = rgb * (1 - 0.35 * vein) + np.array([0.85, 0.70, 0.42]) * 0.35 * vein
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))

# ------------------------------------------------------------------ drawing helpers
def text_size(d, txt, f):
    b = d.textbbox((0, 0), txt, font=f)
    return b[2] - b[0], b[3] - b[1], b

def draw_center_word(im, word, t_local, y=1230, size=150, color=(255, 255, 255), accent=None):
    """Jednoslovný titulok (variant A) s pop-in animáciou."""
    s = 1.0 + 0.22 * (1 - ease_out(t_local / 0.14))
    f = font(F_SANS_XB, size * s)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    tw, th, b = text_size(d, word, f)
    if tw > W - 140:  # prispôsobiť šírke bezpečnej zóny
        f = font(F_SANS_XB, size * s * (W - 140) / tw)
        tw, th, b = text_size(d, word, f)
    x = (W - tw) / 2 - b[0]
    yy = y - th / 2 - b[1]
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).text((x, yy + 8), word, font=f, fill=(0, 0, 0, 170))
    sh = sh.filter(ImageFilter.GaussianBlur(14))
    lay = Image.alpha_composite(sh, lay)
    d = ImageDraw.Draw(lay)
    d.text((x, yy), word, font=f, fill=accent or color, stroke_width=3, stroke_fill=(0, 0, 0, 90))
    im.paste(lay, (0, 0), lay)

def ig_icon(size, colored=False):
    """Logo Instagramu (zaoblený štvorec, kruh, bodka)."""
    S = size * 4
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    lw = int(S * 0.085)
    d.rounded_rectangle((lw / 2, lw / 2, S - lw / 2, S - lw / 2), radius=S * 0.28, outline=255, width=lw)
    r = S * 0.215
    d.ellipse((S / 2 - r, S / 2 - r, S / 2 + r, S / 2 + r), outline=255, width=lw)
    rd = S * 0.055
    d.ellipse((S * 0.76 - rd, S * 0.24 - rd, S * 0.76 + rd, S * 0.24 + rd), fill=255)
    m = m.resize((size, size), Image.LANCZOS)
    if colored:
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
        tt = np.clip((xx * 0.5 + (1 - yy) * 0.5), 0, 1)[..., None]
        c1 = np.array([254, 218, 117.0]); c2 = np.array([250, 126, 30.0])
        c3 = np.array([214, 41, 118.0]); c4 = np.array([150, 47, 191.0])
        col = np.where(tt < 0.33, c1 + (c2 - c1) * (tt / 0.33),
              np.where(tt < 0.66, c2 + (c3 - c2) * ((tt - 0.33) / 0.33), c3 + (c4 - c3) * ((tt - 0.66) / 0.34)))
        # gradient je zdola zľava (žltá) do pravého horného rohu (fialová)
        tt2 = np.clip(((1 - xx) * 0.5 + yy * 0.5), 0, 1)[..., None]
        col = np.where(tt2 > 0.66, c1 + (c2 - c1) * ((1 - tt2) / 0.34),
              np.where(tt2 > 0.33, c2 + (c3 - c2) * ((0.66 - tt2) / 0.33), c3 + (c4 - c3) * ((0.33 - tt2) / 0.33)))
        rgb = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8))
    else:
        rgb = Image.new("RGB", (size, size), (255, 255, 255))
    out = rgb.convert("RGBA")
    out.putalpha(m)
    return out

def watermark(im, handle):
    ic = ig_icon(44)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    x, y = 60, 150
    a = 200
    ic2 = ic.copy(); ic2.putalpha(Image.fromarray((np.asarray(ic.getchannel("A"), np.float32) * a / 255).astype(np.uint8)))
    lay.paste(ic2, (x, y), ic2)
    f = font(F_SANS_B, 32)
    d.text((x + 58, y + 4), handle, font=f, fill=(255, 255, 255, a), stroke_width=2, stroke_fill=(0, 0, 0, 60))
    im.paste(lay, (0, 0), lay)

def rounded_paste(base, im, box, radius=46, shadow=True, border=None):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    im = im.resize((w, h), Image.BILINEAR) if im.size != (w, h) else im
    m = Image.new("L", (w * 2, h * 2), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * 2 - 1, h * 2 - 1), radius=radius * 2, fill=255)
    m = m.resize((w, h), Image.LANCZOS)
    if shadow:
        sh = Image.new("RGBA", base.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle((x0, y0 + 16, x1, y1 + 16), radius=radius, fill=(0, 0, 0, 150))
        sh = sh.filter(ImageFilter.GaussianBlur(22))
        base.paste(sh, (0, 0), sh)
    base.paste(im.convert("RGB"), (x0, y0), m)
    if border:
        lay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        ImageDraw.Draw(lay).rounded_rectangle(box, radius=radius, outline=border, width=3)
        base.paste(lay, (0, 0), lay)

# ------------------------------------------------------------------ animated scenes
class Assets:
    def __init__(self):
        print("generujem textúry…", file=sys.stderr)
        self.plaster = make_plaster(1400, 2300, seed=11)
        self.plaster2 = make_plaster(1400, 1400, seed=23, tint=(1.0, 0.95, 0.88))
        self.dark_tex = Image.fromarray((np.asarray(make_plaster(W, H, seed=5), np.float32) * 0.28).astype(np.uint8))
        # ťahy štetcom do kríža (X) pre krok 2
        rng = np.random.default_rng(3)
        self.strokes = []
        for i in range(40):
            cx, cy = rng.uniform(0, 940), rng.uniform(0, 760)
            ang = math.radians(rng.choice([45, -45]) + rng.uniform(-12, 12))
            L = rng.uniform(260, 460)
            self.strokes.append((cx, cy, ang, L, rng.uniform(120, 180)))
        self.ig_w = ig_icon(300, False)
        self.ig_c = ig_icon(300, True)

def scene_plaster_pan(A, t, dur, x0=0.1, y0=0.15, x1=0.35, y1=0.35, z0=1.0, z1=1.12, tex=None):
    tex = tex or A.plaster
    tw, th = tex.size
    p = t / dur
    z = z0 + (z1 - z0) * p
    cw, ch = tw / 1.3 / z, tw / 1.3 / z * H / W
    ch = min(ch, th)
    cx = (x0 + (x1 - x0) * p) * (tw - cw)
    cy = (y0 + (y1 - y0) * p) * (th - ch)
    return tex.resize((W, H), Image.BILINEAR, box=(cx, cy, cx + cw, cy + ch))

def anim_brush(A, t, dur, w, h):
    flat = Image.new("RGB", (w, h), (206, 200, 190))
    tex = A.plaster.resize((w, h), Image.BILINEAR, box=(100, 300, 100 + 1100, 300 + 1100 * h / w))
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    n = len(A.strokes)
    prog = min(t / (dur * 0.92), 1.0) * n
    last = None
    sx, sy = w / 940, h / 760
    for i, (cx, cy, ang, L, width) in enumerate(A.strokes):
        if i > prog:
            break
        cx, cy, L, width = cx * sx, cy * sy, L * sx, width * sx
        frac = min(prog - i, 1.0)
        dx, dy = math.cos(ang) * L / 2, math.sin(ang) * L / 2
        x0, y0 = cx - dx, cy - dy
        x1, y1 = x0 + 2 * dx * frac, y0 + 2 * dy * frac
        d.line((x0, y0, x1, y1), fill=200, width=int(width))
        last = (x1, y1, ang)
    m = m.filter(ImageFilter.GaussianBlur(14 * sx))
    im = Image.composite(tex, flat, m)
    if last is not None and t < dur * 0.95:
        x, y, ang = last
        lay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        dd = ImageDraw.Draw(lay)
        # široký štetec na limewash (štetiny + drevené telo + rukoväť)
        pa = ang + math.pi / 2
        ca, sa = math.cos(pa), math.sin(pa)
        k = sx
        def R(px, py):
            px, py = px * k, py * k
            return (x + px * ca - py * sa, y + px * sa + py * ca)
        dd.polygon([R(-95, -10), R(95, -10), R(85, 60), R(-85, 60)], fill=(232, 220, 196, 255))   # štetiny
        dd.polygon([R(-100, 60), R(100, 60), R(100, 110), R(-100, 110)], fill=(150, 104, 62, 255))  # telo
        dd.polygon([R(-18, 110), R(18, 110), R(14, 260), R(-14, 260)], fill=(120, 80, 46, 255))    # rukoväť
        im.paste(lay, (0, 0), lay)
    return im

def anim_calculator(A, t):
    im = A.dark_tex.copy()
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    a = ease_out(t / 0.4)
    x0, y0, x1, y1 = 110, int(330 + (1 - a) * 80), 970, int(1130 + (1 - a) * 80)
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((x0, y0 + 24, x1, y1 + 24), radius=44, fill=(0, 0, 0, 180))
    sh = sh.filter(ImageFilter.GaussianBlur(30))
    lay = Image.alpha_composite(lay, sh)
    d = ImageDraw.Draw(lay)
    d.rounded_rectangle((x0, y0, x1, y1), radius=44, fill=(250, 245, 236, 255))
    d.text((x0 + 60, y0 + 56), "Kalkulačka ceny", font=font(F_SANS_XB, 54), fill=DARK)
    d.text((x0 + 60, y0 + 126), "limewash na mieru", font=font(F_SERIF_I, 40), fill=(150, 112, 50))
    area = int(24 * ease(min((t - 0.5) / 1.2, 1)))
    rows = [("Plocha steny", f"{max(area,0)} m²"), ("Odtieň", "Piesková" if t > 1.6 else "—"),
            ("Povrch", "Matný" if t > 2.0 else "—")]
    for i, (k, v) in enumerate(rows):
        yy = y0 + 230 + i * 120
        d.rounded_rectangle((x0 + 50, yy, x1 - 50, yy + 92), radius=22, fill=(238, 230, 216, 255))
        d.text((x0 + 80, yy + 26), k, font=font(F_SANS_M, 36), fill=(110, 100, 90))
        f = font(F_SANS_B, 40)
        d.text((x1 - 80 - d.textlength(v, font=f), yy + 24), v, font=f, fill=DARK)
    pressed = 2.5 < t < 2.75
    by = y0 + 610
    d.rounded_rectangle((x0 + 50, by + (4 if pressed else 0), x1 - 50, by + 110), radius=30,
                        fill=(GOLD if not pressed else (180, 146, 84)) + (255,))
    lab = "Napíš mi – spočítam cenu" if t > 2.6 else "Spočítať cenu"
    f = font(F_SANS_XB, 42)
    d.text(((x0 + x1) / 2 - d.textlength(lab, font=f) / 2, by + 30), lab, font=f, fill=DARK)
    if t > 0.5:  # kurzor/prst
        px = x1 - 200 + 60 * math.sin(t * 2) if t < 2.4 else (x0 + x1) / 2 + 120
        py = y0 + 300 + 120 * min(t / 1.5, 2) if t < 2.4 else by + 70
        d.ellipse((px - 26, py - 26, px + 26, py + 26), fill=(255, 255, 255, 170), outline=(0, 0, 0, 120), width=3)
    im.paste(lay, (0, 0), lay)
    return im

# ------------------------------------------------------------------ dabing (voiceover)
VOICE = "sk-SK-LukasNeural"
VO_RATE = "+6%"
GAP = 0.18  # pauza medzi vetami (pauzy vystrihnuté, len nádych)

# (text pre hlas, kľúčové slová farebne, titulky – skupiny slov oddelené „|“, počet slov = počet slov v texte)
VO_LINES = [
    ("Toto nie je betón.", {"BETÓN"}, "TOTO|NIE JE|BETÓN"),
    ("Ani tapeta.", {"TAPETA"}, "ANI|TAPETA"),
    ("Je to lajmvoš. Prírodná vápenná farba.", {"LIMEWASH"}, "JE TO|LIMEWASH|PRÍRODNÁ|VÁPENNÁ|FARBA"),
    ("Matná, priedušná a každá stena je originál.", {"ORIGINÁL"}, "MATNÁ|PRIEDUŠNÁ|A KAŽDÁ|STENA|JE ORIGINÁL"),
    ("Nanášam ju ručne, ťah po ťahu.", {"RUČNE"}, "NANÁŠAM JU|RUČNE|ŤAH|PO ŤAHU"),
    ("Výsledok? Interiér ako z magazínu.", {"MAGAZÍNU"}, "VÝSLEDOK?|INTERIÉR|AKO Z|MAGAZÍNU"),
    ("Pre byty, domy aj prevádzky.", {"PREVÁDZKY"}, "PRE BYTY|DOMY|AJ PREVÁDZKY"),
    ("Napíš mi rozmery steny a spočítam ti cenu.", {"CENU"}, "NAPÍŠ MI|ROZMERY|STENY|A SPOČÍTAM|TI CENU"),
    ("Sleduj ma!", set(), "SLEDUJ MA"),
]

def gen_vo(vo_dir):
    """Vygeneruje dabing (edge-tts, neurálny slovenský hlas) + časy slov. Výsledok sa cachuje."""
    import json, asyncio
    os.makedirs(vo_dir, exist_ok=True)
    meta_p = os.path.join(vo_dir, "vo.json")
    if os.path.exists(meta_p):
        meta = json.load(open(meta_p))
        if [m["text"] for m in meta] == [l[0] for l in VO_LINES]:
            return meta
    import edge_tts
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")

    async def one(i, text):
        comm = edge_tts.Communicate(text, VOICE, rate=VO_RATE, boundary="WordBoundary", proxy=proxy)
        audio, words = b"", []
        async for ch in comm.stream():
            if ch["type"] == "audio":
                audio += ch["data"]
            elif ch["type"] == "WordBoundary":
                words.append((ch["offset"] / 1e7, (ch["offset"] + ch["duration"]) / 1e7, ch["text"]))
        mp3 = os.path.join(vo_dir, f"line{i}.mp3")
        open(mp3, "wb").write(audio)
        wav = os.path.join(vo_dir, f"line{i}.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp3, "-ar", "44100", "-ac", "1", wav], check=True)
        # oreže ticho na konci vety
        with wave.open(wav) as wf:
            n = wf.getnframes()
            x = np.frombuffer(wf.readframes(n), np.int16)
        nz = np.nonzero(np.abs(x) > 300)[0]
        start = max(nz[0] - 600, 0) if len(nz) else 0
        end = min(nz[-1] + 2500, n) if len(nz) else n
        x = x[start:end]
        with wave.open(wav, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(44100); wf.writeframes(x.tobytes())
        sh = start / 44100
        words = [(a - sh, b - sh, w) for a, b, w in words]
        return {"text": text, "wav": f"line{i}.wav", "dur": len(x) / 44100, "words": words}

    async def run():
        return [await one(i, t) for i, (t, _, _) in enumerate(VO_LINES)]
    meta = asyncio.run(run())
    json.dump(meta, open(meta_p, "w"), ensure_ascii=False, indent=1)
    return meta

def build_captions(meta, starts):
    """Titulky (1–2 slová) presne na hovorené slovo podľa časov z TTS."""
    caps = []
    for li, m in enumerate(meta):
        _, keys, groups = VO_LINES[li]
        words = m["words"]
        line_end = starts[li] + m["dur"]
        k, items = 0, []
        for g in groups.split("|"):
            n = len(g.split())
            items.append((starts[li] + words[min(k, len(words) - 1)][0], g))
            k += n
        assert k == len(words), (groups, [w[2] for w in words])
        for q, (a, g) in enumerate(items):
            end = items[q + 1][0] if q + 1 < len(items) else line_end
            accent = GOLD_L if any(x.strip("?") in keys for x in g.split()) else None
            caps.append((a, end, g, accent))
    return caps

# ------------------------------------------------------------------ timeline
def build_timeline(src, A, meta):
    """Vizuál po vetách: každá veta = 1–3 zábery (strih každých ~0,6–1,3 s)."""
    starts, t = [], 0.0
    for m in meta:
        starts.append(t)
        t += m["dur"] + GAP
    vo_end = t
    outro_len = 1.8
    total = vo_end + outro_len

    S = src.get
    P = lambda *a, **k: (lambda t, d: scene_plaster_pan(A, t, d, *a, **k))
    calc = lambda t, d: anim_calculator(A, 0.3 + t * 3.0 / d)
    brush = lambda t, d: anim_brush(A, t + 1.0, d + 1.4, W, H)
    visuals = [
        # 0 „Toto nie je betón.“ – hook: prudké punch-iny
        [lambda t, d: S(0.3 + t, 1.7 - 0.3 * t / d, 0.22, 0.30),
         lambda t, d: S(2.8 + t, 1.25, 0.55, 0.5)],
        # 1 „Ani tapeta.“
        [P(0.6, 0.2, 0.4, 0.3, 1.3, 1.5), lambda t, d: S(4.6 + t, 1.4, 0.4, 0.3)],
        # 2 „Je to limewash. Prírodná vápenná farba.“
        [P(0.1, 0.1, 0.4, 0.25), lambda t, d: S(0.0 + t, 1.05 + 0.08 * t / d),
         lambda t, d: S(0.6 + t * 0.5, 1.9, 0.2, 0.3)],
        # 3 „Matná, priedušná a každá stena je originál.“
        [lambda t, d: S(1.0 + t * 0.5, 2.3 + 0.15 * t / d, 0.22, 0.30),
         P(0.6, 0.5, 0.3, 0.7, 1.15, 1.3),
         lambda t, d: S(2.6 + t * 0.8, 1.08 + 0.05 * t / d, 0.55, 0.5)],
        # 4 „Nanášam ju ručne, ťah po ťahu.“
        [brush, lambda t, d: S(4.4 + t, 1.6, 0.35, 0.3)],
        # 5 „Výsledok? Interiér ako z magazínu.“
        [lambda t, d: S(6.9 + t, 1.0), lambda t, d: S(8.7 + t * 0.9, 1.05 + 0.05 * t / d)],
        # 6 „Pre byty, domy aj prevádzky.“
        [lambda t, d: S(7.2 + t, 1.1, 0.5, 0.5), P(0.3, 0.6, 0.6, 0.4, 1.0, 1.15), lambda t, d: S(10.0 + t, 1.2, 0.5, 0.45)],
        # 7 „Napíš mi rozmery steny a spočítam ti cenu.“
        [calc],
        # 8 „Sleduj ma!“ – zlúčené s outrom
        [],
    ]
    segs = []
    for li, m in enumerate(meta):
        vs = visuals[li]
        d_line = m["dur"] + GAP
        if not vs:
            continue
        n = len(vs)
        for k, fn in enumerate(vs):
            segs.append((starts[li] + d_line * k / n, d_line / n, fn, True))

    def outro(t, d):
        if t < 0.25:  # rýchly fade z kalkulačky
            base = anim_calculator(A, 3.3)
            return Image.fromarray((np.asarray(base, np.float32) * (1 - ease(t / 0.25))).astype(np.uint8))
        im = Image.new("RGB", (W, H), (8, 7, 6))
        tt = t - 0.25
        a = ease_out(tt / 0.3)
        s = int(300 * (0.85 + 0.15 * a))
        colored = int(tt / 0.45) % 2 == 1
        ic = (A.ig_c if colored else A.ig_w).resize((s, s), Image.LANCZOS)
        ic.putalpha(Image.fromarray((np.asarray(ic.getchannel("A"), np.float32) * a).astype(np.uint8)))
        im.paste(ic, ((W - s) // 2, 700 - s // 2), ic)
        d_ = ImageDraw.Draw(im)
        c = tuple(int(v * a) for v in CREAM)
        g = tuple(int(v * ease((tt - 0.15) / 0.3)) for v in GOLD_L)
        f = font(F_SANS_XB, 64)
        d_.text(((W - d_.textlength(HANDLE, font=f)) / 2, 930), HANDLE, font=f, fill=c)
        f2 = font(F_SERIF_I, 92)
        d_.text(((W - d_.textlength("sleduj ma", font=f2)) / 2, 1060), "sleduj ma", font=f2, fill=g)
        f3 = font(F_SANS_M, 36)
        txt = "Michaelo Cavalli · limewash"
        gg = tuple(int(v * ease((tt - 0.3) / 0.3)) for v in (190, 180, 165))
        d_.text(((W - d_.textlength(txt, font=f3)) / 2, 1200), txt, font=f3, fill=gg)
        k = 1 - ease((t - (d - 0.3)) / 0.3)
        if k < 1:
            im = Image.fromarray((np.asarray(im, np.float32) * k).astype(np.uint8))
        return im
    o0 = starts[-1]
    segs.append((o0, total - o0, outro, False))
    caps = [c for c in build_captions(meta, starts) if c[0] < o0 - 0.1]  # v outre bez titulkov (text je v logu)
    return segs, caps, starts, total

# ------------------------------------------------------------------ audio (syntéza hudby + zvukových akcentov)
def make_audio(total, cut_times, path, sr=44100):
    n = int(total * sr)
    t = np.arange(n) / sr
    out = np.zeros(n, np.float32)
    bpm = 118.0
    beat = 60 / bpm
    # akordy Am – F – C – G (teplý pad)
    chords = [[220.0, 261.63, 329.63], [174.61, 220.0, 261.63], [196.0, 261.63, 329.63], [196.0, 246.94, 293.66]]
    bar = beat * 4
    for i in range(int(total / bar) + 1):
        s, e = int(i * bar * sr), min(int((i + 1) * bar * sr), n)
        if s >= n:
            break
        tt = t[s:e] - i * bar
        env = np.minimum(tt / 0.4, 1) * np.exp(-tt * 0.25)
        for f in chords[i % 4]:
            out[s:e] += 0.05 * env * (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * f * 2 * tt))
        # bas
        bf = chords[i % 4][0] / 2
        out[s:e] += 0.10 * np.exp(-tt * 1.5) * np.sin(2 * np.pi * bf * tt)
    rng = np.random.default_rng(0)
    start_beats = 0.0  # rytmika hneď od začiatku
    for k in range(int(total / beat)):
        bt = k * beat
        if bt > total - 1.2:
            break
        s = int(bt * sr)
        L = int(0.35 * sr)
        tt = np.arange(min(L, n - s)) / sr
        if bt >= start_beats or k % 2 == 0:
            kick = np.sin(2 * np.pi * (50 + 90 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 9)
            out[s:s + len(tt)] += 0.45 * kick
        if bt >= start_beats:  # hi-hat na pol dobách
            s2 = int((bt + beat / 2) * sr)
            Lh = min(int(0.05 * sr), n - s2)
            if Lh > 0:
                hh = rng.standard_normal(Lh) * np.exp(-np.arange(Lh) / sr * 90)
                hh = np.diff(np.concatenate([[0], hh]))
                out[s2:s2 + Lh] += 0.06 * hh
    # whoosh pri veľkých prechodoch
    for ct in cut_times:
        L = int(0.45 * sr)
        s = int((ct - 0.3) * sr)
        if s < 0 or s + L > n:
            continue
        nz = rng.standard_normal(L)
        k = np.linspace(0, 1, L)
        env = np.sin(np.pi * k) ** 2
        # jednoduchý pohyblivý dolnopriepustný filter
        y = np.zeros(L); a_prev = 0.0
        alpha = 0.02 + 0.5 * k
        for i in range(L):
            a_prev = a_prev + alpha[i] * (nz[i] - a_prev)
            y[i] = a_prev
        out[s:s + L] += 0.35 * env * y
    # ticho na začiatku/konci + fade
    fi = int(0.05 * sr); out[:fi] *= np.linspace(0, 1, fi)
    fo_s = int((total - 1.5) * sr)
    out[fo_s:] *= np.linspace(1, 0, n - fo_s) ** 1.5
    out /= max(np.abs(out).max(), 1e-6)
    out *= 0.9
    st = np.stack([out, out], 1)
    # mierne stereo rozšírenie padu
    st[:, 1] = np.roll(st[:, 1], int(0.008 * sr))
    pcm = (st * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

# ------------------------------------------------------------------ main
HANDLE = "@michaelo.cavalli"

def build_vo_track(meta, starts, total, vo_dir, path, sr=44100):
    out = np.zeros(int(total * sr) + sr, np.float32)
    for m, st in zip(meta, starts):
        with wave.open(os.path.join(vo_dir, m["wav"])) as wf:
            x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768
        i = int(st * sr)
        out[i:i + len(x)] += x
    out = out[:int(total * sr)]
    pcm = (np.clip(out, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(pcm.tobytes())

def main():
    global HANDLE
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="zdrojové video (screen recording)")
    ap.add_argument("--out", default="michaelo_cavalli_reel.mp4")
    ap.add_argument("--handle", default=HANDLE)
    ap.add_argument("--work", default=tempfile.gettempdir())
    ap.add_argument("--vo-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "vo"))
    ap.add_argument("--preview", type=float, default=None, help="vyrenderuj len 1 snímku v čase t (s) do PNG")
    args = ap.parse_args()
    HANDLE = args.handle

    raw = os.path.join(args.work, "src.rgb")
    if not os.path.exists(raw):
        # orez čiernych pásov screen recordingu, koniec pred ovládacím centrom (11.15 s)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", args.src, "-t", "11.15", "-vf",
                        "crop=1206:2142:0:240,scale=1080:1920:flags=lanczos,fps=30",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", raw], check=True)
    src = Source(raw)
    meta = gen_vo(args.vo_dir)
    A = Assets()
    segs, caps, starts, total = build_timeline(src, A, meta)
    seg_starts = [s[0] for s in segs]
    print(f"dĺžka videa: {total:.2f} s, záberov: {len(segs)}", file=sys.stderr)

    def render(T):
        i = max(int(np.searchsorted(seg_starts, T, side="right") - 1), 0)
        st, dur, fn, wm = segs[i]
        im = fn(T - st, dur)
        for a, b, w, acc in caps:
            if a <= T < b:
                draw_center_word(im, w, T - a, accent=acc)
        if wm:
            watermark(im, HANDLE)
        return im

    if args.preview is not None:
        render(args.preview).save(args.out)
        return

    music = os.path.join(args.work, "music.wav")
    make_audio(total, [s[0] for s in segs[1:] if s[1] > 0.5], music)
    vo = os.path.join(args.work, "vo.wav")
    build_vo_track(meta, starts, total, args.vo_dir, vo)

    nframes = int(round(total * FPS))
    afilt = ("[1:a]volume=0.32[m];[2:a]aresample=44100,pan=stereo|c0=c0|c1=c0,"
             "highpass=f=80,acompressor=threshold=-20dB:ratio=3:attack=5:release=80,volume=1.6,asplit[v][sc];"
             "[m][sc]sidechaincompress=threshold=0.02:ratio=8:attack=15:release=250[md];"
             "[md][v]amix=inputs=2:normalize=0,loudnorm=I=-14:TP=-1:LRA=7,aresample=44100[a]")
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", music, "-i", vo,
           "-filter_complex", afilt, "-map", "0:v", "-map", "[a]",
           "-c:v", "libx265", "-preset", "medium", "-b:v", "14M", "-maxrate", "15M", "-bufsize", "24M",
           "-x265-params", "log-level=error", "-tag:v", "hvc1", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
           "-movflags", "+faststart", "-t", f"{total:.3f}", args.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(nframes):
        p.stdin.write(render(f / FPS).convert("RGB").tobytes())
        if f % 150 == 0:
            print(f"  snímka {f}/{nframes}", file=sys.stderr)
    p.stdin.close()
    p.wait()
    if p.returncode:
        sys.exit(p.returncode)
    print("hotovo:", args.out, file=sys.stderr)

if __name__ == "__main__":
    main()
