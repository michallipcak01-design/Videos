#!/usr/bin/env python3
"""Predajné reels video pre Michaelo Cavalli (limewash – matná dekoratívna vápenná farba) podľa style.md.

Použitie:
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

def draw_phrase(im, lines, t_local, x=90, y=1180, size=78, color=CREAM, accent=GOLD_L, shade=True):
    """Variant B: dvojriadková serif fráza, kľúčové slovo kurzívou. lines: [[(text, italic), ...], ...]"""
    a = ease(t_local / 0.35)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    lh = size * 1.18
    if shade:
        grad = np.zeros((H, W), np.uint8)
        y0 = int(y - 220)
        ramp = np.clip((np.arange(H) - y0) / 260.0, 0, 1) * 170
        grad[:] = ramp[:, None].astype(np.uint8)
        g = Image.fromarray(grad)
        black = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        black.putalpha(Image.fromarray((np.asarray(g, np.float32) * a).astype(np.uint8)))
        im.paste(black, (0, 0), black)
    off = (1 - a) * 30
    for li, parts in enumerate(lines):
        cx = x
        for txt, it in parts:
            f = font(F_SERIF_I if it else F_SERIF, size * (1.08 if it else 1.0))
            col = accent if it else color
            d.text((cx, y + li * lh + off), txt, font=f, fill=col + (int(255 * a),))
            cx += d.textlength(txt, font=f)
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

def step_badge(im, num, title, t_local, x=90, y=225, sub=None):
    a = ease_out(t_local / 0.3)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    r = 46 * (0.6 + 0.4 * a)
    cx, cy = x + 46, y + 46
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=GOLD + (int(255 * a),))
    f = font(F_SANS_XB, 50 * (0.6 + 0.4 * a))
    tw, th, b = text_size(d, str(num), f)
    d.text((cx - tw / 2 - b[0], cy - th / 2 - b[1]), str(num), font=f, fill=DARK + (int(255 * a),))
    f2 = font(F_SANS_XB, 58)
    d.text((x + 120 + (1 - a) * 40, y + 6), title.upper(), font=f2, fill=CREAM + (int(255 * a),))
    if sub:
        f3 = font(F_SERIF_I, 42)
        d.text((x + 122 + (1 - a) * 40, y + 74), sub, font=f3, fill=GOLD_L + (int(230 * a),))
    im.paste(lay, (0, 0), lay)

def bg_dark():
    yy = np.linspace(0, 1, H)[:, None, None]
    top = np.array(DARK2, np.float32); bot = np.array(DARK, np.float32)
    g = top + (bot - top) * yy
    return Image.fromarray(np.broadcast_to(g, (H, W, 3)).astype(np.uint8))

# ------------------------------------------------------------------ animated scenes
class Assets:
    def __init__(self):
        print("generujem textúry…", file=sys.stderr)
        self.plaster = make_plaster(1400, 2300, seed=11)
        self.plaster2 = make_plaster(1400, 1400, seed=23, tint=(1.0, 0.95, 0.88))
        self.bg = bg_dark()
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

def scene_boring_room(t, variant):
    """Problém: obyčajná biela stena (studená, plochá, nudná)."""
    im = Image.new("RGB", (W, H), (200, 203, 207))
    d = ImageDraw.Draw(im)
    z = 1.0 + 0.04 * t
    def P(x, y):
        return ((x - W / 2) * z + W / 2, (y - H / 2) * z + H / 2)
    # strop, podlaha, bočné steny
    d.polygon([P(0, 0), P(W, 0), P(860, 360), P(220, 360)], fill=(232, 233, 235))
    d.polygon([P(0, H), P(W, H), P(860, 1460), P(220, 1460)], fill=(150, 146, 140))
    d.polygon([P(0, 0), P(220, 360), P(220, 1460), P(0, H)], fill=(178, 181, 186))
    d.polygon([P(W, 0), P(860, 360), P(860, 1460), P(W, H)], fill=(170, 173, 178))
    d.polygon([P(220, 360), P(860, 360), P(860, 1460), P(220, 1460)], fill=(214, 216, 219))
    for lx in (380, 540, 700):
        x, y = P(lx, 330)
        d.ellipse((x - 14, y - 6, x + 14, y + 6), fill=(255, 255, 250))
    if variant == 1:  # valček – stále tá istá farba
        p = (t * 0.9) % 1.0
        x0, y0 = P(300, 420); x1, y1 = P(780, 1400)
        for k in range(4):
            bx = x0 + (x1 - x0) * (k / 4)
            d.rectangle((bx, y0, bx + (x1 - x0) / 4 - 6, y1), fill=(220, 222, 224))
        ry = y0 + (y1 - y0) * (0.5 + 0.45 * math.sin(t * 5))
        rx = x0 + (x1 - x0) * (min(int(t * 1.4), 3) / 4) + 40
        d.rounded_rectangle((rx, ry - 26, rx + 120, ry + 26), radius=20, fill=(245, 245, 245), outline=(120, 120, 120), width=4)
        d.line((rx + 60, ry + 26, rx + 60, ry + 260), fill=(90, 90, 90), width=10)
    if variant == 2:  # praskliny
        rng = np.random.default_rng(1)
        prog = ease(t / 1.4)
        for c in range(3):
            x, y = P(320 + c * 190, 520 + c * 160)
            pts = [(x, y)]
            for k in range(int(14 * prog)):
                x += rng.uniform(-30, 40); y += rng.uniform(20, 55)
                pts.append((x, y))
            if len(pts) > 1:
                d.line(pts, fill=(110, 112, 116), width=5, joint="curve")
        # odlupnutá farba
        if prog > 0.5:
            x, y = P(600, 1180)
            d.polygon([(x, y), (x + 90, y - 20), (x + 120, y + 50), (x + 30, y + 80)], fill=(178, 176, 170))
    return im

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

def anim_swatches(A, t, w, h):
    im = Image.new("RGB", (w, h), (40, 34, 30))
    d = ImageDraw.Draw(im)
    tints = [((1.0, 0.94, 0.84), "Piesková"), ((0.95, 0.95, 0.95), "Perlová"), ((0.62, 0.60, 0.58), "Grafit"),
             ((1.0, 0.86, 0.62), "Zlatá"), ((1.0, 0.86, 0.84), "Blush"), ((0.80, 0.72, 0.62), "Mocca")]
    base = np.asarray(A.plaster2.resize((260, 260)), np.float32) / 255.0
    sel = min(int(t / 1.2), len(tints) - 1)
    for i, (tn, name) in enumerate(tints):
        col, row = i % 3, i // 3
        x = 50 + col * (w - 100) / 3 + 10
        y = 70 + row * 330
        a = ease_out((t - i * 0.12) / 0.4)
        if a <= 0:
            continue
        sw = (base * np.array(tn)) * 255
        s = Image.fromarray(np.clip(sw, 0, 255).astype(np.uint8)).resize((int(w / 3 - 60), 240))
        yy = int(y + (1 - a) * 60)
        rounded_paste(im, s, (int(x), yy, int(x) + s.size[0], yy + 240), radius=24, shadow=False,
                      border=GOLD_L + (255,) if i == sel else (255, 255, 255, 40))
        f = font(F_SANS_M, 30)
        d = ImageDraw.Draw(im)
        d.text((x + 6, yy + 254), name, font=f, fill=GOLD_L if i == sel else (200, 192, 180))
    return im

def anim_brush(A, t, dur, w, h):
    flat = Image.new("RGB", (w, h), (206, 200, 190))
    tex = A.plaster.resize((w, h), Image.BILINEAR, box=(100, 300, 100 + 1100, 300 + 1100 * h / w))
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    n = len(A.strokes)
    prog = min(t / (dur * 0.92), 1.0) * n
    last = None
    for i, (cx, cy, ang, L, width) in enumerate(A.strokes):
        if i > prog:
            break
        frac = min(prog - i, 1.0)
        dx, dy = math.cos(ang) * L / 2, math.sin(ang) * L / 2
        x0, y0 = cx - dx, cy - dy
        x1, y1 = x0 + 2 * dx * frac, y0 + 2 * dy * frac
        d.line((x0, y0, x1, y1), fill=200, width=int(width))
        last = (x1, y1, ang)
    m = m.filter(ImageFilter.GaussianBlur(14))
    im = Image.composite(tex, flat, m)
    if last is not None and t < dur * 0.95:
        x, y, ang = last
        lay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        dd = ImageDraw.Draw(lay)
        # široký štetec na limewash (štetiny + drevené telo + rukoväť)
        pa = ang + math.pi / 2
        ca, sa = math.cos(pa), math.sin(pa)
        def R(px, py):
            return (x + px * ca - py * sa, y + px * sa + py * ca)
        dd.polygon([R(-95, -10), R(95, -10), R(85, 60), R(-85, 60)], fill=(232, 220, 196, 255))   # štetiny
        dd.polygon([R(-100, 60), R(100, 60), R(100, 110), R(-100, 110)], fill=(150, 104, 62, 255))  # telo
        dd.polygon([R(-18, 110), R(18, 110), R(14, 260), R(-14, 260)], fill=(120, 80, 46, 255))    # rukoväť
        im.paste(lay, (0, 0), lay)
    return im

def anim_matte(A, t, w, h):
    """Matný povrch: pohyblivé bočné svetlo zvýrazní hĺbku, bez lesku."""
    tex = A.plaster.resize((w, h), Image.BILINEAR, box=(200, 900, 200 + 1000, 900 + 1000 * h / w))
    a = np.asarray(tex, np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lx = w * (0.15 + 0.7 * ease(t / 5.0))
    light = 0.78 + 0.32 * np.exp(-(((xx - lx) / (w * 0.55)) ** 2 + ((yy - h * 0.3) / (h * 0.9)) ** 2))
    a = a * light[..., None] * np.array([1.02, 1.0, 0.96])
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im, "RGBA")
    al = int(255 * ease((t - 0.6) / 0.5))
    if al > 0:  # štítok MATT
        d.rounded_rectangle((w - 250, h - 110, w - 40, h - 40), radius=34, fill=(20, 17, 15, int(al * 0.75)))
        f = font(F_SANS_XB, 38)
        d.text((w - 145 - d.textlength("MATT", font=f) / 2, h - 97), "MATT", font=f, fill=GOLD_L + (al,))
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

# ------------------------------------------------------------------ timeline
def build_timeline(src, A):
    """Každý segment: (dĺžka, render(t) -> PIL.Image, [titulky])."""
    segs = []
    def add(dur, fn, caps=(), wm=True):
        segs.append((dur, fn, list(caps), wm))

    # ---- 0–2.0 s HOOK: flash montáž (8 strihov × 0.25 s)
    flash = [(0.3, 1.25, 0.25, 0.30), (2.9, 1.0, .5, .5), (1.0, 1.35, 0.20, 0.25), (9.6, 1.1, .5, .5),
             (4.7, 1.3, 0.55, 0.30), (3.0, 1.2, .5, .45), (0.6, 1.5, 0.12, 0.28), (10.2, 1.15, .5, .5)]
    hook_words = ["TVOJA", "TVOJA", "STENA", "STENA", "AKO", "AKO", "Z", "GALÉRIE"]
    for i, (st, z, cx, cy) in enumerate(flash):
        add(0.25, (lambda st=st, z=z, cx=cx, cy=cy: lambda t: src.get(st + t, z + 0.15 * t, cx, cy))())
    segs_hook_caps = [(0.0, 0.5, "TVOJA"), (0.5, 1.0, "STENA"), (1.0, 1.5, "AKO Z"), (1.5, 2.0, "GALÉRIE?")]
    # ---- 2.0–3.4 roztvorené ruky
    add(1.4, lambda t: src.get(2.6 + t * 0.8, 1.08 + 0.05 * t, 0.55, 0.5),
        [(0.0, 1.4, "word", "LUXUS", GOLD_L)])
    # ---- 3.4–11.4 PROBLÉM (animované)
    add(2.0, lambda t: scene_boring_room(t, 0), [(0.0, 1.0, "word", "BIELA", None), (1.0, 2.0, "word", "STENA?", None)])
    add(2.0, lambda t: scene_boring_room(t, 1), [(0.0, 1.0, "word", "PLOCHÁ", None), (1.0, 2.0, "word", "NUDA", None)])
    add(2.0, lambda t: scene_boring_room(t, 2), [(0.0, 1.0, "word", "PRASKLINY", None), (1.0, 2.0, "word", "ŠEDÁ", None)])
    add(2.0, lambda t: scene_boring_room(1.5 + t, 0).filter(ImageFilter.GaussianBlur(2)),
        [(0.0, 2.0, "phrase", [[("každý byt vyzerá", False)], [("rovnako…", True)]])])
    # ---- 11.4–13.4 ZLOM: čierna karta
    def black_card(t):
        im = Image.new("RGB", (W, H), (0, 0, 0))
        d = ImageDraw.Draw(im)
        a = ease(t / 0.5) * (1 - ease((t - 1.75) / 0.25))
        c = tuple(int(v * a) for v in CREAM)
        g = tuple(int(v * a) for v in GOLD_L)
        f1, f2 = font(F_SERIF, 76), font(F_SERIF_I, 96)
        for txt, f, col, y in (("Čo keby tvoja stena", f1, c, 840), ("vyzerala ako umenie?", f2, g, 950)):
            tw = d.textlength(txt, font=f)
            d.text(((W - tw) / 2, y), txt, font=f, fill=col)
        return im
    add(2.0, black_card, wm=False)
    # ---- 13.4–23.4 RIEŠENIE: reálne zábery + jump-zoomy
    add(1.4, lambda t: scene_plaster_pan(A, t, 1.4, 0.1, 0.1, 0.4, 0.25), [(0, 1.4, "word", "LIMEWASH", GOLD_L)])
    add(1.4, lambda t: src.get(0.0 + t * 0.9, 1.05), [(0, 1.4, "word", "VÁPENNÁ", None)])
    add(1.3, lambda t: src.get(1.0 + t * 0.5, 2.3 + 0.1 * t, 0.22, 0.30), [(0, 1.3, "word", "FARBA", None)])
    add(1.3, lambda t: src.get(1.6 + t * 0.8, 1.15, 0.55, 0.5), [(0, 1.3, "word", "MATNÁ", None)])
    add(1.3, lambda t: src.get(0.4 + t * 0.4, 1.0, 0.5, 0.5).resize((W, H), Image.BILINEAR, box=(40, 120, 640, 1186)),
        [(0, 1.3, "word", "ŤAH", None)])
    add(1.3, lambda t: scene_plaster_pan(A, t, 1.3, 0.6, 0.5, 0.3, 0.7, 1.15, 1.3), [(0, 1.3, "word", "PO ŤAHU", None)])
    add(2.0, lambda t: src.get(4.3 + t * 0.65, 1.10 + 0.04 * t, 0.6, 0.45),
        [(0, 2.0, "phrase", [[("žiadne dve steny", False)], [("nie sú rovnaké", True)]])])
    # ---- 23.4–38.4 KROKY (rozdelená obrazovka)
    def split(fn_top, src_t0, src_rate, num, title, sub, cap2=None):
        def r(t):
            im = A.bg.copy()
            top = fn_top(t, 940, 760)
            rounded_paste(im, top.resize((940, 720)), (70, 390, 1010, 1110), radius=40, border=(255, 255, 255, 30))
            fr = src.get(src_t0 + t * src_rate, 1.0)
            fr = fr.resize((W, H), Image.BILINEAR, box=(80, 600, 1080, 1630))  # orez okolo postavy
            rounded_paste(im, fr, (230, 1150, 850, 1690), radius=40, border=GOLD + (160,))
            step_badge(im, num, title, t, sub=sub(t) if callable(sub) else sub)
            return im
        return r
    add(5.0, split(lambda t, w, h: anim_swatches(A, t, w, h), 5.6, 0.25, 1, "Výber odtieňa",
                   lambda t: "nájdeme tvoj odtieň" if t < 2.5 else "presne k tvojmu interiéru"))
    add(5.0, split(lambda t, w, h: anim_brush(A, t, 5.0, w, h), 6.85, 0.25, 2, "Aplikácia",
                   lambda t: "štetcom, ťahy do kríža" if t < 2.5 else "každý ťah je originál"))
    add(5.0, split(lambda t, w, h: anim_matte(A, t, w, h), 8.6, 0.45, 3, "Matný finiš",
                   lambda t: "zamatový, bez lesku" if t < 2.5 else "hĺbka v každom svetle"))
    # ---- 38.4–42.4 ZHRNUTIE
    add(2.4, lambda t: src.get(8.7 + t, 1.05 + 0.04 * t), [(0, 1.2, "word", "PRÍRODNÁ", None), (1.2, 2.4, "word", "VÁPENNÁ", GOLD_L)])
    quick = [(lambda t: src.get(0.2 + t, 1.6, 0.25, 0.25), "MATNÁ"),
             (lambda t: scene_plaster_pan(A, t, 0.4, 0.2, 0.7, 0.3, 0.75, 1.3, 1.4), "PRIEDUŠNÁ"),
             (lambda t: src.get(3.0 + t, 1.25, 0.55, 0.5), "UNIKÁTNA"),
             (lambda t: src.get(10.4 + t, 1.1), "LUXUSNÁ")]
    for fn, w_ in quick:
        add(0.4, fn, [(0, 0.4, "word", w_, GOLD_L if w_ == "LUXUSNÁ" else None)])
    # ---- 42.4–46.4 CTA kalkulačka
    add(4.0, lambda t: anim_calculator(A, t),
        [(0, 4.0, "phrase", [[("pošli mi rozmery", False)], [("spočítam ti cenu", True)]])])
    # ---- 46.4–50.4 OUTRO
    def outro(t):
        # fade z posledného záberu do tmy, potom logo
        if t < 0.8:
            base = anim_calculator(A, 4.0)
            k = 1 - ease(t / 0.8)
            return Image.fromarray((np.asarray(base, np.float32) * k).astype(np.uint8))
        im = Image.new("RGB", (W, H), (8, 7, 6))
        tt = t - 0.8
        a = ease_out(tt / 0.4)
        s = int(300 * (0.85 + 0.15 * a))
        colored = int(tt / 0.6) % 2 == 1
        ic = (A.ig_c if colored else A.ig_w).resize((s, s), Image.LANCZOS)
        al = Image.fromarray((np.asarray(ic.getchannel("A"), np.float32) * a).astype(np.uint8))
        ic.putalpha(al)
        im.paste(ic, ((W - s) // 2, 640 - s // 2 + 60), ic)
        d = ImageDraw.Draw(im)
        c = tuple(int(v * a) for v in CREAM)
        g = tuple(int(v * ease((tt - 0.3) / 0.4)) for v in GOLD_L)
        f = font(F_SANS_XB, 64)
        tw = d.textlength(HANDLE, font=f)
        d.text(((W - tw) / 2, 930), HANDLE, font=f, fill=c)
        f2 = font(F_SERIF_I, 92)
        for txt, y in (("sleduj ma", 1060),):
            tw = d.textlength(txt, font=f2)
            d.text(((W - tw) / 2, y), txt, font=f2, fill=g)
        f3 = font(F_SANS_M, 36)
        txt = "Michaelo Cavalli · limewash"
        gg = tuple(int(v * ease((tt - 0.6) / 0.4)) for v in (190, 180, 165))
        tw = d.textlength(txt, font=f3)
        d.text(((W - tw) / 2, 1200), txt, font=f3, fill=gg)
        # záverečný fade
        k = 1 - ease((t - 3.7) / 0.3)
        if k < 1:
            im = Image.fromarray((np.asarray(im, np.float32) * k).astype(np.uint8))
        return im
    add(4.0, outro, wm=False)
    return segs, segs_hook_caps

# ------------------------------------------------------------------ audio (syntéza hudby + zvukových akcentov)
def make_audio(total, cut_times, path, sr=44100):
    n = int(total * sr)
    t = np.arange(n) / sr
    out = np.zeros(n, np.float32)
    bpm = 100.0
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
    start_beats = 2.0  # rytmika od ~2 s (po hooku)
    for k in range(int(total / beat)):
        bt = k * beat
        if bt > total - 3.5:
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
    fo_s = int((total - 3.5) * sr)
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

def main():
    global HANDLE
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="zdrojové video (screen recording)")
    ap.add_argument("--out", default="michaelo_cavalli_reel.mp4")
    ap.add_argument("--handle", default=HANDLE)
    ap.add_argument("--work", default=tempfile.gettempdir())
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
    A = Assets()
    segs, hook_caps = build_timeline(src, A)
    total = sum(s[0] for s in segs)
    starts = np.cumsum([0] + [s[0] for s in segs])
    print(f"dĺžka videa: {total:.1f} s, segmentov: {len(segs)}", file=sys.stderr)

    def render(T):
        i = int(np.searchsorted(starts, T, side="right") - 1)
        i = min(i, len(segs) - 1)
        dur, fn, caps, wm = segs[i]
        tl = T - starts[i]
        im = fn(tl)
        if T < 2.0:
            for a, b, w in hook_caps:
                if a <= T < b:
                    draw_center_word(im, w, T - a, accent=GOLD_L if w.startswith("GAL") else None)
        for c in caps:
            a, b, kind = c[0], c[1], c[2]
            if a <= tl < b:
                if kind == "word":
                    draw_center_word(im, c[3], tl - a, accent=c[4])
                else:
                    draw_phrase(im, c[3], tl - a)
        if wm:
            watermark(im, HANDLE)
        return im

    if args.preview is not None:
        render(args.preview).save(args.out)
        return

    # časy veľkých prechodov pre whoosh
    big = [2.0, 3.4, 11.4, 13.4, 23.4, 28.4, 33.4, 38.4, 42.4, 46.4]
    wav = os.path.join(args.work, "music.wav")
    make_audio(total, big, wav)

    nframes = int(round(total * FPS))
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", wav,
           "-filter_complex", "[1:a]loudnorm=I=-15:TP=-1:LRA=7,aresample=44100[a]",
           "-map", "0:v", "-map", "[a]",
           "-c:v", "libx265", "-preset", "medium", "-b:v", "12M", "-maxrate", "15M", "-bufsize", "24M",
           "-x265-params", "log-level=error", "-tag:v", "hvc1", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
           "-movflags", "+faststart", "-shortest", args.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(nframes):
        T = f / FPS
        p.stdin.write(render(T).convert("RGB").tobytes())
        if f % 150 == 0:
            print(f"  snímka {f}/{nframes}", file=sys.stderr)
    p.stdin.close()
    p.wait()
    if p.returncode:
        sys.exit(p.returncode)
    print("hotovo:", args.out, file=sys.stderr)

if __name__ == "__main__":
    main()
