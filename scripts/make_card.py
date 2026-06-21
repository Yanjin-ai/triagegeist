"""Render the Kaggle writeup Card (1200x630) and Thumbnail (1000x1000) as PNGs.

    python scripts/make_card.py   ->  docs/assets/card.png, docs/assets/thumbnail.png

Pure Pillow (no SVG rasterizer needed). Dark theme matching the product UI.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"

BG = (15, 20, 25)
PANEL = (22, 28, 36)
LINE = (42, 52, 65)
TEXT = (230, 237, 243)
MUTED = (139, 152, 168)
ACCENT = (74, 163, 255)
GREEN = (62, 207, 142)
ESI = {1: (255, 77, 79), 2: (255, 140, 59), 3: (245, 197, 24), 4: (62, 207, 142), 5: (125, 135, 148)}

BOLD = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf", "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc"]
REG = ["/System/Library/Fonts/Supplemental/Arial.ttf",
       "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
       "/System/Library/Fonts/SFNSText.ttf"]


def font(size, bold=False):
    for p in (BOLD if bold else REG):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def tw(d, t, f):
    return d.textlength(t, font=f)


def cross(d, x, y, s, col):
    a = s // 3
    d.rounded_rectangle((x + a, y, x + 2 * a, y + s), a // 2, fill=col)
    d.rounded_rectangle((x, y + a, x + s, y + 2 * a), a // 2, fill=col)


def pills(d, x, y, h, gap, nums, fsize):
    f = font(fsize, True)
    for n in nums:
        d.rounded_rectangle((x, y, x + h, y + h), 10, fill=ESI[n])
        c = (4, 18, 31) if n != 5 else (255, 255, 255)
        d.text((x + h / 2 - tw(d, str(n), f) / 2, y + h / 2 - fsize * 0.62), str(n), font=f, fill=c)
        x += h + gap


def chip(d, x, y, text, fsize=26, fg=TEXT, border=LINE, fill=PANEL):
    f = font(fsize, True)
    w = tw(d, text, f) + 34
    d.rounded_rectangle((x, y, x + w, y + fsize + 22), 10, fill=fill, outline=border, width=2)
    d.text((x + 17, y + 10), text, font=f, fill=fg)
    return x + w + 14


def card():
    W, H = 1200, 630
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((6, 6, W - 6, H - 6), 22, outline=LINE, width=2)
    d.rectangle((6, 6, W - 6, 14), fill=ACCENT)

    cross(d, 70, 70, 64, ESI[1])
    d.text((156, 64), "Triage Copilot", font=font(74, True), fill=TEXT)
    d.text((158, 158), "An honest ED triage decision-support stack", font=font(32), fill=MUTED)

    pills(d, 72, 250, 64, 16, [1, 2, 3, 4, 5], 30)
    d.text((470, 264), "predict  →  calibrate  →  audit  →  operate", font=font(34, True), fill=ACCENT)

    # honest-core line
    d.rounded_rectangle((70, 348, W - 70, 420), 12, fill=(19, 32, 46), outline=(37, 65, 92), width=2)
    d.text((90, 363), "Honest core: a text-blind structured model (the label is leaked by the complaint text),",
           font=font(24), fill=(207, 216, 227))
    d.text((90, 389), "wrapped in calibration, conformal uncertainty, fairness, and supervisable operations.",
           font=font(24), fill=(207, 216, 227))

    x = 72
    for t, fg, bd in [("QWK 0.93 · text-blind", (4, 18, 31), None), ("conformal sets", TEXT, LINE),
                      ("fairness + CI", TEXT, LINE), ("oversight + audit", TEXT, LINE)]:
        x = chip(d, x, 452, t, 24, fg, bd or GREEN, GREEN if bd is None else PANEL)
    d.text((72, 560), "synthetic data · research prototype (TRL 3–4) · not for clinical use · Triagegeist / Kaggle",
           font=font(22), fill=MUTED)
    im.save(OUT / "card.png")
    print("wrote", OUT / "card.png", im.size)


def thumbnail():
    W = H = 1000
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((10, 10, W - 10, H - 10), 28, outline=LINE, width=3)
    cross(d, W // 2 - 60, 130, 120, ESI[1])
    t = "Triage Copilot"
    d.text((W / 2 - tw(d, t, font(76, True)) / 2, 300), t, font=font(76, True), fill=TEXT)
    s = "an honest ED triage AI"
    d.text((W / 2 - tw(d, s, font(38)) / 2, 392), s, font=font(38), fill=MUTED)
    # centered pills
    n = 5; h = 92; gap = 22
    total = n * h + (n - 1) * gap
    pills(d, (W - total) // 2, 470, h, gap, [1, 2, 3, 4, 5], 42)
    # metric chips centered
    f = font(30, True)
    chips = ["text-blind  QWK 0.93", "calibrated · conformal · fair"]
    ys = 610
    for c in chips:
        w = tw(d, c, f) + 40
        d.rounded_rectangle(((W - w) // 2, ys, (W + w) // 2, ys + 56), 12, fill=PANEL, outline=LINE, width=2)
        d.text(((W - tw(d, c, f)) // 2, ys + 12), c, font=f, fill=TEXT)
        ys += 72
    d.text((W / 2 - tw(d, "predict · calibrate · audit · operate", font(30, True)) / 2, 770),
           "predict · calibrate · audit · operate", font=font(30, True), fill=ACCENT)
    d.text((W / 2 - tw(d, "research prototype · not for clinical use", font(24)) / 2, 900),
           "research prototype · not for clinical use", font=font(24), fill=MUTED)
    im.save(OUT / "thumbnail.png")
    print("wrote", OUT / "thumbnail.png", im.size)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    card()
    thumbnail()
