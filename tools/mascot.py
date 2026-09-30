"""Draw Bobby, bobcat's mascot, into the README banner and the social preview card.

Usage:

    uv run python mascot.py ../assets

The script writes the animated `banner.svg` and renders `social-preview.png`
with `rsvg-convert`, which Homebrew's `librsvg` provides. Bobby starts from
one head drawn as text, one character per pixel. The left half of each row is
drawn, and the right half mirrors it.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess

# The colors each grid character stands for. D is the outline and the ear tufts,
# M the markings, L the fur, W the muzzle, ruff, and inner ears, E the iris, P
# the pupil, N the nose, B the blush, and R the light rim that keeps the
# outline visible on dark backgrounds. A dot is transparent.
PALETTE = {
    "D": "#2b1a10",
    "M": "#7a4724",
    "L": "#c28d5a",
    "W": "#f4e4c6",
    "E": "#e0b53c",
    "P": "#2b1a10",
    "N": "#b8685a",
    "B": "#e39a86",
    "R": "#f4e4c6",
}

# The left half of the 32 by 27 head. Tall tufted ears sit on a forehead with
# dark stripes, and a short ruff flares at the cheeks above a short round chin.
HEAD_32 = [
    ".....D..........",
    ".....D..........",
    ".....DD.........",
    "....DDD.........",
    "....DWLD........",
    "....DWWLD.......",
    "....DWWLD.......",
    "....DWWLLD......",
    "....DWWLLLD.....",
    "....DWWLLLLDDDDD",
    "....DLWLLLLMLMLM",
    "....DLLLLLLMLMLM",
    "....DLLLLLLLLMLL",
    "...DLLLLLLLLLLLL",
    "...DLLLLWWWWLLLL",
    "...DLLLWDDDDWLLL",
    "...DLLLDWPEEDLLL",
    "...DLLLDEPEEDLLL",
    "...DLLLWDDDDWLLL",
    "...DLLLLWWWWLLLW",
    "..DLLLLLLLLLLWNN",
    "..DWLLLLLLLLWWWD",
    ".DWDLLLLLLLWWDWD",
    "..DWDLLLLLLWWWDW",
    "...DDLLLLLLLWWWW",
    ".....DDLLLLLLWWW",
    ".......DDDDDDDDD",
]

# Details the doubled head adds, as (x, y, shade) in the left half.
HEAD_64_DETAILS = [
    # Spots on the forehead and the cheeks.
    (12, 24, "M"), (18, 26, "M"), (9, 30, "M"), (8, 38, "M"),
    # Two dark lines running from the outer corner of each eye across the cheek.
    (12, 39, "M"), (11, 40, "M"), (10, 41, "M"), (9, 42, "M"),
    (13, 42, "M"), (12, 43, "M"), (11, 44, "M"), (10, 45, "M"),
    # Rows of dark dots across the whisker pads.
    (22, 46, "M"), (24, 46, "M"), (21, 48, "M"), (23, 48, "M"),
    # Dark strands through the ruff.
    (5, 44, "M"), (7, 47, "M"),
]  # fmt: skip

# The eyes of each expression, ten pixels wide and seven tall. A dot keeps the
# fur beneath. Both eyes use the template unmirrored, so a highlight falls on
# the same side of both.
EYES = {
    "open": [
        "..DDDDDD..",
        ".DEWWEEED.",
        "DEEWWPPEED",
        "DEEPPPPEED",
        "DEEPPPWEED",
        ".DEEPPEED.",
        "..DDDDDD..",
    ],
    "closed": [
        "..........",
        "..........",
        "..........",
        ".DDDDDDDD.",
        "..........",
        "..........",
        "..........",
    ],
    "happy": [
        "..........",
        "....DD....",
        "...D..D...",
        "..D....D..",
        ".D......D.",
        "..........",
        "..........",
    ],
    "sleepy": [
        "..........",
        "..........",
        "DDDDDDDDDD",
        "DEEPPPPEED",
        ".DEEPPEED.",
        "..DDDDDD..",
        "..........",
    ],
    "thinking": [
        "..DDDDDD..",
        ".DEPWPPED.",
        "DEEPPPPEED",
        "DEEEPPEEED",
        "DEEEEEEEED",
        ".DEEEEEED.",
        "..DDDDDD..",
    ],
}
EYE_TOP = 30
EYE_LEFTS = (15, 39)

# The glyphs each expression draws above the head, as (glyph, left, top).
GLYPHS = {
    "sleepy": [
        (["DDDDD", "...D.", "..D..", ".D...", "DDDDD"], 35, 4),
        (["DDD", ".D.", "DDD"], 29, 11),
    ],
    "thinking": [
        (["MM", "MM"], 28, 14),
        ([".M.", "MMM", ".M."], 32, 9),
        ([".MM.", "MMMM", "MMMM", ".MM."], 37, 3),
    ],
}

# A cat smile under the nose, centered on the doubled head.
MOUTH = [
    ".....DD.....",
    ".....DD.....",
    "D...D..D...D",
    ".DDD....DDD.",
]
MOUTH_TOP = 42
MOUTH_LEFT = 26

# Each expression shows over the open face during its windows of the banner's
# 16 second loop, in seconds.
BANNER_LOOP = 16.0
BANNER_TIMELINE = {
    "closed": [(3.0, 3.15), (13.0, 13.15)],
    "happy": [(5.0, 7.0)],
    "thinking": [(9.0, 11.5)],
    "sleepy": [(14.0, 16.0)],
}

# The banner's wordmark draws each font pixel as a square of this many pixels,
# inside an outline this many pixels thick.
WORDMARK_SCALE = 5
WORDMARK_OUTLINE = 2

# A 5 by 7 pixel font in the style of 8-bit game text, holding only the
# characters the banner and the social card write.
FONT = {
    "b": ["D....", "D....", "DDDD.", "D...D", "D...D", "D...D", "DDDD."],
    "o": [".....", ".....", ".DDD.", "D...D", "D...D", "D...D", ".DDD."],
    "c": [".....", ".....", ".DDDD", "D....", "D....", "D....", ".DDDD"],
    "a": [".....", ".....", ".DDD.", "....D", ".DDDD", "D...D", ".DDDD"],
    "t": [".D...", ".D...", "DDDD.", ".D...", ".D...", ".D..D", "..DD."],
    "A": [".DDD.", "D...D", "D...D", "DDDDD", "D...D", "D...D", "D...D"],
    "C": [".DDD.", "D...D", "D....", "D....", "D....", "D...D", ".DDD."],
    "E": ["DDDDD", "D....", "D....", "DDDD.", "D....", "D....", "DDDDD"],
    "F": ["DDDDD", "D....", "D....", "DDDD.", "D....", "D....", "D...."],
    "G": [".DDD.", "D...D", "D....", "D.DDD", "D...D", "D...D", ".DDD."],
    "I": [".DDD.", "..D..", "..D..", "..D..", "..D..", "..D..", ".DDD."],
    "L": ["D....", "D....", "D....", "D....", "D....", "D....", "DDDDD"],
    "N": ["D...D", "DD..D", "D.D.D", "D.D.D", "D..DD", "D...D", "D...D"],
    "O": [".DDD.", "D...D", "D...D", "D...D", "D...D", "D...D", ".DDD."],
    "P": ["DDDD.", "D...D", "D...D", "DDDD.", "D....", "D....", "D...."],
    "R": ["DDDD.", "D...D", "D...D", "DDDD.", "D.D..", "D..D.", "D...D"],
    "S": [".DDD.", "D...D", "D....", ".DDD.", "....D", "D...D", ".DDD."],
    ".": [".....", ".....", ".....", ".....", ".....", ".DD..", ".DD.."],
    " ": [".....", ".....", ".....", ".....", ".....", ".....", "....."],
}


def mirror(left: list[str]) -> list[str]:
    """Return the full rows of a sprite whose left halves are `left`."""
    width = len(left[0])
    for index, half in enumerate(left):
        if len(half) != width:
            raise ValueError(f"row {index} has {len(half)} pixels, expected {width}")
    return [half + half[::-1] for half in left]


def scale2x(grid: list[str]) -> list[str]:
    """Double `grid` with the Scale2x pixel-art rule, which smooths diagonal edges."""
    height, width = len(grid), len(grid[0])

    def at(x: int, y: int) -> str:
        return grid[min(max(y, 0), height - 1)][min(max(x, 0), width - 1)]

    out = [[""] * (2 * width) for _ in range(2 * height)]
    for y in range(height):
        for x in range(width):
            p = at(x, y)
            a, b, c, d = at(x, y - 1), at(x + 1, y), at(x - 1, y), at(x, y + 1)
            out[2 * y][2 * x] = a if c == a and c != d and a != b else p
            out[2 * y][2 * x + 1] = b if a == b and a != c and b != d else p
            out[2 * y + 1][2 * x] = c if d == c and d != b and c != a else p
            out[2 * y + 1][2 * x + 1] = d if b == d and b != a and d != c else p
    return ["".join(row) for row in out]


def stamp(grid: list[list[str]], glyph: list[str], left: int, top: int) -> None:
    """Draw `glyph` onto `grid` with its top-left corner at (`left`, `top`), skipping dots."""
    for dy, row in enumerate(glyph):
        for dx, shade in enumerate(row):
            if shade != ".":
                grid[top + dy][left + dx] = shade


def face(expression: str) -> list[list[str]]:
    """Return Bobby's 64 pixel wide head with the expression `expression`.

    The head is the doubled 32 pixel head with its fine detail: the spots, the whiskers, the
    ruff strands, the smile, the expression's eyes, and the glyphs above the head.
    """
    grid = [list(row) for row in scale2x(mirror(HEAD_32))]
    # Clear the doubled mouth, which reaches two pixels past the new one.
    for y in range(MOUTH_TOP, MOUTH_TOP + len(MOUTH) + 2):
        for x in range(MOUTH_LEFT - 2, MOUTH_LEFT + len(MOUTH[0]) + 2):
            grid[y][x] = "W"
    stamp(grid, MOUTH, MOUTH_LEFT, MOUTH_TOP)
    for x, y, shade in HEAD_64_DETAILS:
        grid[y][x] = shade
        grid[y][63 - x] = shade
    eyes = EYES[expression]
    for eye_left in EYE_LEFTS:
        # Clear the doubled eye and its light ring, which reach past the new eye.
        for y in range(EYE_TOP - 2, EYE_TOP + len(eyes) + 3):
            for x in range(eye_left - 2, eye_left + len(eyes[0]) + 2):
                grid[y][x] = "L"
        # A bobcat has a crescent of light fur under each eye. A full ring makes the eyes stare.
        drawn = {
            (eye_left + dx, EYE_TOP + dy)
            for dy, row in enumerate(eyes)
            for dx, shade in enumerate(row)
            if shade != "."
        }
        for x, y in drawn:
            if (x, y + 1) not in drawn and y >= EYE_TOP + len(eyes) // 2:
                grid[y + 1][x] = "W"
        stamp(grid, eyes, eye_left, EYE_TOP)
    # A faint blush under each eye warms the face.
    for x in (14, 15, 16, 47, 48, 49):
        grid[EYE_TOP + 9][x] = "B"
    for glyph, left, top in GLYPHS.get(expression, []):
        stamp(grid, glyph, left, top)
    return grid


def rim(grid: list[list[str]]) -> list[str]:
    """Return `grid` padded by one pixel, with a light rim around every drawn pixel."""
    height, width = len(grid), len(grid[0])
    padded = [["."] * (width + 2)] + [["."] + row + ["."] for row in grid] + [["."] * (width + 2)]
    out = [row[:] for row in padded]
    for y in range(height + 2):
        for x in range(width + 2):
            if padded[y][x] != ".":
                continue
            neighbors = ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
            if any(
                0 <= nx < width + 2 and 0 <= ny < height + 2 and padded[ny][nx] != "."
                for nx, ny in neighbors
            ):
                out[y][x] = "R"
    return ["".join(row) for row in out]


def rects(grid: list[str], only: set[tuple[int, int]] | None = None) -> list[str]:
    """Return one SVG rect per horizontal run of one color in `grid`, limited to `only`."""

    def kept(x: int, y: int) -> bool:
        return only is None or (x, y) in only

    out = []
    for y, row in enumerate(grid):
        x = 0
        while x < len(row):
            shade = row[x]
            end = x + 1
            while end < len(row) and row[end] == shade and kept(end, y) == kept(x, y):
                end += 1
            if shade in PALETTE and kept(x, y):
                out.append(
                    f'<rect x="{x}" y="{y}" width="{end - x}" height="1" fill="{PALETTE[shade]}"/>'
                )
            x = end
    return out


def lettering(text: str, shade: str) -> list[str]:
    """Return `text` as pixel rows in `shade`, with strokes doubled in width as in game fonts."""
    rows = []
    for y in range(7):
        row = []
        for char in text:
            glyph = FONT[char][y]
            bold = "".join(
                "D" if glyph[x] == "D" or (x > 0 and glyph[x - 1] == "D") else "."
                for x in range(len(glyph))
            )
            row.append(bold + ("D" if glyph[-1] == "D" else ".") + ".")
        rows.append("".join(row).replace("D", shade))
    return rows


def wordmark(text: str) -> list[list[str]]:
    """Return `text` as a big pixel wordmark: fur inside a dark outline."""
    font = lettering(text, "D")
    pad = WORDMARK_OUTLINE
    height = len(font) * WORDMARK_SCALE + 2 * pad
    width = len(font[0]) * WORDMARK_SCALE + 2 * pad
    letter = [[False] * width for _ in range(height)]
    for fy, row in enumerate(font):
        for fx, shade in enumerate(row):
            if shade == "D":
                for dy in range(WORDMARK_SCALE):
                    for dx in range(WORDMARK_SCALE):
                        y = pad + fy * WORDMARK_SCALE + dy
                        x = pad + fx * WORDMARK_SCALE + dx
                        letter[y][x] = True
    grid = [["."] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            if letter[y][x]:
                grid[y][x] = "L"
            elif any(
                letter[ny][nx]
                for ny in range(max(0, y - pad), min(height, y + pad + 1))
                for nx in range(max(0, x - pad), min(width, x + pad + 1))
            ):
                grid[y][x] = "D"
    return grid


def banner_canvas(head: list[list[str]], word: list[list[str]]) -> list[list[str]]:
    """Return `head` on the left and `word` on the right, both centered vertically."""
    gap = 8
    height = max(len(head), len(word))
    width = len(head[0]) + gap + len(word[0])
    canvas = [["."] * width for _ in range(height)]
    stamp(canvas, ["".join(row) for row in head], 0, (height - len(head)) // 2)
    stamp(
        canvas,
        ["".join(row) for row in word],
        len(head[0]) + gap,
        (height - len(word)) // 2,
    )
    return canvas


def banner(scale: int) -> str:
    """Return the animated README banner: Bobby, who changes expressions, and the wordmark."""
    word = wordmark("bobcat")
    base = rim(banner_canvas(face("open"), word))
    layers = []
    for expression, windows in BANNER_TIMELINE.items():
        grid = rim(banner_canvas(face(expression), word))
        changed = {
            (x, y)
            for y, (a, b) in enumerate(zip(base, grid, strict=True))
            for x, (p, q) in enumerate(zip(a, b, strict=True))
            if p != q and q != "."
        }
        times = ["0"]
        values = ["0"]
        for start, end in windows:
            times += [
                f"{start / BANNER_LOOP:.4f}",
                f"{min(end / BANNER_LOOP, 1.0):.4f}",
            ]
            values += ["1", "0"]
        pixels = "\n    ".join(rects(grid, changed))
        layers.append(
            '  <g opacity="0">\n'
            f'    <animate attributeName="opacity" values="{";".join(values)}" '
            f'keyTimes="{";".join(times)}" dur="{BANNER_LOOP:g}s" '
            'repeatCount="indefinite" calcMode="discrete"/>\n'
            f"    {pixels}\n  </g>\n"
        )
    height, width = len(base), len(base[0])
    body = "\n  ".join(rects(base))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width * scale}" height="{height * scale}" shape-rendering="crispEdges">\n'
        f"  <title>bobcat</title>\n  {body}\n{''.join(layers)}</svg>\n"
    )


def pixel_group(grid: list[str], left: int, top: int, scale: int) -> str:
    """Return an SVG group that draws `grid` at (`left`, `top`) with `scale` pixel squares."""
    pixels = "\n    ".join(rects(grid))
    return (
        f'  <g transform="translate({left} {top}) scale({scale})" '
        f'shape-rendering="crispEdges">\n    {pixels}\n  </g>\n'
    )


def social_card() -> str:
    """Return the 1280 by 640 social preview card, with Bobby beside the tagline."""
    title = lettering("bobcat", "D")
    shadow = lettering("bobcat", "L")
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="640" '
        'viewBox="0 0 1280 640">\n'
        f'  <rect width="1280" height="640" fill="{PALETTE["W"]}"/>\n'
        + pixel_group(rim(face("open")), 104, 110, 7)
        + pixel_group(shadow, 613, 243, 13)
        + pixel_group(title, 600, 230, 13)
        + pixel_group(lettering("AN INFERENCE ENGINE", "M"), 602, 372, 4)
        + pixel_group(lettering("FOR APPLE SILICON.", "M"), 602, 420, 4)
        + "</svg>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw Bobby, bobcat's mascot.")
    parser.add_argument("out", type=pathlib.Path, help="the directory to write into")
    args = parser.parse_args()
    out: pathlib.Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    (out / "banner.svg").write_text(banner(2))
    card = out / "social-preview.svg"
    card.write_text(social_card())
    subprocess.run(
        ["rsvg-convert", "-o", str(out / "social-preview.png"), str(card)],
        check=True,
    )
    card.unlink()


if __name__ == "__main__":
    main()
