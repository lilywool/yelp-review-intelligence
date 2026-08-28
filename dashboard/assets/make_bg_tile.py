"""Generate a seamless, scattered Yelp-logo background tile for the dashboard.

Places the logo at varied sizes/rotations/opacities/tints (orange + black + red)
across a square canvas, wrapping placements at the edges so the tile repeats
seamlessly with CSS `background-repeat: repeat`.

Run:
    .venv/Scripts/python.exe dashboard/assets/make_bg_tile.py
"""

import random

from PIL import Image

ASSETS_DIR = __import__("pathlib").Path(__file__).resolve().parent
LOGO_PATH = ASSETS_DIR / "yelp_transparent_logo.png"
OUTPUT_PATH = ASSETS_DIR / "yelp_bg_tile.png"

TILE_SIZE = 900
LOGO_SIZES = [70, 100, 140]
OPACITIES = [14, 22, 30]
# (r, g, b) tints applied to the logo silhouette; keeps the red/orange/black scheme
TINTS = [(240, 100, 35), (17, 17, 17), (201, 73, 20), (196, 30, 20)]

random.seed(7)


def tinted_logo(logo: Image.Image, size: int, tint: tuple, opacity: int) -> Image.Image:
    resized = logo.resize((size, size), Image.LANCZOS)
    alpha = resized.split()[3].point(lambda a: a * opacity // 100)
    solid = Image.new("RGBA", resized.size, tint + (255,))
    solid.putalpha(alpha)
    return solid


def stamp_wrapped(canvas: Image.Image, sprite: Image.Image, cx: int, cy: int) -> None:
    w, h = sprite.size
    x, y = cx - w // 2, cy - h // 2
    for ox in (-TILE_SIZE, 0, TILE_SIZE):
        for oy in (-TILE_SIZE, 0, TILE_SIZE):
            canvas.alpha_composite(sprite, (x + ox, y + oy))


def main() -> None:
    logo = Image.open(LOGO_PATH).convert("RGBA")
    canvas = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))

    # Jittered grid so logos spread evenly across the tile instead of clumping,
    # while still looking organic (random offset/size/rotation/tint per cell).
    grid = 4
    cell = TILE_SIZE / grid
    for row in range(grid):
        for col in range(grid):
            size = random.choice(LOGO_SIZES)
            opacity = random.choice(OPACITIES)
            tint = random.choice(TINTS)
            angle = random.uniform(0, 360)

            sprite = tinted_logo(logo, size, tint, opacity)
            sprite = sprite.rotate(angle, expand=True, resample=Image.BICUBIC)

            cx = int((col + 0.5) * cell + random.uniform(-cell * 0.25, cell * 0.25))
            cy = int((row + 0.5) * cell + random.uniform(-cell * 0.25, cell * 0.25))
            stamp_wrapped(canvas, sprite, cx, cy)

    canvas.save(OUTPUT_PATH)
    print(f"Saved tile: {OUTPUT_PATH} ({TILE_SIZE}x{TILE_SIZE})")


if __name__ == "__main__":
    main()
