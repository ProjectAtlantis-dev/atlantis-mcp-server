"""Build a standalone diagnostic preview; never read or write the live DB.

Run from the repository root:
PYTHONPATH=python-server python-server/venv/bin/python \
  python-server/dynamic_functions/Terrain/Test/ocean_texture_preview.py /tmp/ocean-preview
"""

import base64
import io
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from dynamic_functions.Terrain.ocean_texture import (
    detect_white_ocean_gaps,
    texture_ocean_mask,
    texture_pixel_area_m2,
    white_ocean_repair_mask,
)


def build_preview(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    fixture_dir = Path(__file__).with_name("fixtures")
    # Diagnostic fill sampled from the example's dark ocean (RGB median).
    # This is not a change to the live ocean material or to cached textures.
    fill = np.array([10, 20, 25], dtype=np.uint8)
    panels = []
    summary = Image.new("RGB", (960, 1110), (18, 24, 30))
    draw = ImageDraw.Draw(summary)
    draw.text((20, 12), "Ocean imagery gap test: original / detected / candidate fill", fill="white")
    for index, (tile, name) in enumerate((
        ("11-934-28", "ocean_white_wedge_11-934-28.jpg"),
        ("9-230-6", "ocean_white_block_9-230-6.jpg"),
        ("12-1858-52", "ocean_ice_12-1858-52.jpg"),
    )):
        pixels = np.asarray(Image.open(fixture_dir / name).convert("RGB"))
        ocean = texture_ocean_mask(np.ones((65, 65), dtype=bool), pixels.shape[:2])
        area = texture_pixel_area_m2(tile, pixels.shape[:2])
        mask = detect_white_ocean_gaps(pixels, ocean, pixel_area_m2=area)
        repair_mask = white_ocean_repair_mask(pixels, ocean, pixel_area_m2=area)
        marked = pixels.copy()
        marked[mask] = (255, 96, 32)
        repaired = pixels.copy()
        repaired[repair_mask] = fill
        assert np.array_equal(repaired[~repair_mask], pixels[~repair_mask])
        description = f"{tile}: {int(mask.sum()):,} pixels flagged ({mask.mean():.2%})"
        draw.text((20, 40 + index * 350), description, fill="white")
        cards = []
        for column, (label, values) in enumerate((
            ("Original", pixels), ("Detected pixels", marked), ("Candidate fill", repaired),
        )):
            image = Image.fromarray(values)
            png = io.BytesIO()
            image.save(png, format="PNG")
            encoded = base64.b64encode(png.getvalue()).decode("ascii")
            cards.append(f'<figure><figcaption>{label}</figcaption><img alt="{tile}: {label}" src="data:image/png;base64,{encoded}"></figure>')
            summary.paste(image.resize((300, 300), Image.Resampling.NEAREST), (20 + column * 315, 65 + index * 350))
        panels.append(f'<section><h2>{description}</h2><div class="panels">{"".join(cards)}</div></section>')
    draw.text((20, 1090), "Preview only. Original cache unchanged. Fill RGB 10,20,25. Bright JPEG fringe included (max 3px).", fill="white")
    summary.save(destination / "comparison.png")
    html = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ocean texture gap test</title><style>
body{background:#12181e;color:#edf3f7;font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px}
h1{font-size:26px}h2{font-size:19px}.panels{display:flex;gap:16px;flex-wrap:wrap}
figure{margin:0;flex:1;min-width:220px}figcaption{margin:12px 0}img{width:100%;image-rendering:pixelated}
p{max-width:900px;line-height:1.5}section{margin:32px 0}
</style><h1>Ocean texture gap test</h1>
<p>Three original Dataforsyningen textures. GTK50 classifies all three tiles entirely as ocean.
Orange shows the detected uniform white areas. The candidate fill uses RGB (10,20,25),
sampled from dark ocean in the first example. The repair also includes bright neutral
JPEG fringes within three pixels of the accepted core, constrained to confirmed ocean.
All pixels outside that repair mask are unchanged.</p>
<p>This is a diagnostic preview, not a live repair. The colour test cannot distinguish uniformly white sea ice
from missing imagery. Each accepted component must cover at least one hectare
(10,000 square metres in EPSG:3413). Original database textures remain unchanged.</p>'''
    (destination / "index.html").write_text(html + "".join(panels) + "</html>")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: ocean_texture_preview.py OUTPUT_DIRECTORY")
    build_preview(Path(sys.argv[1]))
