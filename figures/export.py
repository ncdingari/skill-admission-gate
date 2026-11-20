"""Export a pyecharts chart to figures/out/{stem}.html, .png (300 dpi), .pdf and .options.json.

PNG: canvas renderer, selenium (headless Chrome) snapshot with pixel_ratio=3,
then Pillow writes 300 dpi metadata. PDF: the same option rendered with the
SVG renderer, selenium SVG snapshot, then cairosvg.svg2pdf (vector).
The ECharts library is the vendored figures/vendor/echarts.min.js (6.0.0, the
version pyecharts 2.1 references), so rendering needs no network.
The options JSON is the exact chart option (JsCode kept as pyecharts
placeholders); tests read it for axis units, colors and font sizes.
"""
from __future__ import annotations

import argparse
import atexit
import os
import tempfile
from pathlib import Path

import cairosvg
import simplejson
from PIL import Image
from pyecharts.charts.base import default
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
VENDOR = HERE / "vendor"
PIXEL_RATIO = 3
DPI = 300
DELAY_S = 0.6

_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        from selenium import webdriver
        o = webdriver.ChromeOptions()
        o.add_argument("--headless=new")
        o.add_argument("--window-size=1400,2400")
        o.add_argument("--allow-file-access-from-files")
        _driver = webdriver.Chrome(options=o)
        atexit.register(_driver.quit)
    return _driver


def options_json(chart) -> str:
    return simplejson.dumps(chart.get_options(), separators=(",", ":"), default=default,
                            ignore_nan=True, sort_keys=True)


def save(chart, stem: str, out_dir: str | Path | None = None) -> dict:
    out = Path(out_dir) if out_dir else OUT
    out.mkdir(parents=True, exist_ok=True)
    chart.js_host = os.path.relpath(VENDOR, out) + "/"
    html = out / f"{stem}.html"
    png = out / f"{stem}.png"
    pdf = out / f"{stem}.pdf"
    (out / f"{stem}.options.json").write_text(options_json(chart) + "\n")

    chart.renderer = "canvas"
    chart.render(str(html))
    drv = _get_driver()
    make_snapshot(snapshot, str(html), str(png), pixel_ratio=PIXEL_RATIO, delay=DELAY_S, driver=drv)
    with Image.open(png) as im:
        im.load()
        rgb = Image.new("RGB", im.size, (255, 255, 255))
        rgb.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
    rgb.save(png, dpi=(DPI, DPI))

    with tempfile.TemporaryDirectory() as td:
        svg_html = Path(td) / f"{stem}_svg.html"
        chart.js_host = str(VENDOR) + "/"
        chart.renderer = "svg"
        chart.render(str(svg_html))
        svg = Path(td) / f"{stem}.svg"
        make_snapshot(snapshot, str(svg_html), str(svg), delay=DELAY_S, driver=drv)
        cairosvg.svg2pdf(bytestring=svg.read_bytes(), write_to=str(pdf))
    chart.renderer = "canvas"
    chart.js_host = os.path.relpath(VENDOR, out) + "/"
    return {"html": html, "png": png, "pdf": pdf}


def cli(build_data, render, doc: str | None = None):
    """Standard figure-script entry point: build_data() then render(), or --from-data."""
    ap = argparse.ArgumentParser(description=doc)
    ap.add_argument("--from-data", action="store_true",
                    help="re-render from figures/data/ CSVs only (no results/, no npz)")
    ap.add_argument("--out", default=str(OUT), help="output directory (default figures/out)")
    a = ap.parse_args()
    if not a.from_data:
        build_data()
    paths = render(a.out)
    print(", ".join(str(p) for p in paths.values()))
