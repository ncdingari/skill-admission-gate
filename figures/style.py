"""Shared figure style: Okabe-Ito palette, fonts sized for 0.9 x the TMLR text width (copied from EM_Solute_Korhonen).

Every chart is WIDTH_PX = 1050 px wide. Printed at the JAP single-column width
of 3.37 in that is 311.6 px/in, so 8 pt (1/9 in) is 34.6 px; all text uses at
least MIN_FONT_PX = 36 px (8.3 pt at single-column width).

Figures are built as raw ECharts option dicts (full control over multi-panel
grids) and rendered with pyecharts via ``make_chart``. Every color in an
option comes from PALETTE (Okabe & Ito 2008, https://jfly.uni-koeln.de/color/)
or is the white background / visualMap midpoint (WHITE).
"""
from __future__ import annotations

from pyecharts import options as opts
from pyecharts.charts.base import Base

PALETTE = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
WHITE = "#FFFFFF"
BLACK = PALETTE["black"]
ALLOWED_COLORS = [*PALETTE.values(), WHITE]

WIDTH_PX = 1050
SINGLE_COLUMN_IN = 3.37
MIN_FONT_PX = 22
FS = 24            # tick labels, legend, axis names (9 pt at 0.9 x TMLR text width)
FS_TITLE = 26
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"   # Helvetica Neue has the superscript glyphs cairosvg needs

# Consistent encodings across figures
CLASS_COLOR = {"wide": PALETTE["blue"], "narrow": PALETTE["vermillion"],
               "nearbamboo": PALETTE["bluish_green"], "uniform": PALETTE["black"]}
CLASS_LABEL = {"wide": "wide 10 µm", "narrow": "narrow 3 µm", "nearbamboo": "near-bamboo",
               "uniform": "uniform"}
CU_SYMBOL = {0.0: "circle", 0.5: "triangle", 1.0: "rect", 2.0: "diamond"}
CU_LINE = {0.0: "solid", 0.5: "dashed", 1.0: "dashed", 2.0: "dotted"}
SITE_COLOR = {"cathode": PALETTE["orange"], "interior": PALETTE["sky_blue"],
              "immortal": PALETTE["black"]}

# Suffixes that mark a numeric figure-data column as unit-labeled (R49).
# Dimensionless quantities carry an explicit ratio or "_dimless"/"_frac"/"_count".
UNIT_SUFFIXES = ("_h", "_s", "_um", "_MPa", "_MAcm2", "_A_per_cm", "_wt", "_C", "_pct",
                 "_frac", "_count", "_dimless", "_over_L", "_over_c0", "_index")


def text(size: int = FS, **kw) -> dict:
    return {"color": BLACK, "fontSize": size, "fontFamily": FONT, **kw}


def axis(name: str, kind: str = "value", grid: int = 0, *, gap: int = 60, loc: str = "middle",
         label: dict | None = None, **kw) -> dict:
    """One axis, black lines, no grid lines, name with a unit in brackets."""
    a = {
        "type": kind, "gridIndex": grid, "name": name, "nameLocation": loc, "nameGap": gap,
        "nameTextStyle": text(), "axisLine": {"show": True, "onZero": False, "lineStyle": {"color": BLACK, "width": 2}},
        "axisTick": {"show": True, "inside": True, "lineStyle": {"color": BLACK, "width": 2}},
        "axisLabel": {**text(), **(label or {})},
        "splitLine": {"show": False},
    }
    if kind == "log":
        a["minorTick"] = {"show": True, "lineStyle": {"color": BLACK}}
        a["axisLabel"]["formatter"] = log_sup_formatter()
    a.update(kw)
    return a


def hidden_axis(kind: str = "value", grid: int = 0, **kw) -> dict:
    return {"type": kind, "gridIndex": grid, "show": False, **kw}


def title(txt: str, left, top, sub: str | None = None, size: int = FS_TITLE) -> dict:
    t = {"text": txt, "left": left, "top": top, "textStyle": text(size, fontWeight="normal")}
    if sub:
        t["subtext"] = sub
        t["subtextStyle"] = text(FS)
        t["itemGap"] = 4
    return t


def legend(data=None, **kw) -> dict:
    lg = {"show": True, "textStyle": text(), "itemWidth": 44, "itemHeight": 20,
          "inactiveColor": WHITE, "itemGap": 18, **kw}
    if data is not None:
        lg["data"] = data
    return lg


def base_opts(title_text: str, x_label_with_unit: str, y_label_with_unit: str,
              x_kind: str = "value", y_kind: str = "value") -> dict:
    """Single-grid option skeleton: title, one x and one y axis, palette, white background."""
    return {
        "backgroundColor": WHITE,
        "animation": False,
        "color": list(PALETTE.values()),
        "textStyle": text(),
        "title": [title(title_text, 20, 6)],
        "grid": [{"left": 170, "right": 40, "top": 110, "bottom": 150}],
        "xAxis": [axis(x_label_with_unit, x_kind)],
        "yAxis": [axis(y_label_with_unit, y_kind, gap=110)],
        "series": [],
    }


def skeleton() -> dict:
    """Empty multi-panel option with the shared global settings."""
    return {"backgroundColor": WHITE, "animation": False, "color": list(PALETTE.values()),
            "textStyle": text(), "title": [], "grid": [], "xAxis": [], "yAxis": [], "series": []}


def make_chart(option: dict, height_px: int) -> Base:
    """Wrap a raw ECharts option in a pyecharts chart of WIDTH_PX x height_px."""
    ch = Base(init_opts=opts.InitOpts(width=f"{WIDTH_PX}px", height=f"{height_px}px",
                                      bg_color=WHITE, renderer="canvas",
                                      animation_opts=opts.AnimationOpts(animation=False)))
    ch.options.update(option)
    ch.options["_emsk_width_px"] = WIDTH_PX
    ch.options["_emsk_height_px"] = height_px
    return ch


def fmt_num(v: float) -> str:
    """Compact label for a tick value (used in category labels)."""
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.3g}"
    return f"{v:.2g}"


def log_sup_formatter():
    """The single label formatter of every log axis: each decade as 10 with a
    superscript exponent (10⁰, 10¹, 10⁻²); values that are not decades get no label."""
    from pyecharts.commons.utils import JsCode
    return JsCode(
        "function(v){var e=Math.log10(v);if(Math.abs(e-Math.round(e))>1e-9){return '';}"
        "e=Math.round(e);"
        "var m={'-':'⁻','0':'⁰','1':'¹','2':'²','3':'³','4':'⁴',"
        "'5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'};"
        "return '10'+String(e).split('').map(function(c){return m[c];}).join('');}")


LEGEND_ICON = "roundRect"


def proxy(name: str, color: str, symbol: str = "rect", grid: int = 0, line: str | None = None) -> dict:
    """Empty series used only as a legend entry (color key or symbol key)."""
    s = {"type": "line" if line else "scatter", "name": name, "xAxisIndex": grid, "yAxisIndex": grid,
         "data": [], "symbol": symbol, "itemStyle": {"color": color, "opacity": 1}}
    if line:
        s["lineStyle"] = {"color": color, "type": line, "width": 4}
        s["showSymbol"] = False
    return s


def key_legends(colour_items, symbol_items, top: int, left: int = 40) -> tuple[list, list]:
    """Two-row legend: classes by color (row 1), Cu content by symbol (row 2).
    Returns (proxy series, legend list)."""
    series = [proxy(n, c) for n, c in colour_items]
    series += [proxy(n, BLACK, sym) for n, sym in symbol_items]
    legends = [legend([n for n, _ in colour_items], left=left, top=top, itemWidth=34, itemHeight=30),
               legend([n for n, _ in symbol_items], left=left, top=top + 56, itemWidth=34, itemHeight=30)]
    return series, legends
