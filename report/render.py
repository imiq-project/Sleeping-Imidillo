"""
Render: approved report text -> everything the email needs.

  charts      PNGs drawn from facts (never from the text): one figure per
              data family (typical weekday profile + this month vs last),
              and one 2x2 figure for the air-quality pollutants
  html        the Markdown converted to email-safe HTML: inline styles only,
              a 600px table layout, avatar and charts embedded by CID (the
              one image mechanism Gmail and Outlook both show without a
              "load images" prompt)
  text        the Markdown itself, as the plain-text alternative
  pdf         the same content through xhtml2pdf with a flowing (table-free)
              layout, images embedded as base64

render() returns a Rendered object; the mailer only has to send it.

Run manually:
    python -m report.render 2026-08     # needs output/2026-08/{facts.json,report.md}
Writes report.html, report.pdf and chart_*.png next to them; open report.html
in a browser to check the layout.
"""

from __future__ import annotations

import base64
import io
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import markdown
import matplotlib

matplotlib.use("Agg")                     # no display on the server
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402

import config                             # noqa: E402
from report.facts import load_facts       # noqa: E402

AVATAR_CID = "imidillo-avatar"
AVATAR_PATH = config.STATIC_DIR / "avatar-standing.png"
BRAND = "#7a003f"                         # same dark magenta as the v1 emails
GREY = "#b4b2a9"
FONT = "-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MAX_LINES = 6                             # more series than this -> grey lines + median
MAX_BARS = 6                              # comparison bars: significant changes first


@dataclass
class Rendered:
    subject: str
    html: str
    text: str
    inline_images: list[tuple[str, bytes, str]] = field(default_factory=list)   # (cid, bytes, mime)
    attachments: list[tuple[str, bytes, str]] = field(default_factory=list)     # (filename, bytes, mime)


# --------------------------------------------------------------------------- #
# Charts (from facts, never from the prose)                                    #
# --------------------------------------------------------------------------- #

def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def _tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=7)


def _profile_panel(ax, fam: dict, title: str) -> bool:
    """Typical weekday profile of every modelled series. Returns False if nothing to draw."""
    series = fam["patterns"]["series"]
    if not series:
        ax.text(0.5, 0.5, "no usable series", ha="center", va="center", fontsize=9, color="#888",
                transform=ax.transAxes)
        ax.set_axis_off()
        return False
    hours = range(24)
    profiles = {k: [v if v is not None else np.nan for v in s["profile_weekday"]] for k, s in series.items()}
    if len(profiles) <= MAX_LINES:
        for k, ys in profiles.items():
            ax.plot(hours, ys, marker="o", markersize=2, linewidth=1.4, label=series[k]["name"])
        ax.legend(fontsize=6, frameon=False)
    else:
        for ys in profiles.values():
            ax.plot(hours, ys, color=GREY, linewidth=0.8, alpha=0.7)
        stack = np.array(list(profiles.values()), dtype=float)
        # median only where at least half of the sensors have a value; a lone
        # night-time reading would otherwise draw a spike nobody measured
        need = max(2, len(profiles) // 2)
        median = np.array([np.nanmedian(col) if np.isfinite(col).sum() >= need else np.nan for col in stack.T])
        ax.plot(hours, median, color=BRAND, linewidth=2.2, label=f"median of {len(profiles)} sensors")
        ax.legend(fontsize=6, frameon=False)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 3)])
    ax.set_ylabel(f"{fam['variable']} ({fam['unit']})", fontsize=7)
    ax.set_title(title, fontsize=9)
    if fam["unit"] == "%":
        ax.set_ylim(0, 100)
    _tidy(ax)
    return True


def _comparison_panel(ax, fam: dict, title: str) -> bool:
    """Mean per series, previous vs current month. Up to MAX_BARS series:
    significant changes first, then the largest changes."""
    comp = fam["comparison"]
    rows = list(comp["series"].items())
    if not rows:
        ax.text(0.5, 0.5, "nothing to compare", ha="center", va="center", fontsize=9, color="#888",
                transform=ax.transAxes)
        ax.set_axis_off()
        return False
    rows.sort(key=lambda kv: (not kv[1]["adjusted_shift"]["significant"], -abs(kv[1]["adjusted_shift"]["delta"])))
    rows = rows[:MAX_BARS]
    names = [textwrap.fill(v["name"], 12) for _, v in rows]
    prev = [v["mean"]["previous"] for _, v in rows]
    cur = [v["mean"]["current"] for _, v in rows]
    x = np.arange(len(rows))
    w = 0.38
    ax.bar(x - w / 2, prev, w, color=GREY, label=comp["previous_month_label"])
    ax.bar(x + w / 2, cur, w, color=BRAND, label=comp["month_label"])
    for i, (_, v) in enumerate(rows):
        if v["adjusted_shift"]["significant"]:
            ax.text(i, max(prev[i], cur[i]) * 1.02, "*", ha="center", fontsize=11, color=BRAND)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6, rotation=25 if len(rows) > 4 else 0,
                       ha="right" if len(rows) > 4 else "center")
    top = max(prev + cur) if prev + cur else 1
    ax.set_ylim(0, top * 1.25)
    ax.set_ylabel(f"mean {fam['variable']} ({fam['unit']})", fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, frameon=False)
    _tidy(ax)
    return True


def chart_family(fam: dict) -> bytes | None:
    """Two panels: typical weekday profile, and this month vs last."""
    if not fam["patterns"]["series"] and not fam["comparison"]["series"]:
        return None
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    _profile_panel(left, fam, f"{fam['label']}: typical weekday ({fam['patterns']['month_label']})")
    _comparison_panel(right, fam, "Mean by sensor  (* = significant change)")
    fig.tight_layout()
    return _png(fig)


def chart_air(families: dict) -> bytes | None:
    """One panel per pollutant that has modelled stations."""
    air = [(k, f) for k, f in families.items() if k.startswith("air_") and f["patterns"]["series"]]
    if not air:
        return None
    cols = 2 if len(air) > 1 else 1
    rows = (len(air) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 2.9 * rows), squeeze=False)
    for ax, (k, fam) in zip(axes.flat, air):
        _profile_panel(ax, fam, f"{fam['label']}: typical weekday")
    for ax in list(axes.flat)[len(air):]:
        ax.set_axis_off()
    fig.tight_layout()
    return _png(fig)


# --------------------------------------------------------------------------- #
# Markdown -> email HTML                                                       #
# --------------------------------------------------------------------------- #

_STYLES = {
    "h1": f"font-family:{FONT};font-size:22px;font-weight:600;color:{BRAND};margin:0 0 4px 0;",
    "h2": f"font-family:{FONT};font-size:16px;font-weight:600;color:{BRAND};margin:22px 0 6px 0;"
          "padding-bottom:3px;border-bottom:1px solid #eee;",
    "p": f"font-family:{FONT};font-size:14px;line-height:1.55;color:#222;margin:0 0 10px 0;",
    "ul": "margin:0 0 10px 20px;padding:0;",
    "li": f"font-family:{FONT};font-size:14px;line-height:1.5;color:#222;margin:0 0 5px 0;",
}


def markdown_to_html(md: str) -> str:
    html = markdown.markdown(md)
    for tag, style in _STYLES.items():
        html = html.replace(f"<{tag}>", f'<{tag} style="{style}">')
    return html


def _img(cid: str, alt: str) -> str:
    return (f'<p style="margin:12px 0 16px 0;"><img src="cid:{cid}" alt="{alt}" width="600" '
            f'style="display:block;width:100%;max-width:600px;height:auto;border:1px solid #eee;"></p>')


def _insert_before_heading(html: str, heading_starts_with: str, block: str) -> str:
    m = re.search(rf"<h2[^>]*>{re.escape(heading_starts_with)}", html)
    return html[:m.start()] + block + html[m.start():] if m else html + block


def _wrap(title: str, body: str, generated_at: str) -> str:
    avatar = (f'<td style="vertical-align:middle;padding-right:14px;">'
              f'<img src="cid:{AVATAR_CID}" alt="Imidillo" width="64" '
              f'style="display:block;width:64px;height:auto;"></td>') if AVATAR_PATH.exists() else ""
    return f"""\
<div style="background:#f6f6f6;padding:20px 0;">
<table role="presentation" cellpadding="0" cellspacing="0" width="600"
       style="max-width:600px;margin:0 auto;background:#ffffff;border:1px solid #e6e6e6;border-radius:8px;">
  <tr><td style="padding:24px 28px 8px 28px;">
    <table role="presentation" cellpadding="0" cellspacing="0"><tr>
      {avatar}
      <td style="vertical-align:middle;">
        <h1 style="{_STYLES['h1']}">{title}</h1>
        <p style="{_STYLES['p']}color:#666;margin:0;">Magdeburg city report &middot; IMIQ project</p>
      </td>
    </tr></table>
  </td></tr>
  <tr><td style="padding:8px 28px 24px 28px;">
    {body}
    <p style="{_STYLES['p']}font-size:11px;color:#888;margin-top:24px;">
      Generated automatically on {generated_at}. Every figure in this email was checked
      against the sensor data before sending; nothing was added by hand.
    </p>
  </td></tr>
</table>
</div>"""


# --------------------------------------------------------------------------- #
# PDF                                                                          #
# --------------------------------------------------------------------------- #

def _wrap_pdf(title: str, body: str, generated_at: str) -> str:
    """PDF layout: the same body, but flowing divs instead of the email table.
    xhtml2pdf cannot split a table row across pages, and turns width:100%
    images into zero-width boxes, so charts get an absolute width."""
    avatar = (f'<img src="cid:{AVATAR_CID}" alt="Imidillo" '
              f'style="width:52px;height:auto;margin-right:12px;">') if AVATAR_PATH.exists() else ""
    body = re.sub(r'<p[^>]*><img src="cid:(chart-[^"]+)"[^>]*></p>',
                  r'<p style="margin:8pt 0 12pt 0;"><img src="cid:\1" style="width:16cm;"></p>', body)
    return f"""\
<html><head><style>
  @page {{ size: A4; margin: 1.8cm 1.8cm 2cm 1.8cm; }}
  body {{ font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; color: #222; }}
</style></head><body>
<table style="width:100%;border-bottom:1px solid #ddd;margin-bottom:10pt;"><tr>
  <td style="width:64px;vertical-align:middle;">{avatar}</td>
  <td style="vertical-align:middle;">
    <h1 style="{_STYLES['h1']}">{title}</h1>
    <p style="{_STYLES['p']}color:#666;margin:0;">Magdeburg city report &middot; IMIQ project</p>
  </td>
</tr></table>
{body}
<p style="{_STYLES['p']}font-size:8.5pt;color:#888;margin-top:18pt;">
  Generated automatically on {generated_at}. Every figure in this report was checked
  against the sensor data before sending; nothing was added by hand.
</p>
</body></html>"""


def build_pdf(html: str, inline_images: list[tuple[str, bytes, str]]) -> bytes | None:
    """HTML from _wrap_pdf with images embedded as base64. None if xhtml2pdf is
    unavailable or fails; the email still goes out without the attachment."""
    try:
        from xhtml2pdf import pisa
    except ImportError:
        print("[RENDER] xhtml2pdf not installed, skipping PDF")
        return None
    for cid, data, mime in inline_images:
        html = html.replace(f"cid:{cid}", f"data:{mime};base64,{base64.b64encode(data).decode()}")
    out = io.BytesIO()
    result = pisa.CreatePDF(html, dest=out, encoding="utf-8")
    if result.err:
        print(f"[RENDER] PDF generation reported {result.err} error(s), skipping PDF")
        return None
    return out.getvalue()


# --------------------------------------------------------------------------- #
# The one function the monthly job calls                                       #
# --------------------------------------------------------------------------- #

def render(facts: dict, report_md: str, with_pdf: bool = True) -> Rendered:
    r = facts["report"]
    subject = f"Imidillo report, {r['month_label']}"
    fams = facts["families"]

    body = markdown_to_html(report_md)
    body = re.sub(r"<h1[^>]*>.*?</h1>\s*", "", body, count=1, flags=re.DOTALL)  # title lives in the header

    inline: list[tuple[str, bytes, str]] = []
    if AVATAR_PATH.exists():
        inline.append((AVATAR_CID, AVATAR_PATH.read_bytes(), "image/png"))
    else:
        print(f"[RENDER] avatar not found at {AVATAR_PATH}, sending without it")

    # one chart per family, placed at the end of that family's section
    placements = [("parking", "chart-parking", "Traffic"), ("traffic", "chart-traffic", "Air quality")]
    for key, cid, next_heading in placements:
        if key in fams:
            png = chart_family(fams[key])
            if png:
                inline.append((cid, png, "image/png"))
                body = _insert_before_heading(body, next_heading, _img(cid, f"{fams[key]['label']} charts"))
    air_png = chart_air(fams)
    if air_png:
        inline.append(("chart-air", air_png, "image/png"))
        body = _insert_before_heading(body, "Data quality", _img("chart-air", "Air quality daily profiles"))

    html = _wrap(subject, body, r["generated_at"])
    attachments: list[tuple[str, bytes, str]] = []
    if with_pdf:
        pdf = build_pdf(_wrap_pdf(subject, body, r["generated_at"]), inline)
        if pdf:
            attachments.append((f"imidillo-report-{r['month']}.pdf", pdf, "application/pdf"))

    return Rendered(subject=subject, html=html, text=report_md,
                    inline_images=inline, attachments=attachments)


def save_rendered(rendered: Rendered, month: str) -> Path:
    out_dir = config.OUTPUT_DIR / month
    out_dir.mkdir(parents=True, exist_ok=True)
    preview = rendered.html
    for cid, data, mime in rendered.inline_images:          # browser preview needs real images
        preview = preview.replace(f"cid:{cid}", f"data:{mime};base64,{base64.b64encode(data).decode()}")
    (out_dir / "report.html").write_text(preview, encoding="utf-8")
    for cid, data, _ in rendered.inline_images:
        if cid.startswith("chart-"):
            (out_dir / f"{cid.replace('-', '_')}.png").write_bytes(data)
    for filename, data, _ in rendered.attachments:
        (out_dir / filename).write_bytes(data)
    return out_dir / "report.html"


# --------------------------------------------------------------------------- #
# Manual check                                                                 #
# --------------------------------------------------------------------------- #

def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m report.render <YYYY-MM>")
        return 1
    month = sys.argv[1]
    facts = load_facts(month)
    report_md = (config.OUTPUT_DIR / month / "report.md").read_text(encoding="utf-8")
    rendered = render(facts, report_md)
    path = save_rendered(rendered, month)
    print(f"[RENDER] subject      : {rendered.subject}")
    print(f"[RENDER] html         : {len(rendered.html) / 1024:.1f} KB")
    print(f"[RENDER] inline images: {[(cid, f'{len(b) / 1024:.0f} KB') for cid, b, _ in rendered.inline_images]}")
    print(f"[RENDER] attachments  : {[(n, f'{len(b) / 1024:.0f} KB') for n, b, _ in rendered.attachments]}")
    print(f"[RENDER] preview      : {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
