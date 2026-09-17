# Render: approved report text -> everything the email needs.

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

import config                             # noqa: E402
from report.facts import load_facts       # noqa: E402

AVATAR_CID = "imidillo-avatar"
AVATAR_PATH = config.STATIC_DIR / "avatar-standing.png"
BRAND = "#7a003f"                         # same dark magenta as the v1 emails
GREY = "#b4b2a9"
FONT = "-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


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
    ax.tick_params(labelsize=8)


def chart_profiles(facts: dict) -> bytes | None:
    """Typical weekday occupancy by hour, one line per modelled lot."""
    lots = facts["parking_patterns"]["lots"]
    if not lots:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for f in lots.values():
        ys = [v if v is not None else float("nan") for v in f["profile_weekday_pct"]]
        ax.plot(range(24), ys, marker="o", markersize=2.5, linewidth=1.6, label=f["name"])
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_ylim(0, 100)
    ax.set_ylabel("occupancy %", fontsize=8)
    ax.set_title(f"Typical weekday, {facts['report']['month_label']} (local time)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    _tidy(ax)
    fig.tight_layout()
    return _png(fig)


def chart_comparison(facts: dict) -> bytes | None:
    """Mean occupancy, previous month vs report month, per compared lot."""
    comp = facts["parking_comparison"]
    lots = comp["lots"]
    if not lots:
        return None
    names = [textwrap.fill(v["name"], 16) for v in lots.values()]
    prev = [v["mean_occ_pct"]["previous"] for v in lots.values()]
    cur = [v["mean_occ_pct"]["current"] for v in lots.values()]
    sig = [v["adjusted_shift"]["significant"] for v in lots.values()]
    x = range(len(lots))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar([i - w / 2 for i in x], prev, w, color=GREY, label=comp["previous_month_label"])
    ax.bar([i + w / 2 for i in x], cur, w, color=BRAND, label=comp["month_label"])
    for i, s in enumerate(sig):
        if s:
            ax.text(i, max(prev[i], cur[i]) + 2, "*", ha="center", fontsize=12, color=BRAND)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylim(0, max(prev + cur) * 1.25 if prev + cur else 100)
    ax.set_ylabel("mean occupancy %", fontsize=8)
    ax.set_title("Mean occupancy by lot   (* = statistically significant change)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    _tidy(ax)
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
    xhtml2pdf cannot split a table row across pages, so the email layout
    produced a near-empty first page with everything on page two."""
    avatar = (f'<img src="cid:{AVATAR_CID}" alt="Imidillo" '
              f'style="width:52px;height:auto;margin-right:12px;">') if AVATAR_PATH.exists() else ""
    # xhtml2pdf turns the email's width:100% images into zero-width boxes;
    # give the charts an absolute width instead.
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

    body = markdown_to_html(report_md)
    body = re.sub(r"<h1[^>]*>.*?</h1>\s*", "", body, count=1, flags=re.DOTALL)  # title lives in the header

    inline: list[tuple[str, bytes, str]] = []
    if AVATAR_PATH.exists():
        inline.append((AVATAR_CID, AVATAR_PATH.read_bytes(), "image/png"))
    else:
        print(f"[RENDER] avatar not found at {AVATAR_PATH}, sending without it")

    profiles = chart_profiles(facts)
    if profiles:
        inline.append(("chart-profiles", profiles, "image/png"))
        body = _insert_before_heading(body, "Compared with", _img("chart-profiles", "Typical weekday occupancy by hour"))
    comparison = chart_comparison(facts)
    if comparison:
        inline.append(("chart-comparison", comparison, "image/png"))
        body = _insert_before_heading(body, "Data quality", _img("chart-comparison", "Mean occupancy, this month vs last"))

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