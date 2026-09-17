# SMTP mailer: Imidillo sends from ITS OWN mailbox (config.SMTP_USERNAME).

from __future__ import annotations

import argparse
import smtplib
import socket
import ssl
import sys
from datetime import datetime
from email.message import EmailMessage
from email.utils import formataddr

import config

_SMTP_TIMEOUT_S = 30.0


class MailerError(RuntimeError):
    """SMTP failure with a message safe to log (never carries the password)."""


def mailer_available() -> bool:
    return config.smtp_configured()


# --------------------------------------------------------------------------- #
# Build                                                                        #
# --------------------------------------------------------------------------- #

def build_message(subject: str, html_body: str, text_body: str | None,
                  to: list[str],
                  attachments: list[tuple[str, bytes, str]] | None = None,
                  inline_images: list[tuple[str, bytes, str]] | None = None) -> EmailMessage:
    """Assemble the MIME message. Pure: no network, no config checks beyond From."""
    if not to:
        raise MailerError("no recipients")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr((config.SMTP_FROM_NAME or "Imidillo", config.SMTP_USERNAME))
    msg["To"] = ", ".join(to)                     # one message, everyone visible in To
    msg.set_content(text_body or html_body)       # plain-text part first
    msg.add_alternative(html_body, subtype="html")

    if inline_images:
        # Attach onto the HTML alternative so it becomes multipart/related and
        # mail clients pair each <img src="cid:..."> with its image part.
        html_part = msg.get_payload()[-1]
        for cid, content, mime_type in inline_images:
            maintype, _, subtype = (mime_type or "image/png").partition("/")
            html_part.add_related(content, maintype=maintype, subtype=subtype or "png",
                                  cid=f"<{cid}>")

    for filename, content, mime_type in attachments or []:
        maintype, _, subtype = (mime_type or "application/octet-stream").partition("/")
        msg.add_attachment(content, maintype=maintype, subtype=subtype or "octet-stream",
                           filename=filename)
    return msg


# --------------------------------------------------------------------------- #
# Send                                                                         #
# --------------------------------------------------------------------------- #

def send_message(msg: EmailMessage, to: list[str]) -> list[str]:
    """Deliver via SMTP. Returns the addresses accepted by the server."""
    if not mailer_available():
        raise MailerError("SMTP not configured: set SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD "
                          "and REPORT_EMAIL_TO in .env")
    context = ssl.create_default_context()
    try:
        if int(config.SMTP_PORT) == 465:
            with smtplib.SMTP_SSL(config.SMTP_HOST, config.SMTP_PORT,
                                  timeout=_SMTP_TIMEOUT_S, context=context) as server:
                server.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
                refused = server.send_message(msg, to_addrs=to)
        else:
            with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=_SMTP_TIMEOUT_S) as server:
                server.starttls(context=context)
                server.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
                refused = server.send_message(msg, to_addrs=to)
    except smtplib.SMTPRecipientsRefused as e:
        raise MailerError(f"ALL recipients refused: {', '.join(e.recipients)}. "
                          "Check the addresses in REPORT_EMAIL_TO.") from e
    except smtplib.SMTPAuthenticationError as e:
        raise MailerError(f"SMTP login rejected for {config.SMTP_USERNAME!r} (code {e.smtp_code}). "
                          "For Gmail this must be an app password, not the account password.") from e
    except (smtplib.SMTPException, OSError) as e:
        raise MailerError(f"SMTP send failed: {type(e).__name__}: {e}") from e

    # smtplib treats a partial refusal as success; surface it loudly.
    if refused:
        detail = ", ".join(f"{addr} (code {code})" for addr, (code, _) in refused.items())
        print(f"[MAILER] WARNING: {len(refused)} of {len(to)} recipient(s) refused: {detail}")
    delivered = [r for r in to if r not in refused]
    print(f"[MAILER] sent {msg['Subject']!r} to {', '.join(delivered)}")
    return delivered


def send_email(subject: str, html_body: str, text_body: str | None = None,
               to: list[str] | None = None,
               attachments: list[tuple[str, bytes, str]] | None = None,
               inline_images: list[tuple[str, bytes, str]] | None = None) -> list[str]:
    recipients = list(to) if to else list(config.REPORT_RECIPIENTS)
    msg = build_message(subject, html_body, text_body, recipients, attachments, inline_images)
    return send_message(msg, recipients)


def send_rendered(rendered, to: list[str] | None = None) -> list[str]:
    """Send a report/render.py Rendered object. Defaults to REPORT_RECIPIENTS."""
    return send_email(rendered.subject, rendered.html, rendered.text, to=to,
                      attachments=rendered.attachments, inline_images=rendered.inline_images)


# --------------------------------------------------------------------------- #
# Manual checks                                                                #
# --------------------------------------------------------------------------- #

def _test_message() -> tuple[str, str, str]:
    now = datetime.now().strftime("%A, %d %B %Y, %H:%M")
    host = socket.gethostname()
    subject = f"Imidillo mailer test, {now}"
    text = (f"Hello,\n\nthis is a delivery test from Sleeping Imidillo, sent on {now} "
            f"from '{host}'.\n\nIf you can read this, SMTP works.\n\nImidillo")
    html = (f'<div style="font-family:Segoe UI,Roboto,sans-serif;max-width:560px">'
            f'<h2 style="color:#7a003f">Imidillo mailer test</h2>'
            f'<p>Sent on <strong>{now}</strong> from <code>{host}</code>.</p>'
            f'<p>If you can read this, SMTP works.</p><p>Imidillo</p></div>')
    return subject, html, text


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a test email or a rendered report.")
    parser.add_argument("--test", action="store_true", help="send a stamped test email")
    parser.add_argument("--send", metavar="YYYY-MM", help="send the rendered report from output/<month>/")
    parser.add_argument("--to", help="recipient(s), comma-separated; default: the bot's own mailbox")
    args = parser.parse_args()

    if not mailer_available():
        print("[MAILER] SMTP not configured (see .env.example)")
        return 2
    to = [a.strip() for a in args.to.split(",") if a.strip()] if args.to else [config.SMTP_USERNAME]

    try:
        if args.test:
            subject, html, text = _test_message()
            send_email(subject, html, text, to=to)
        elif args.send:
            from report.facts import load_facts
            from report.render import render
            facts = load_facts(args.send)
            report_md = (config.OUTPUT_DIR / args.send / "report.md").read_text(encoding="utf-8")
            send_rendered(render(facts, report_md), to=to)
        else:
            parser.print_help()
            return 1
    except MailerError as e:
        print(f"[MAILER] {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())