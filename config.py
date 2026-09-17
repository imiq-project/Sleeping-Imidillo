from __future__ import annotations
import os
import re
from pathlib import Path
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        print(f"[CONFIG] WARNING: {name}={raw!r} is not an integer, using {default}")
        return default


def _env_list(name: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,;]", _env(name)) if p.strip()]

OUTPUT_DIR = PROJECT_ROOT / "output"         
STATIC_DIR = PROJECT_ROOT / "static"         
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"  
# Sensor timestamps arrive in UTC
LOCAL_TZ = _env("LOCAL_TZ", "Europe/Berlin")

#QuantumLeap (FIWARE historical data)                                              
ORION_BASE_URL = _env("ORION_BASE_URL", "https://imiq-public.et.uni-magdeburg.de/api/orion")

QL_BASE_URL = _env("QL_BASE_URL", "https://imiq-public.et.uni-magdeburg.de/api/quantumleap")
FIWARE_API_KEY = _env("FIWARE_API_KEY")      
FIWARE_SERVICE = _env("FIWARE_SERVICE")
FIWARE_SERVICE_PATH = _env("FIWARE_SERVICE_PATH")
QL_TIMEOUT_S = _env_int("QL_TIMEOUT_S", 60)

# LLM       
OPENAI_API_KEY = _env("OPENAI_API_KEY")
OPENAI_BASE_URL = _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
WRITER_MODEL = _env("WRITER_MODEL", "gpt-5.4")      
VERIFIER_MODEL = _env("VERIFIER_MODEL", "gpt-5.4") 
LLM_REASONING = _env("LLM_REASONING", "high")   
LLM_TIMEOUT_S = _env_int("LLM_TIMEOUT_S", 180)
# SMTP (the bot's own mailbox)                                         
SMTP_HOST = _env("SMTP_HOST")
SMTP_PORT = _env_int("SMTP_PORT", 587)   
SMTP_USERNAME = _env("SMTP_USERNAME")
SMTP_PASSWORD = _env("SMTP_PASSWORD")
SMTP_FROM_NAME = _env("SMTP_FROM_NAME", "Imidillo")

# Report                                                                    
REPORT_RECIPIENTS = _env_list("REPORT_EMAIL_TO")   #one or many addresses
REPORT_LANGUAGE = _env("REPORT_LANGUAGE", "en")   

def quantumleap_configured() -> bool:

    return bool(QL_BASE_URL)

def llm_configured() -> bool:
    return bool(OPENAI_API_KEY and WRITER_MODEL and VERIFIER_MODEL)


def smtp_configured() -> bool:
    return bool(SMTP_HOST and SMTP_USERNAME and SMTP_PASSWORD and REPORT_RECIPIENTS)


def missing() -> list[str]:
    """Names of the variables the full monthly job cannot run without."""
    required = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "SMTP_HOST": SMTP_HOST,
        "SMTP_USERNAME": SMTP_USERNAME,
        "SMTP_PASSWORD": SMTP_PASSWORD,
        "REPORT_EMAIL_TO": ",".join(REPORT_RECIPIENTS),
    }
    return [name for name, value in required.items() if not value]

def _mask(secret: str) -> str:
    if not secret:
        return "MISSING"
    return f"set ({secret[:3]}...{secret[-2:]}, {len(secret)} chars)" if len(secret) > 8 else "set"


def _print_summary() -> None:
    print(f"Project root : {PROJECT_ROOT}")
    print(f".env present : {(PROJECT_ROOT / '.env').exists()}")
    print(f"Local tz     : {LOCAL_TZ}")
    print()
    print("QuantumLeap")
    print(f"   base url  : {QL_BASE_URL}")
    print(f"   api key   : {_mask(FIWARE_API_KEY)}")
    print(f"   service   : {FIWARE_SERVICE or '(none)'} / {FIWARE_SERVICE_PATH or '(none)'}")
    print(f"   orion url : {ORION_BASE_URL}")
    print(f"   ready     : {quantumleap_configured()}")
    print()
    print("LLM")
    print(f"   base url  : {OPENAI_BASE_URL}")
    print(f"   api key   : {_mask(OPENAI_API_KEY)}")
    print(f"   writer    : {WRITER_MODEL}")
    print(f"   verifier  : {VERIFIER_MODEL}")
    print(f"   ready     : {llm_configured()}")
    print()
    print("SMTP")
    print(f"   host      : {SMTP_HOST or 'MISSING'}:{SMTP_PORT}")
    print(f"   username  : {SMTP_USERNAME or 'MISSING'}")
    print(f"   password  : {_mask(SMTP_PASSWORD)}")
    print(f"   from name : {SMTP_FROM_NAME}")
    print(f"   ready     : {smtp_configured()}")
    print()
    print("Report")
    print(f"   recipients: {', '.join(REPORT_RECIPIENTS) or 'MISSING'}")
    print(f"   language  : {REPORT_LANGUAGE}")
    print(f"   output dir: {OUTPUT_DIR}")
    print()
    gaps = missing()
    if gaps:
        print(f"MISSING for the full monthly job: {', '.join(gaps)}")
    else:
        print("All required settings present.")

if __name__ == "__main__":
    _print_summary()
    raise SystemExit(1 if missing() else 0)