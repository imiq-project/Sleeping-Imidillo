
# door to the OpenAI Responses API, shared by writer and verifier.

from __future__ import annotations

import time

import openai

import config

_RETRIES = 3
_BACKOFF_S = 5.0
_client: openai.OpenAI | None = None


class LLMError(RuntimeError):
    """The model could not be reached or returned nothing usable."""


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        if not config.llm_configured():
            raise LLMError("OPENAI_API_KEY not set in .env")
        _client = openai.OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL,
                                timeout=config.LLM_TIMEOUT_S, max_retries=0)
    return _client


def complete(instructions: str, user_input: str, *, model: str, label: str = "llm",
             json_schema: dict | None = None, schema_name: str = "result",
             reasoning: str | None = None) -> str:
    """Return the model's text. With json_schema the output is guaranteed to be
    JSON matching the schema (strict mode); parse it with json.loads."""
    kwargs: dict = {"model": model, "instructions": instructions, "input": user_input}
    effort = reasoning or config.LLM_REASONING
    if effort and effort != "none":
        kwargs["reasoning"] = {"effort": effort}
    if json_schema:
        kwargs["text"] = {"format": {"type": "json_schema", "name": schema_name,
                                     "schema": json_schema, "strict": True}}

    last: Exception | None = None
    for attempt in range(1, _RETRIES + 1):
        t0 = time.perf_counter()
        try:
            r = _get_client().responses.create(**kwargs)
        except (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError) as e:
            last = e
        except openai.APIStatusError as e:
            if e.status_code >= 500:
                last = e
            else:
                raise LLMError(f"[{label}] {model}: HTTP {e.status_code}: {e.message}") from e
        else:
            text = (r.output_text or "").strip()
            u = r.usage
            print(f"[{label}] {model}: {u.input_tokens} in / {u.output_tokens} out, "
                  f"{time.perf_counter() - t0:.1f}s")
            if not text:
                raise LLMError(f"[{label}] {model} returned an empty response")
            return text
        if attempt < _RETRIES:
            print(f"[{label}] attempt {attempt} failed ({type(last).__name__}), retrying ...")
            time.sleep(_BACKOFF_S * attempt)
    raise LLMError(f"[{label}] {model} failed after {_RETRIES} attempts: {last}")