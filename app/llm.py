"""
Small shared helper around the LLM client.

Every agent needs the same things:
  * a client + model name for the active backend (Ollama or OpenAI),
  * a way to suppress qwen3's "thinking" tokens, and
  * ONE place that logs every LLM interaction so you can trace exactly what
    prompt was sent and what the model decided (enable with LOG_LEVEL=DEBUG).
"""

from __future__ import annotations

from app.config import get_chat_model, get_llm_client, settings
from app.logging_config import log


def nothink(system_prompt: str) -> str:
    """
    Append qwen3's `/no_think` directive when we're on a qwen model via Ollama.
    For any other model (e.g. llama3.2) this is a harmless no-op.
    """
    if settings.LLM_BACKEND == "ollama" and "qwen" in settings.OLLAMA_MODEL.lower():
        return system_prompt.rstrip() + " /no_think"
    return system_prompt


# One shared client/model for the whole app.
client = get_llm_client()
MODEL = get_chat_model()


def _log_request(agent: str, messages: list[dict], tools) -> None:
    """DEBUG: print the full prompt (and offered tools) sent to the model."""
    log.debug(f"[{agent}] ┌── LLM REQUEST (model={MODEL}) ──────────────")
    for m in messages:
        log.debug(f"[{agent}] │ {m['role'].upper()}:\n{m['content']}")
    if tools:
        names = [t["function"]["name"] for t in tools]
        log.debug(f"[{agent}] │ TOOLS OFFERED TO MODEL: {names}")
    log.debug(f"[{agent}] └────────────────────────────────────────────")


def _log_response(agent: str, msg) -> None:
    """DEBUG: print what the model decided — a tool call or a text reply."""
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            log.debug(f"[{agent}] ↳ MODEL CALLED TOOL: "
                      f"{tc.function.name}({tc.function.arguments})")
    else:
        log.debug(f"[{agent}] ↳ MODEL REPLIED: {msg.content}")


def chat(agent: str, messages: list[dict], tools=None, temperature: float = 0):
    """
    Make one chat completion, logging the request and response.

    `agent` is just a label for the logs (e.g. "DataAgent"). Returns the raw
    message object so callers can read `.tool_calls` or `.content`.
    """
    _log_request(agent, messages, tools)
    kwargs = {"model": MODEL, "messages": messages, "temperature": temperature}
    if tools:
        kwargs["tools"] = tools
    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    _log_response(agent, msg)
    return msg
