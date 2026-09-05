"""
Small shared helper around the LLM client.

Every agent needs the same two things:
  * a client + model name for the active backend (Ollama or OpenAI), and
  * a way to suppress qwen3's "thinking" tokens so we get clean output.

Keeping this in one place means the agents stay focused on their own logic.
"""

from __future__ import annotations

from app.config import get_chat_model, get_llm_client, settings


def nothink(system_prompt: str) -> str:
    """
    Append qwen3's `/no_think` directive when we're on a qwen model via Ollama.

    qwen3 is a "thinking" model: without this it emits reasoning tokens that
    pollute structured output. For any other model this is a harmless no-op, so
    we only add it when it actually applies.
    """
    if settings.LLM_BACKEND == "ollama" and "qwen" in settings.OLLAMA_MODEL.lower():
        return system_prompt.rstrip() + " /no_think"
    return system_prompt


# One shared client/model for the whole app (cheap to reuse).
client = get_llm_client()
MODEL = get_chat_model()
