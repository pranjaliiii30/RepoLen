"""Gemini generation with context-only answering and bounded retries."""

import logging
import os
import time
from collections.abc import Iterator
from typing import Any

from google import genai
from google.genai import types

from core.models import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a repository code assistant. Answer ONLY from the provided indexed code context.
Cite every factual claim using the exact format [file_path:start_line-end_line].
Never invent files, symbols, behavior, or citations. If the context is insufficient, say exactly:
"I couldn't find this in the indexed code"
You may then explain what the provided context does and what additional code would be needed.
Keep explanations clear and developer-friendly. If context supports no useful answer, return only the required insufficient-context sentence."""


def create_gemini_client(api_key: str | None = None) -> Any:
    """Create a Gemini API client from an explicit key or environment variable."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Set GEMINI_API_KEY in the environment or .env file to use Gemini.")
    return genai.Client(api_key=key)


def _is_retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if status in (408, 429, 500, 502, 503, 504):
        return True
    text = f"{type(error).__name__} {error}".lower()
    return any(term in text for term in ("timeout", "temporarily unavailable", "resource_exhausted"))


def stream_answer(
    client: Any,
    model: str,
    question: str,
    sources: list[RetrievedChunk],
    temperature: float = 0.2,
    retries: int = 3,
) -> Iterator[str]:
    """Stream a grounded answer, retrying transient failures only before output."""
    context = "\n\n".join(
        f"--- SOURCE {item.chunk.path}:{item.chunk.start_line}-{item.chunk.end_line} "
        f"({item.chunk.symbol}) ---\n{item.chunk.content}"
        for item in sources
    )
    prompt = f"Question:\n{question}\n\nIndexed code context:\n{context or '(No chunks were retrieved.)'}"
    for attempt in range(retries + 1):
        emitted = False
        try:
            response_stream = client.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=temperature,
                ),
            )
            for response in response_stream:
                text = getattr(response, "text", None)
                if text:
                    emitted = True
                    yield text
            return
        except Exception as error:
            if emitted or attempt >= retries or not _is_retryable(error):
                logger.exception("Gemini response generation failed")
                raise RuntimeError(
                    "Gemini could not complete the response. Check the API key, quota, "
                    "network, and selected model, then try again."
                ) from error
            delay = min(2**attempt, 8)
            logger.warning("Retryable Gemini error; retrying in %s seconds", delay)
            time.sleep(delay)


def generate_text(
    client: Any,
    model: str,
    prompt: str,
    system_instruction: str,
    temperature: float = 0.2,
    retries: int = 3,
) -> str:
    """Generate a non-streaming text result with transient-error retries."""
    for attempt in range(retries + 1):
        try:
            result = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                ),
            )
            return result.text or ""
        except Exception as error:
            if attempt >= retries or not _is_retryable(error):
                logger.exception("Gemini text generation failed")
                raise RuntimeError(
                    "Gemini could not generate this content. Check the API key, quota, "
                    "network, and selected model, then try again."
                ) from error
            delay = min(2**attempt, 8)
            logger.warning("Retryable Gemini error; retrying in %s seconds", delay)
            time.sleep(delay)

    raise RuntimeError("Gemini generation stopped unexpectedly.")


def rewrite_query(client: Any, model: str, question: str) -> list[str]:
    """Ask Gemini for concise code-search variants of a developer question."""
    result = generate_text(
        client,
        model,
        f"Create two concise repository-code search queries for this question. "
        f"Return only the two queries, one per line.\nQuestion: {question}",
        "Rewrite the question for code retrieval. Do not answer it.",
        temperature=0,
        retries=1,
    )
    return [line.strip("- •\t ") for line in result.splitlines() if line.strip()][:2]


def generate_followups(
    client: Any, model: str, question: str, answer: str
) -> list[str]:
    """Generate three short, relevant follow-up questions."""
    result = generate_text(
        client,
        model,
        f"Suggest exactly three short developer follow-up questions based on this exchange. "
        f"Return one question per line, without numbering.\nQuestion: {question}\nAnswer: {answer}",
        "Generate follow-up questions only. Do not answer them.",
        temperature=0.4,
        retries=1,
    )
    suggestions = [line.strip("- •\t ") for line in result.splitlines() if line.strip()][:3]
    defaults = [
        "Which tests cover this behavior?",
        "What other files are involved?",
        "Where is this called from?",
    ]
    for suggestion in defaults:
        if len(suggestions) == 3:
            break
        if suggestion not in suggestions:
            suggestions.append(suggestion)
    return suggestions
