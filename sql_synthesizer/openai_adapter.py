"""Adapter for OpenAI completions used to generate SQL."""

from __future__ import annotations

try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional
    openai = None


class OpenAIAdapter:
    """Thin wrapper around the OpenAI chat completion API."""

    def __init__(
        self, api_key: str, model: str = "gpt-3.5-turbo", timeout: float | None = None
    ) -> None:
        if not openai:
            raise RuntimeError("openai package not available")
        openai.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate_sql(self, question: str) -> str:
        prompt = (
            "Translate the following natural language request into an SQL query:\n"
            f"{question}\nSQL:"
        )
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=self.timeout,
        )
        return response.choices[0].message["content"].strip()
