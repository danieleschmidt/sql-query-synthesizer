"""Adapter for OpenAI completions used to generate SQL."""

from __future__ import annotations

import time

try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional
    openai = None

from . import metrics
from .user_experience import (
    create_openai_package_missing_error,
    create_empty_question_error,
)


class OpenAIAdapter:
    """Thin wrapper around the OpenAI chat completion API."""

    def __init__(
        self, api_key: str, model: str = "gpt-3.5-turbo", timeout: float | None = None
    ) -> None:
        if not openai:
            raise create_openai_package_missing_error()
        openai.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate_sql(self, question: str, available_tables: list[str] | None = None) -> str:
        """Generate SQL with proper schema context and safety constraints."""
        # Sanitize input question
        question = question.strip()
        if not question:
            raise create_empty_question_error()
        
        # Build secure prompt with table context
        table_context = ""
        if available_tables:
            table_list = ", ".join(available_tables)
            table_context = f"\nAvailable tables: {table_list}\n"
        
        prompt = (
            "You are a SQL assistant. Generate ONLY safe SELECT queries based on the following constraints:\n"
            "- Use only SELECT statements\n"
            "- Use only the provided table names\n"
            "- Do not use multiple statements or semicolons\n"
            "- Do not use DROP, DELETE, UPDATE, INSERT, or other destructive operations\n"
            f"{table_context}"
            f"Translate this natural language request into a single SQL SELECT query:\n"
            f"{question}\n\n"
            "Return only the SQL query without any explanation or additional text:\n"
        )
        
        start_time = time.time()
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=self.timeout,
            )
            duration = time.time() - start_time
            metrics.record_openai_request(duration, "success")
            return response.choices[0].message.content.strip()
        except Exception as e:
            duration = time.time() - start_time
            metrics.record_openai_request(duration, "error")
            raise
