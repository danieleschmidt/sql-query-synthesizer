import types
import pytest

import sql_synthesizer.openai_adapter as oa


class FakeMessage:
    def __init__(self, content: str):
        self.content = content

class FakeChoice:
    def __init__(self, content: str):
        self.message = FakeMessage(content)


class FakeOpenAI:
    def __init__(self):
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self.create)
        )
        self.last_prompt = None
        self.last_timeout = None

    def create(self, model, messages, temperature, timeout=None):
        self.last_prompt = messages[0]["content"]
        self.last_timeout = timeout
        return types.SimpleNamespace(choices=[FakeChoice("SELECT 1;")])


def test_openai_adapter(monkeypatch):
    fake = FakeOpenAI()
    monkeypatch.setattr(oa, "openai", fake)
    adapter = oa.OpenAIAdapter(api_key="key", model="test", timeout=3)
    sql = adapter.generate_sql("hi")
    assert sql == "SELECT 1;"
    assert "hi" in fake.last_prompt
    assert fake.last_timeout == 3


def test_openai_adapter_with_tables(monkeypatch):
    """Test OpenAI adapter with available tables context."""
    fake = FakeOpenAI()
    monkeypatch.setattr(oa, "openai", fake)
    adapter = oa.OpenAIAdapter(api_key="key", model="test")
    
    sql = adapter.generate_sql("show users", available_tables=["users", "orders"])
    assert sql == "SELECT 1;"
    assert "Available tables: users, orders" in fake.last_prompt
    assert "safe SELECT queries" in fake.last_prompt


def test_openai_adapter_empty_question(monkeypatch):
    """Test OpenAI adapter rejects empty questions."""
    fake = FakeOpenAI()
    monkeypatch.setattr(oa, "openai", fake)
    adapter = oa.OpenAIAdapter(api_key="key")
    
    with pytest.raises(ValueError, match="Question cannot be empty"):
        adapter.generate_sql("")
    
    with pytest.raises(ValueError, match="Question cannot be empty"):
        adapter.generate_sql("   ")


def test_openai_adapter_no_openai(monkeypatch):
    """Test OpenAI adapter when openai package is not available."""
    # Mock openai to be None
    monkeypatch.setattr(oa, "openai", None)
    with pytest.raises(RuntimeError, match="openai package not available"):
        oa.OpenAIAdapter(api_key="key")  # pragma: allowlist secret
