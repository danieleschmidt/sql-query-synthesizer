import types

import sql_synthesizer.openai_adapter as oa


class FakeChoice:
    def __init__(self, content: str):
        self.message = {"content": content}


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
