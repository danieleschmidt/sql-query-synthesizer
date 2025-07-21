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
    
    with pytest.raises(ValueError, match="Please provide a question"):
        adapter.generate_sql("")
    
    with pytest.raises(ValueError, match="Please provide a question"):
        adapter.generate_sql("   ")


def test_openai_adapter_no_openai(monkeypatch):
    """Test OpenAI adapter when openai package is not available."""
    # Mock openai to be None
    monkeypatch.setattr(oa, "openai", None)
    with pytest.raises(ValueError, match="OpenAI package.*install"):
        oa.OpenAIAdapter(api_key="key")  # pragma: allowlist secret


def test_openai_adapter_circuit_breaker_integration(monkeypatch):
    """Test OpenAI adapter integrates with circuit breaker correctly."""
    fake = FakeOpenAI()
    monkeypatch.setattr(oa, "openai", fake)
    
    adapter = oa.OpenAIAdapter(
        api_key="key", 
        model="test",
        circuit_breaker_failure_threshold=2,
        circuit_breaker_recovery_timeout=30
    )
    
    # Verify circuit breaker is initialized
    assert adapter.circuit_breaker is not None
    status = adapter.get_circuit_breaker_status()
    assert status['provider_name'] == 'openai'
    assert status['failure_threshold'] == 2
    assert status['recovery_timeout'] == 30


def test_openai_adapter_circuit_breaker_blocks_after_failures(monkeypatch):
    """Test circuit breaker blocks requests after repeated failures."""
    class FailingOpenAI:
        def __init__(self):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self.create)
            )
        
        def create(self, model, messages, temperature, timeout=None):
            raise Exception("API Error")
    
    failing_fake = FailingOpenAI()
    monkeypatch.setattr(oa, "openai", failing_fake)
    
    adapter = oa.OpenAIAdapter(
        api_key="key",
        circuit_breaker_failure_threshold=2,
        circuit_breaker_recovery_timeout=60
    )
    
    # First failure
    with pytest.raises(oa.ProviderError):
        adapter.generate_sql("test")
    
    # Second failure - should open circuit
    with pytest.raises(oa.ProviderError):
        adapter.generate_sql("test")
    
    # Third request should be blocked by circuit breaker
    with pytest.raises(oa.ProviderError, match="circuit breaker open"):
        adapter.generate_sql("test")


def test_openai_adapter_circuit_breaker_recovery(monkeypatch):
    """Test circuit breaker allows recovery after timeout."""
    import time
    
    class ConditionalOpenAI:
        def __init__(self):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self.create)
            )
            self.should_fail = True
        
        def create(self, model, messages, temperature, timeout=None):
            if self.should_fail:
                raise Exception("API Error")
            return types.SimpleNamespace(choices=[FakeChoice("SELECT 1;")])
    
    conditional_fake = ConditionalOpenAI()
    monkeypatch.setattr(oa, "openai", conditional_fake)
    
    with monkeypatch.context() as m:
        # Mock time to control circuit breaker timing
        mock_time = 1000.0
        m.setattr(time, "time", lambda: mock_time)
        
        adapter = oa.OpenAIAdapter(
            api_key="key",
            circuit_breaker_failure_threshold=2,
            circuit_breaker_recovery_timeout=30
        )
        
        # Cause failures to open circuit
        with pytest.raises(oa.ProviderError):
            adapter.generate_sql("test")
        with pytest.raises(oa.ProviderError):
            adapter.generate_sql("test")
        
        # Should be blocked
        with pytest.raises(oa.ProviderError, match="circuit breaker open"):
            adapter.generate_sql("test")
        
        # Advance time past recovery timeout
        mock_time = 1035.0  # 35 seconds later
        
        # Fix the API
        conditional_fake.should_fail = False
        
        # Should now succeed and close circuit
        result = adapter.generate_sql("test")
        assert result == "SELECT 1;"
        
        # Circuit should be closed now
        status = adapter.get_circuit_breaker_status()
        assert status['state'] == 'closed'


def test_openai_adapter_get_capabilities_includes_circuit_breaker(monkeypatch):
    """Test get_capabilities includes circuit breaker status."""
    fake = FakeOpenAI()
    monkeypatch.setattr(oa, "openai", fake)
    
    adapter = oa.OpenAIAdapter(api_key="key", model="test")
    capabilities = adapter.get_capabilities()
    
    assert 'circuit_breaker' in capabilities
    assert capabilities['circuit_breaker']['provider_name'] == 'openai'
    assert 'state' in capabilities['circuit_breaker']
