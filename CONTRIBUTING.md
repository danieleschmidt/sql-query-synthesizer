# Contributing

Thank you for your interest in contributing! Please follow these steps:

1. **Set up environment**
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov ruff
   ```
2. **Run linters and tests** before committing:
   ```bash
   ruff .
   pytest --cov=sql_synthesizer --cov=query_agent.py
   ```
3. **Pull requests** should be linked to an issue and pass CI.

