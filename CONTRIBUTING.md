# Contributing

Thank you for your interest in contributing! Please follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Development Setup

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed setup instructions.

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Before Submitting

1. **Code Quality**: Run `ruff .` for linting
2. **Tests**: Execute `pytest --cov=sql_synthesizer --cov=query_agent.py`
3. **Security**: Review [SECURITY.md](SECURITY.md) guidelines

## Pull Request Process

* Link PRs to related issues
* Ensure CI checks pass
* Follow [conventional commits](https://www.conventionalcommits.org/)
* Update documentation as needed

For architecture decisions, see [ADR documentation](docs/adr/).

