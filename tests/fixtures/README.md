# Test Fixtures

This directory contains test fixtures and sample data for the SQL Query Synthesizer test suite.

## Structure

- `sample_queries.json` - Sample natural language queries and expected SQL outputs
- `database_schemas/` - Sample database schemas for testing schema introspection
- `mock_responses/` - Mock responses from external services (OpenAI, etc.)
- `test_data/` - Sample data sets for testing query execution

## Usage

Test fixtures are automatically loaded by the test framework and available for use in unit, integration, and end-to-end tests.

```python
import json
from pathlib import Path

def load_fixture(filename):
    fixture_path = Path(__file__).parent / "fixtures" / filename
    with open(fixture_path) as f:
        return json.load(f)
```

## Adding New Fixtures

When adding new test fixtures:

1. Place them in the appropriate subdirectory
2. Use descriptive filenames
3. Include documentation in this README
4. Ensure fixtures don't contain sensitive data
5. Keep fixtures small and focused on specific test scenarios