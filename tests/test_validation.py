from sql_query_synthesizer.validation import ValidationResult, validate_sql


def test_validate_sql_returns_result_object():
    result = validate_sql("SELECT 1")
    assert isinstance(result, ValidationResult)


def test_validation_result_fields():
    result = validate_sql("SELECT 1")
    assert isinstance(result.is_valid, bool)
    assert isinstance(result.suggestions, list)


def test_invalid_sql_sets_is_valid_false():
    result = validate_sql("SELCT * FRM table")
    assert result.is_valid is False
    assert result.suggestions, "Expected suggestions for invalid SQL"


def test_select_star_triggers_suggestion():
    result = validate_sql("SELECT * FROM users")
    assert result.is_valid
    assert any("specify columns" in s.lower() for s in result.suggestions)


def test_update_without_where_triggers_warning():
    result = validate_sql("UPDATE users SET name='foo'")
    assert result.is_valid
    assert any("where" in s.lower() for s in result.suggestions)


def test_delete_without_where_triggers_warning():
    result = validate_sql("DELETE FROM users")
    assert result.is_valid
    assert any("where" in s.lower() for s in result.suggestions)
