"""User experience utilities for friendly error messages and CLI improvements."""

from typing import List, Optional


class UserFriendlyError(ValueError):
    """Exception class for user-friendly error messages with suggestions."""
    
    def __init__(self, message: str, suggestion: str = None, error_code: str = None):
        self.message = message
        self.suggestion = suggestion
        self.error_code = error_code
        
        full_message = message
        if suggestion:
            full_message += f"\n\n💡 Suggestion: {suggestion}"
        
        super().__init__(full_message)


def create_empty_question_error() -> UserFriendlyError:
    """Create a user-friendly error for empty questions."""
    return UserFriendlyError(
        "Please provide a question about your data.",
        "Try asking something like 'How many users are there?' or 'Show me the latest orders'",
        "EMPTY_QUESTION"
    )


def create_invalid_table_error(table_name: str, available_tables: List[str]) -> UserFriendlyError:
    """Create a user-friendly error for invalid table names."""
    message = f"Table '{table_name}' was not found in your database."
    
    if available_tables:
        if len(available_tables) <= 5:
            tables_list = ", ".join(available_tables)
            suggestion = f"Available tables are: {tables_list}"
        else:
            tables_list = ", ".join(available_tables[:5])
            suggestion = f"Available tables include: {tables_list} (and {len(available_tables) - 5} more)"
    else:
        suggestion = "Your database doesn't seem to have any tables yet. Try creating some tables first."
    
    return UserFriendlyError(message, suggestion, "INVALID_TABLE")


def create_unsafe_input_error() -> UserFriendlyError:
    """Create a user-friendly error for potentially unsafe input."""
    return UserFriendlyError(
        "Your question contains patterns that could be unsafe for security reasons.",
        "Please ask questions using natural language. For example: 'Show me users' instead of 'SELECT * FROM users; DROP TABLE users'",
        "UNSAFE_INPUT"
    )


def create_question_too_long_error(max_length: int) -> UserFriendlyError:
    """Create a user-friendly error for questions that are too long."""
    return UserFriendlyError(
        f"Your question is too long (maximum {max_length} characters).",
        "Try breaking your question into smaller, more specific parts",
        "QUESTION_TOO_LONG"
    )


def create_invalid_sql_error(attempted_operation: str = None) -> UserFriendlyError:
    """Create a user-friendly error for invalid SQL operations."""
    message = "Only SELECT queries are allowed for security reasons."
    
    if attempted_operation:
        suggestion = f"Instead of '{attempted_operation}', try asking a question like 'Show me data from table_name'"
    else:
        suggestion = "Try asking questions like 'How many records?' or 'Show me the data' instead of writing raw SQL"
    
    return UserFriendlyError(message, suggestion, "INVALID_SQL")


def create_multiple_statements_error() -> UserFriendlyError:
    """Create a user-friendly error for multiple SQL statements."""
    return UserFriendlyError(
        "Multiple SQL statements are not allowed for security reasons.",
        "Please ask one question at a time, like 'Show me users' or 'Count the orders'",
        "MULTIPLE_STATEMENTS"
    )


def create_openai_not_configured_error() -> UserFriendlyError:
    """Create a user-friendly error when OpenAI is not configured."""
    return UserFriendlyError(
        "AI-powered SQL generation is not available because no OpenAI API key is configured.",
        "Set the OPENAI_API_KEY environment variable or use --openai-api-key to enable smart SQL generation",
        "OPENAI_NOT_CONFIGURED"
    )


def create_openai_package_missing_error() -> UserFriendlyError:
    """Create a user-friendly error when OpenAI package is missing."""
    return UserFriendlyError(
        "AI-powered SQL generation requires the OpenAI package to be installed.",
        "Install it with: pip install openai",
        "OPENAI_PACKAGE_MISSING"
    )


def create_invalid_question_type_error() -> UserFriendlyError:
    """Create a user-friendly error for non-string questions."""
    return UserFriendlyError(
        "Questions must be provided as text.",
        "Try asking something like 'Show me the data' or 'How many records are there?'",
        "INVALID_QUESTION_TYPE"
    )


def suggest_natural_language_alternatives(technical_query: str) -> Optional[str]:
    """Suggest natural language alternatives for technical queries."""
    suggestions = {
        "select * from": "Show me all data from",
        "count(*)": "How many records are there",
        "drop table": "This operation is not allowed for security",
        "delete from": "This operation is not allowed for security",
        "update": "This operation is not allowed for security",
        "insert into": "This operation is not allowed for security",
    }
    
    query_lower = technical_query.lower()
    for pattern, suggestion in suggestions.items():
        if pattern in query_lower:
            return suggestion
    
    return None


def format_cli_error(error: Exception) -> str:
    """Format errors for CLI display with colors and helpful information."""
    if isinstance(error, UserFriendlyError):
        # Format user-friendly errors nicely
        message = f"❌ Error: {error.message}"
        if error.suggestion:
            message += f"\n{error.suggestion}"
        return message
    else:
        # Format other errors with some helpful context
        return f"❌ Error: {str(error)}\n\nFor help, use --help or check your question for typos."


def get_usage_examples() -> List[str]:
    """Get list of usage examples for help text."""
    return [
        "query-agent --database-url sqlite:///data.db 'How many users are there?'",
        "query-agent --interactive",
        "query-agent --list-tables",
        "query-agent 'Show me the top 10 orders by amount'",
        "query-agent --openai-api-key YOUR_KEY 'Find customers from California'",
    ]


def get_common_questions() -> List[str]:
    """Get list of common question patterns users can ask."""
    return [
        "How many [table_name] are there?",
        "Show me all [table_name]",
        "What are the latest [table_name]?", 
        "Find [table_name] with [condition]",
        "Count [table_name] by [column]",
        "Show [table_name] from this week/month/year",
    ]