"""Input validation utilities for enhanced security."""

import re
import string
from typing import Optional


# SQL injection patterns for enhanced detection
SQL_INJECTION_PATTERNS = [
    re.compile(r'\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b', re.IGNORECASE),
    re.compile(r'[;\'"\\].*(-{2}|/\*|\*/)', re.IGNORECASE),
    re.compile(r'\b(or|and)\s+[\'"][^\'"]*(=|like)', re.IGNORECASE),
    re.compile(r'[\'"](\s*(or|and)\s*)?[\'"]?(\s*(=|like)\s*)?[\'"]', re.IGNORECASE),
    re.compile(r'\b(xp_|sp_|cmdshell)', re.IGNORECASE),
    re.compile(r'\\x[0-9a-f]{2}', re.IGNORECASE),  # Hex encoding
    re.compile(r'char\s*\(', re.IGNORECASE),  # CHAR() function
    re.compile(r'ascii\s*\(', re.IGNORECASE),  # ASCII() function
    re.compile(r'waitfor\s+delay', re.IGNORECASE),  # Time-based injection
]

# Safe SQL identifier pattern (tables, columns)
SAFE_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

# Maximum lengths for various inputs
MAX_QUERY_LENGTH = 10000
MAX_TABLE_NAME_LENGTH = 128
MAX_COLUMN_NAME_LENGTH = 128


def validate_query_length(query: str, max_length: int = MAX_QUERY_LENGTH) -> bool:
    """Validate query length to prevent DoS attacks."""
    return len(query.strip()) <= max_length


def validate_table_name(table_name: str) -> bool:
    """Validate table name for safety and compliance."""
    if not table_name or len(table_name) > MAX_TABLE_NAME_LENGTH:
        return False
    
    # Must be a safe SQL identifier
    if not SAFE_IDENTIFIER_PATTERN.match(table_name):
        return False
    
    # Cannot be reserved words (basic list)
    reserved_words = {
        'select', 'insert', 'update', 'delete', 'drop', 'create', 'alter',
        'table', 'database', 'schema', 'index', 'view', 'trigger', 'procedure'
    }
    
    return table_name.lower() not in reserved_words


def sanitize_input(text: str) -> str:
    """Sanitize user input by removing dangerous characters."""
    if not text:
        return ""
    
    # Remove null bytes and control characters
    sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
    
    # Limit to printable ASCII + basic punctuation
    allowed_chars = string.ascii_letters + string.digits + string.punctuation + ' \t\n\r'
    sanitized = ''.join(char for char in sanitized if char in allowed_chars)
    
    # Normalize whitespace
    sanitized = ' '.join(sanitized.split())
    
    return sanitized


def is_safe_sql_identifier(identifier: str) -> bool:
    """Check if string is a safe SQL identifier (table/column name)."""
    if not identifier or len(identifier) > MAX_COLUMN_NAME_LENGTH:
        return False
    
    return SAFE_IDENTIFIER_PATTERN.match(identifier) is not None


def contains_sql_injection_pattern(text: str) -> Optional[str]:
    """Check for SQL injection patterns and return the matched pattern type."""
    if not text:
        return None
    
    text_lower = text.lower()
    
    for i, pattern in enumerate(SQL_INJECTION_PATTERNS):
        if pattern.search(text_lower):
            pattern_names = [
                'sql_keywords',
                'comment_injection', 
                'boolean_injection',
                'quote_injection',
                'system_functions',
                'hex_encoding',
                'char_function',
                'ascii_function',
                'time_based_injection'
            ]
            return pattern_names[min(i, len(pattern_names) - 1)]
    
    return None


def validate_column_list(columns: list) -> bool:
    """Validate a list of column names for safety."""
    if not columns or len(columns) > 100:  # Reasonable limit
        return False
    
    return all(is_safe_sql_identifier(col) for col in columns)


def validate_order_by_clause(order_clause: str) -> bool:
    """Validate ORDER BY clause for safety."""
    if not order_clause:
        return True
    
    # Simple validation - column name optionally followed by ASC/DESC
    pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*(\s+(asc|desc))?(,\s*[a-zA-Z_][a-zA-Z0-9_]*(\s+(asc|desc))?)*$', re.IGNORECASE)
    return pattern.match(order_clause.strip()) is not None


def validate_limit_clause(limit_value: str) -> bool:
    """Validate LIMIT clause to ensure it's a positive integer."""
    try:
        limit = int(limit_value)
        return 0 < limit <= 10000  # Reasonable upper bound
    except (ValueError, TypeError):
        return False