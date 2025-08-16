"""Helper utilities for common operations."""

import json
import re
from typing import Any, Dict, Optional, Union


def format_duration(milliseconds: float) -> str:
    """Format duration in milliseconds to human-readable string."""
    if milliseconds < 1000:
        return f"{milliseconds:.1f}ms"
    elif milliseconds < 60000:
        return f"{milliseconds/1000:.1f}s"
    else:
        minutes = int(milliseconds // 60000)
        seconds = (milliseconds % 60000) / 1000
        return f"{minutes}m {seconds:.1f}s"


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string to maximum length with suffix."""
    if not text or len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def safe_json_dumps(obj: Any, max_length: int = 10000) -> str:
    """Safely serialize object to JSON with length limits."""
    try:
        result = json.dumps(obj, default=str, ensure_ascii=False)
        return truncate_string(result, max_length)
    except (TypeError, ValueError) as e:
        return f"<serialization error: {str(e)}>"


def extract_client_info(request_headers: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Extract client information from HTTP headers."""
    return {
        "user_agent": request_headers.get("User-Agent"),
        "client_ip": (
            request_headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or request_headers.get("X-Real-IP")
            or request_headers.get("Remote-Addr")
        ),
        "accept_language": request_headers.get("Accept-Language"),
        "referer": request_headers.get("Referer"),
    }


def sanitize_error_message(error_msg: str) -> str:
    """Sanitize error messages to prevent information leakage."""
    # Remove file paths
    error_msg = re.sub(r"/[^\s]*", "<path>", error_msg)

    # Remove IP addresses
    error_msg = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "<ip>", error_msg)

    # Remove potential passwords/tokens
    error_msg = re.sub(
        r"password[=:]\s*\S+", "password=<hidden>", error_msg, flags=re.IGNORECASE
    )
    error_msg = re.sub(
        r"token[=:]\s*\S+", "token=<hidden>", error_msg, flags=re.IGNORECASE
    )
    error_msg = re.sub(r"key[=:]\s*\S+", "key=<hidden>", error_msg, flags=re.IGNORECASE)

    return error_msg


def normalize_sql(sql: str) -> str:
    """Normalize SQL query for consistent formatting."""
    if not sql:
        return ""

    # Remove excessive whitespace
    sql = " ".join(sql.split())

    # Ensure semicolon at end
    sql = sql.rstrip(";") + ";"

    return sql


def parse_database_url(url: str) -> Dict[str, Optional[str]]:
    """Parse database URL into components."""
    # Basic URL parsing for database connections
    pattern = re.compile(
        r"(?P<scheme>\w+)://"
        r"(?:(?P<username>[^:@]+)(?::(?P<password>[^@]+))?@)?"
        r"(?P<host>[^:/]+)"
        r"(?::(?P<port>\d+))?"
        r"(?:/(?P<database>\w+))?"
        r"(?:\?(?P<params>.+))?"
    )

    match = pattern.match(url)
    if not match:
        return {}

    return {
        "scheme": match.group("scheme"),
        "username": match.group("username"),
        "password": "***" if match.group("password") else None,  # Hide password
        "host": match.group("host"),
        "port": match.group("port"),
        "database": match.group("database"),
        "params": match.group("params"),
    }


def calculate_cache_key(query: str, params: Dict[str, Any] = None) -> str:
    """Calculate deterministic cache key for query and parameters."""
    import hashlib

    # Normalize query for caching
    normalized = normalize_sql(query.lower().strip())

    # Include parameters if provided
    if params:
        params_str = json.dumps(params, sort_keys=True, default=str)
        cache_input = f"{normalized}|{params_str}"
    else:
        cache_input = normalized

    # Generate hash
    return hashlib.sha256(cache_input.encode()).hexdigest()[:16]


def format_table_info(
    table_name: str, row_count: Optional[int] = None, columns: Optional[list] = None
) -> str:
    """Format table information for display."""
    info = f"Table: {table_name}"

    if row_count is not None:
        info += f" ({row_count:,} rows)"

    if columns:
        column_list = ", ".join(columns[:5])  # Show first 5 columns
        if len(columns) > 5:
            column_list += f"... ({len(columns)} total)"
        info += f"\nColumns: {column_list}"

    return info


def validate_pagination_params(
    page: Union[str, int], page_size: Union[str, int]
) -> tuple:
    """Validate and normalize pagination parameters."""
    try:
        page = int(page)
        page_size = int(page_size)

        # Ensure positive values
        page = max(1, page)
        page_size = max(1, min(page_size, 1000))  # Cap at 1000

        return page, page_size
    except (ValueError, TypeError):
        return 1, 10  # Default values


def format_file_size(size_bytes: int) -> str:
    """Format file size in bytes to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
