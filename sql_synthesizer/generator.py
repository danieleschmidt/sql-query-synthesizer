"""Utility functions for SQL generation."""

import re


def naive_generate_sql(question: str, tables: list[str], max_rows: int = 5) -> str:
    """Return a simple SQL statement for *question* using keyword matching.
    
    This function only generates safe SELECT queries with validated table names.
    """
    q = question.lower()
    
    # Validate max_rows parameter
    if not isinstance(max_rows, int) or max_rows < 1 or max_rows > 10000:
        max_rows = 5
    
    for table in tables:
        # Ensure table name is safe (already validated by caller, but double-check)
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
            continue
            
        if re.search(rf"\b{re.escape(table.lower())}\b", q):
            if any(word in q for word in ["count", "how many", "number"]):
                # Use quoted identifier for safety
                return f'SELECT COUNT(*) FROM "{table}";'
            return f'SELECT * FROM "{table}" LIMIT {max_rows};'
    
    return f"-- No matching table found for: {question}"

