"""Common types for SQL Synthesizer."""

from dataclasses import dataclass, field
from typing import List, Any


@dataclass
class QueryResult:
    """Container for the generated SQL, optional explanation and data."""
    
    sql: str
    explanation: str = ""
    data: List[Any] = field(default_factory=list)