"""Common types for SQL Synthesizer."""

import math
from dataclasses import dataclass, field
from typing import List, Any, Optional, Dict


@dataclass
class PaginationInfo:
    """Information about pagination state for query results."""
    
    page: int
    page_size: int
    total_count: int
    total_pages: int
    has_next: bool
    has_previous: bool
    
    @classmethod
    def create(cls, page: int, page_size: int, total_count: int) -> 'PaginationInfo':
        """Create pagination info with calculated values.
        
        Args:
            page: Current page number (1-based)
            page_size: Number of items per page
            total_count: Total number of items
            
        Returns:
            PaginationInfo with calculated pagination state
        """
        total_pages = math.ceil(total_count / page_size) if total_count > 0 else 0
        has_next = page < total_pages
        has_previous = page > 1
        
        return cls(
            page=page,
            page_size=page_size,
            total_count=total_count,
            total_pages=total_pages,
            has_next=has_next,
            has_previous=has_previous
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pagination info to dictionary for JSON serialization."""
        return {
            'page': self.page,
            'page_size': self.page_size,
            'total_count': self.total_count,
            'total_pages': self.total_pages,
            'has_next': self.has_next,
            'has_previous': self.has_previous
        }


@dataclass
class QueryResult:
    """Container for the generated SQL, optional explanation and data."""
    
    sql: str
    explanation: str = ""
    data: List[Any] = field(default_factory=list)
    pagination: Optional[PaginationInfo] = None