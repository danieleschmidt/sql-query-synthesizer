"""
sql_synthesizer — Natural language to SQL, schema-aware, stdlib only.

Components:
  Schema          — represents tables, columns, types, foreign keys
  NLParser        — parses NL queries into structured intents
  SQLSynthesizer  — maps intent + schema → valid SQL
  QueryValidator  — validates SQL against the schema
  QueryExecutor   — runs SQL against an in-memory SQLite database
"""

from .schema import Schema
from .parser import NLParser
from .synthesizer import SQLSynthesizer
from .validator import QueryValidator
from .executor import QueryExecutor

__all__ = ["Schema", "NLParser", "SQLSynthesizer", "QueryValidator", "QueryExecutor"]
__version__ = "1.0.0"
