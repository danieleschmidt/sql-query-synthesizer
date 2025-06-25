from dataclasses import dataclass, field
from typing import Any, List, Optional

from sqlalchemy import create_engine, text

from .introspection import SchemaInfo, introspect_database
from .validation import ValidationResult, validate_sql


@dataclass
class QueryResult:
    """Result of a generated SQL query."""

    sql: str
    explanation: str
    data: List[Any] = field(default_factory=list)
    validation: ValidationResult | None = None


class QueryAgent:
    """Simple agent that generates SQL queries using schema context."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.schema: SchemaInfo = introspect_database(database_url)

    def _infer_tables(self, prompt: str) -> List[str]:
        """Return a list of table names mentioned in the prompt."""

        prompt_lower = prompt.lower()
        return [t.name for t in self.schema.tables if t.name.lower() in prompt_lower]

    def _find_relationship(
        self, table_a: str, table_b: str
    ) -> Optional[tuple[str, str, str, str]]:
        """Find a foreign key relationship between two tables."""

        for table in self.schema.tables:
            if table.name == table_a:
                for fk in table.foreign_keys:
                    if fk.referred_table == table_b:
                        return table_a, fk.column, table_b, fk.referred_column
            if table.name == table_b:
                for fk in table.foreign_keys:
                    if fk.referred_table == table_a:
                        return table_b, fk.column, table_a, fk.referred_column
        return None

    def generate_sql(self, prompt: str) -> QueryResult:
        """Generate an SQL statement from a natural language prompt."""

        tables = self._infer_tables(prompt)
        if not tables:
            raise ValueError("Could not identify tables from prompt")

        explanation = ""

        if len(tables) == 1:
            table = tables[0]
            sql = f"SELECT * FROM {table} LIMIT 10"
            explanation = f"Selected rows from `{table}`."
        else:
            t1, t2 = tables[:2]
            rel = self._find_relationship(t1, t2)
            if rel:
                left_table, left_col, right_table, right_col = rel
                sql = (
                    f"SELECT * FROM {t1} JOIN {t2} ON "
                    f"{left_table}.{left_col} = {right_table}.{right_col} LIMIT 10"
                )
                explanation = (
                    f"Joined `{t1}` and `{t2}` using foreign key relationship."
                )
            else:
                sql = f"SELECT * FROM {t1}, {t2} LIMIT 10"
                explanation = f"No relationship found between `{t1}` and `{t2}`; returning cartesian product."

        return QueryResult(sql=sql, explanation=explanation)

    def query(self, prompt: str) -> QueryResult:
        """Generate and execute a query based on the prompt."""

        result = self.generate_sql(prompt)
        result.validation = validate_sql(result.sql)
        with self.engine.connect() as conn:
            rows = conn.execute(text(result.sql)).fetchall()
            result.data = [dict(row) for row in rows]
        return result

    def dispose(self) -> None:
        """Dispose the underlying SQLAlchemy engine."""

        self.engine.dispose()
