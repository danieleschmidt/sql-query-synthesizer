"""SQLSynthesizer: maps QueryIntent + Schema → valid SQL."""

from __future__ import annotations

from typing import List, Optional

from .parser import Filter, QueryIntent
from .schema import Schema


class SQLSynthesizer:
    """
    Template-based SQL synthesizer.

    Given a QueryIntent and a Schema, produces a syntactically valid SQL
    query using SELECT / COUNT / SUM / AVG / MIN / MAX templates.
    """

    def __init__(self, schema: Schema) -> None:
        self.schema = schema

    def synthesize(self, intent: QueryIntent) -> str:
        if not intent.tables:
            return "-- Could not identify any table in query"

        primary_table = intent.tables[0]

        if intent.operation == "COUNT":
            return self._build_count(intent, primary_table)
        elif intent.operation in ("SUM", "AVG", "MIN", "MAX"):
            return self._build_aggregate(intent, primary_table)
        else:
            return self._build_select(intent, primary_table)

    # ------------------------------------------------------------------ #
    # Templates                                                            #
    # ------------------------------------------------------------------ #

    def _build_select(self, intent: QueryIntent, primary_table: str) -> str:
        # FROM + optional JOINs
        from_clause, join_clauses = self._build_from(intent, primary_table)

        # SELECT columns
        select_cols = self._resolve_columns(intent, primary_table, join_clauses)

        # WHERE
        where_clause = self._build_where(intent.filters, primary_table, join_clauses)

        # GROUP BY
        group_by = self._build_group_by(intent, primary_table, join_clauses)

        # ORDER BY
        order_by = self._build_order_by(intent, primary_table, join_clauses)

        # LIMIT
        limit = f"\nLIMIT {intent.limit}" if intent.limit else ""

        parts = [f"SELECT {select_cols}"]
        parts.append(f"FROM {from_clause}")
        if join_clauses:
            parts.extend(join_clauses)
        if where_clause:
            parts.append(f"WHERE {where_clause}")
        if group_by:
            parts.append(f"GROUP BY {group_by}")
        if order_by:
            parts.append(f"ORDER BY {order_by}")
        if limit:
            parts.append(f"LIMIT {intent.limit}")

        return "\n".join(parts)

    def _build_count(self, intent: QueryIntent, primary_table: str) -> str:
        from_clause, join_clauses = self._build_from(intent, primary_table)
        where_clause = self._build_where(intent.filters, primary_table, join_clauses)
        group_by = self._build_group_by(intent, primary_table, join_clauses)

        if group_by:
            # COUNT per group: include the group column and count
            parts = [f"SELECT {group_by}, COUNT(*) AS count"]
        else:
            parts = ["SELECT COUNT(*) AS count"]

        parts.append(f"FROM {from_clause}")
        if join_clauses:
            parts.extend(join_clauses)
        if where_clause:
            parts.append(f"WHERE {where_clause}")
        if group_by:
            parts.append(f"GROUP BY {group_by}")
        return "\n".join(parts)

    def _build_aggregate(self, intent: QueryIntent, primary_table: str) -> str:
        from_clause, join_clauses = self._build_from(intent, primary_table)
        where_clause = self._build_where(intent.filters, primary_table, join_clauses)
        group_by = self._build_group_by(intent, primary_table, join_clauses)

        # Determine the target column
        agg_col = intent.aggregate_column
        if not agg_col:
            # Fall back to first numeric column in primary table
            table = self.schema.table(primary_table)
            if table:
                for col in table.columns:
                    if col.is_numeric() and not col.primary_key:
                        agg_col = col.name
                        break
        if not agg_col:
            agg_col = "*"

        qualified_col = self._qualify(agg_col, primary_table, join_clauses)
        agg_expr = f"{intent.operation}({qualified_col}) AS {intent.operation.lower()}_{agg_col.replace('*', 'all')}"

        if group_by:
            select_cols = f"{group_by}, {agg_expr}"
        else:
            select_cols = agg_expr

        parts = [f"SELECT {select_cols}", f"FROM {from_clause}"]
        if join_clauses:
            parts.extend(join_clauses)
        if where_clause:
            parts.append(f"WHERE {where_clause}")
        if group_by:
            parts.append(f"GROUP BY {group_by}")

        # Default ORDER BY for aggregates that imply ranking
        if intent.order_by:
            ob = intent.order_by
            qualified_ob = self._qualify(ob.column, primary_table, join_clauses)
            parts.append(f"ORDER BY {qualified_ob} {ob.direction}")
        elif intent.operation in ("MAX", "MIN"):
            parts.append(f"ORDER BY {qualified_col} {'DESC' if intent.operation == 'MAX' else 'ASC'}")
            parts.append("LIMIT 1")

        if intent.limit:
            parts.append(f"LIMIT {intent.limit}")

        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _build_from(self, intent: QueryIntent, primary_table: str):
        """Returns (from_clause_str, list_of_join_strings)."""
        join_clauses = []
        if len(intent.tables) <= 1:
            return primary_table, []

        for a, b in intent.joins:
            fk = self.schema.join_path(a, b)
            if fk:
                if fk.from_table == a.lower():
                    join_clauses.append(
                        f"JOIN {fk.to_table} ON {a}.{fk.from_column} = {fk.to_table}.{fk.to_column}"
                    )
                else:
                    join_clauses.append(
                        f"JOIN {fk.from_table} ON {a}.{fk.to_column} = {fk.from_table}.{fk.from_column}"
                    )

        return primary_table, join_clauses

    def _resolve_columns(self, intent: QueryIntent, primary_table: str, joins: list) -> str:
        if not intent.columns:
            if joins:
                return f"{primary_table}.*"
            return "*"
        return ", ".join(
            self._qualify(c, primary_table, joins) for c in intent.columns
        )

    def _qualify(self, col: str, primary_table: str, joins: list) -> str:
        """Qualify column with table name if joins are present."""
        if not joins or col == "*":
            return col
        # Find which table owns this column
        matches = self.schema.find_column(col)
        if len(matches) == 1:
            return f"{matches[0][0]}.{col}"
        # Ambiguous or not found — use primary table
        return f"{primary_table}.{col}"

    def _build_where(self, filters: List[Filter], primary_table: str, joins: list) -> str:
        if not filters:
            return ""
        clauses = []
        for f in filters:
            col = self._qualify(f.column, primary_table, joins)
            if f.operator in ("IS NULL", "IS NOT NULL"):
                clauses.append(f"{col} {f.operator}")
            elif f.operator == "LIKE":
                clauses.append(f"{col} LIKE '{f.value}'")
            elif isinstance(f.value, str):
                escaped = str(f.value).replace("'", "''")
                clauses.append(f"{col} {f.operator} '{escaped}'")
            else:
                clauses.append(f"{col} {f.operator} {f.value}")
        return " AND ".join(clauses)

    def _build_group_by(self, intent: QueryIntent, primary_table: str, joins: list) -> str:
        if not intent.group_by:
            return ""
        return ", ".join(
            self._qualify(c, primary_table, joins) for c in intent.group_by
        )

    def _build_order_by(self, intent: QueryIntent, primary_table: str, joins: list) -> str:
        if not intent.order_by:
            return ""
        ob = intent.order_by
        col = self._qualify(ob.column, primary_table, joins)
        return f"{col} {ob.direction}"
