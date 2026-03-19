"""Schema: represents a database schema — tables, columns, types, foreign keys."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Column:
    name: str
    type: str          # normalized: TEXT, INTEGER, REAL, BLOB, NUMERIC
    nullable: bool = True
    primary_key: bool = False

    def is_numeric(self) -> bool:
        return self.type in ("INTEGER", "REAL", "NUMERIC")

    def is_text(self) -> bool:
        return self.type == "TEXT"


@dataclass
class ForeignKey:
    from_table: str
    from_column: str
    to_table: str
    to_column: str


@dataclass
class Table:
    name: str
    columns: List[Column] = field(default_factory=list)

    def column(self, name: str) -> Optional[Column]:
        for c in self.columns:
            if c.name.lower() == name.lower():
                return c
        return None

    def has_column(self, name: str) -> bool:
        return self.column(name) is not None

    @property
    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    @property
    def primary_keys(self) -> List[str]:
        return [c.name for c in self.columns if c.primary_key]


class Schema:
    """
    Represents a database schema with tables, columns, and foreign keys.

    Can be loaded from:
      - a dict (see from_dict)
      - SQL CREATE TABLE statements (see from_sql)
    """

    def __init__(self) -> None:
        self._tables: Dict[str, Table] = {}
        self._foreign_keys: List[ForeignKey] = []

    # ------------------------------------------------------------------ #
    # Factories                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, definition: dict) -> "Schema":
        """
        Build a schema from a dict of the form:
          {
            "tables": {
              "users": {
                "columns": [
                  {"name": "id", "type": "INTEGER", "primary_key": true},
                  {"name": "name", "type": "TEXT"}
                ]
              }
            },
            "foreign_keys": [
              {"from": "orders.user_id", "to": "users.id"}
            ]
          }
        """
        schema = cls()
        for tname, tdef in definition.get("tables", {}).items():
            table = Table(name=tname)
            for cdef in tdef.get("columns", []):
                col = Column(
                    name=cdef["name"],
                    type=cls._normalize_type(cdef.get("type", "TEXT")),
                    nullable=cdef.get("nullable", True),
                    primary_key=cdef.get("primary_key", False),
                )
                table.columns.append(col)
            schema._tables[tname.lower()] = table

        for fk in definition.get("foreign_keys", []):
            from_ref = fk["from"].split(".")
            to_ref = fk["to"].split(".")
            if len(from_ref) == 2 and len(to_ref) == 2:
                schema._foreign_keys.append(
                    ForeignKey(
                        from_table=from_ref[0].lower(),
                        from_column=from_ref[1],
                        to_table=to_ref[0].lower(),
                        to_column=to_ref[1],
                    )
                )
        return schema

    @classmethod
    def from_sql(cls, sql: str) -> "Schema":
        """Parse one or more CREATE TABLE statements into a Schema."""
        schema = cls()
        # Find CREATE TABLE blocks with proper paren matching
        blocks = cls._extract_create_tables(sql)
        for table_name, body in blocks:
            table = Table(name=table_name)
            schema._tables[table_name.lower()] = table

            # Parse columns and inline constraints
            # Split on commas OUTSIDE parentheses only
            lines = [line.strip() for line in cls._split_columns(body) if line.strip()]
            for line in lines:
                # Skip table-level constraints
                upper = line.upper().strip()
                if upper.startswith("PRIMARY KEY") or upper.startswith("UNIQUE") or upper.startswith("CHECK"):
                    continue
                if upper.startswith("FOREIGN KEY"):
                    # FOREIGN KEY (col) REFERENCES other_table(other_col)
                    m = re.match(
                        r"FOREIGN\s+KEY\s*\((\w+)\)\s+REFERENCES\s+(\w+)\s*\((\w+)\)",
                        line,
                        re.IGNORECASE,
                    )
                    if m:
                        schema._foreign_keys.append(
                            ForeignKey(
                                from_table=table_name.lower(),
                                from_column=m.group(1),
                                to_table=m.group(2).lower(),
                                to_column=m.group(3),
                            )
                        )
                    continue
                # Regular column definition: name TYPE [constraints...]
                m = re.match(r"(\w+)\s+(\w+)(.*)", line)
                if not m:
                    continue
                col_name = m.group(1)
                col_type = cls._normalize_type(m.group(2))
                rest = m.group(3).upper()
                is_pk = "PRIMARY KEY" in rest
                not_null = "NOT NULL" in rest
                col = Column(
                    name=col_name,
                    type=col_type,
                    nullable=(not not_null and not is_pk),
                    primary_key=is_pk,
                )
                table.columns.append(col)

        return schema

    # ------------------------------------------------------------------ #
    # Access                                                               #
    # ------------------------------------------------------------------ #

    def table(self, name: str) -> Optional[Table]:
        return self._tables.get(name.lower())

    def has_table(self, name: str) -> bool:
        return name.lower() in self._tables

    @property
    def table_names(self) -> List[str]:
        return [t.name for t in self._tables.values()]

    @property
    def foreign_keys(self) -> List[ForeignKey]:
        return list(self._foreign_keys)

    def join_path(self, table_a: str, table_b: str) -> Optional[ForeignKey]:
        """Return a FK that connects table_a and table_b (in either direction)."""
        a, b = table_a.lower(), table_b.lower()
        for fk in self._foreign_keys:
            if (fk.from_table == a and fk.to_table == b) or (
                fk.from_table == b and fk.to_table == a
            ):
                return fk
        return None

    def find_column(self, col_name: str) -> List[Tuple[str, Column]]:
        """Return all (table_name, Column) pairs where column name matches."""
        results = []
        for tname, table in self._tables.items():
            col = table.column(col_name)
            if col:
                results.append((tname, col))
        return results

    def ddl(self) -> str:
        """Generate CREATE TABLE statements for this schema."""
        lines = []
        for table in self._tables.values():
            cols = []
            for c in table.columns:
                parts = [c.name, c.type]
                if c.primary_key:
                    parts.append("PRIMARY KEY")
                if not c.nullable and not c.primary_key:
                    parts.append("NOT NULL")
                cols.append("  " + " ".join(parts))
            for fk in self._foreign_keys:
                if fk.from_table == table.name.lower():
                    cols.append(
                        f"  FOREIGN KEY ({fk.from_column}) REFERENCES {fk.to_table}({fk.to_column})"
                    )
            lines.append(f"CREATE TABLE {table.name} (\n" + ",\n".join(cols) + "\n);")
        return "\n\n".join(lines)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def _extract_create_tables(cls, sql: str) -> list:
        """Extract (table_name, body) pairs from SQL with correct paren matching."""
        results = []
        header_re = re.compile(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\(",
            re.IGNORECASE,
        )
        pos = 0
        for m in header_re.finditer(sql):
            table_name = m.group(1)
            start = m.end()  # position right after the opening '('
            depth = 1
            i = start
            while i < len(sql) and depth > 0:
                if sql[i] == "(":
                    depth += 1
                elif sql[i] == ")":
                    depth -= 1
                i += 1
            # sql[start:i-1] is the body (excluding the final ')')
            body = sql[start : i - 1]
            results.append((table_name, body))
        return results

    @staticmethod
    def _split_columns(body: str) -> list:
        """Split a CREATE TABLE body on commas, but not commas inside parentheses."""
        parts = []
        depth = 0
        current = []
        for ch in body:
            if ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current))
        return parts

    @staticmethod
    def _normalize_type(t: str) -> str:
        t = t.upper().split("(")[0].strip()
        if t in ("INT", "INTEGER", "SMALLINT", "BIGINT", "TINYINT", "MEDIUMINT"):
            return "INTEGER"
        if t in ("REAL", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "NUMBER"):
            return "REAL"
        if t in ("TEXT", "VARCHAR", "CHAR", "STRING", "NVARCHAR", "CLOB"):
            return "TEXT"
        if t in ("BLOB", "BINARY", "VARBINARY"):
            return "BLOB"
        if t in ("DATE", "DATETIME", "TIMESTAMP", "TIME"):
            return "TEXT"  # SQLite stores dates as TEXT
        return "TEXT"

    def __repr__(self) -> str:
        return f"Schema(tables={self.table_names})"
