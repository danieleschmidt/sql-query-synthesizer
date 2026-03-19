"""NLParser: parses natural language queries into structured intents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .schema import Schema


# ------------------------------------------------------------------ #
# Intent model                                                         #
# ------------------------------------------------------------------ #

@dataclass
class Filter:
    column: str
    operator: str      # =, !=, >, <, >=, <=, LIKE, IN, IS NULL, IS NOT NULL
    value: object      # str, int, float, list, or None


@dataclass
class OrderBy:
    column: str
    direction: str = "ASC"  # ASC | DESC


@dataclass
class QueryIntent:
    """Structured intent extracted from a natural language query."""
    operation: str                    # SELECT | COUNT | SUM | AVG | MIN | MAX
    tables: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)  # [] means *
    filters: List[Filter] = field(default_factory=list)
    joins: List[Tuple[str, str]] = field(default_factory=list)  # (table_a, table_b)
    group_by: List[str] = field(default_factory=list)
    order_by: Optional[OrderBy] = None
    limit: Optional[int] = None
    aggregate_column: Optional[str] = None  # column for SUM/AVG/MIN/MAX
    raw: str = ""


# ------------------------------------------------------------------ #
# Parser                                                               #
# ------------------------------------------------------------------ #

class NLParser:
    """
    Rule-based NL → QueryIntent parser.

    Strategy:
      1. Detect aggregation / operation type
      2. Identify tables referenced (by name match against schema)
      3. Detect columns referenced
      4. Extract filter conditions (WHERE clauses)
      5. Detect GROUP BY, ORDER BY, LIMIT patterns
      6. Infer JOINs when multiple tables are mentioned
    """

    # Patterns for limit / top-N
    _LIMIT_RE = re.compile(
        r"(?:top|first|limit|show\s+me|give\s+me|get)\s+(\d+)", re.IGNORECASE
    )
    _LIMIT_ALT_RE = re.compile(r"(\d+)\s+(?:records?|rows?|results?)", re.IGNORECASE)

    # Patterns for comparisons in filters
    _FILTER_RE = re.compile(
        r"(?:where\s+)?(\w+)\s*(>=|<=|!=|<>|>|<|=|is\s+not\s+null|is\s+null|like|not\s+like)\s*['\"]?([^'\"]+?)['\"]?"
        r"(?=\s+(?:and|or|order|group|limit|$)|\Z)",
        re.IGNORECASE,
    )
    _EQUALS_RE = re.compile(
        r"(\w+)\s+(?:is|=|equals?|are)\s+['\"]?(\w[\w\s]*?)['\"]?"
        r"(?=\s+(?:and|or|order|group|limit|where|$)|\Z)",
        re.IGNORECASE,
    )

    # Aggregation keywords
    _AGG_PATTERNS = [
        (r"\bhow\s+many\b|\bcount\b|\bnumber\s+of\b|\btotal\s+number\b", "COUNT"),
        (r"\baverage\b|\bavg\b|\bmean\b", "AVG"),
        (r"\bsum\s+of\b|\bsum\b|\btotal\s+(?!number)\b", "SUM"),
        (r"\bmaximum\b|\bmax\b|\bhighest\b|\bmost\s+expensive\b|\blargest\b", "MAX"),
        (r"\bminimum\b|\bmin\b|\blowest\b|\bcheapest\b|\bsmallest\b", "MIN"),
    ]

    # Sort direction keywords
    _ORDER_DESC = re.compile(
        r"\b(?:descending|desc|highest|most|largest|newest|latest|recent|expensive)\b",
        re.IGNORECASE,
    )
    _ORDER_ASC = re.compile(
        r"\b(?:ascending|asc|lowest|least|smallest|oldest|earliest|cheapest)\b",
        re.IGNORECASE,
    )
    _ORDER_BY_COL = re.compile(
        r"(?:order(?:ed)?\s+by|sort(?:ed)?\s+by|ranked\s+by)\s+(\w+)", re.IGNORECASE
    )

    # Group by
    _GROUP_BY = re.compile(
        r"(?:group(?:ed)?\s+by|per|by\s+each)\s+(\w+)", re.IGNORECASE
    )

    # Numeric value patterns for filters
    _NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")

    def __init__(self, schema: Schema) -> None:
        self.schema = schema

    def parse(self, query: str) -> QueryIntent:
        q = query.strip()
        intent = QueryIntent(operation="SELECT", raw=q)

        # 1. Detect operation
        intent.operation = self._detect_operation(q)

        # 2. Identify tables
        intent.tables = self._detect_tables(q)

        # 3. Detect columns
        intent.columns, intent.aggregate_column = self._detect_columns(q, intent.tables, intent.operation)

        # 4. Detect filters
        intent.filters = self._detect_filters(q, intent.tables)

        # 5. Detect GROUP BY
        intent.group_by = self._detect_group_by(q, intent.tables)

        # 6. Detect ORDER BY
        intent.order_by = self._detect_order_by(q, intent.tables, intent.operation, intent.aggregate_column)

        # 7. Detect LIMIT
        intent.limit = self._detect_limit(q)

        # 8. Infer JOINs from multiple tables
        if len(intent.tables) > 1:
            intent.joins = self._infer_joins(intent.tables)

        return intent

    # ------------------------------------------------------------------ #
    # Step implementations                                                 #
    # ------------------------------------------------------------------ #

    def _detect_operation(self, q: str) -> str:
        for pattern, op in self._AGG_PATTERNS:
            if re.search(pattern, q, re.IGNORECASE):
                return op
        return "SELECT"

    def _detect_tables(self, q: str) -> List[str]:
        found = []
        q_lower = q.lower()
        # Sort by length desc so longer names match first (e.g. 'order_items' before 'orders')
        for tname in sorted(self.schema.table_names, key=len, reverse=True):
            tname_l = tname.lower()
            base = re.escape(tname_l)
            # Handle both plural and singular matching:
            #   - table "users"    → match "user" or "users"
            #   - table "products" → match "product" or "products"
            #   - table "user"     → match "user" or "users"
            # Strategy: try exact match, then strip trailing 's' from table name
            if re.search(rf"\b{base}s?\b", q_lower):
                found.append(tname)
                q_lower = re.sub(rf"\b{base}s?\b", " ", q_lower)
            elif tname_l.endswith("s"):
                # table = "products", try matching singular "product"
                singular = re.escape(tname_l[:-1])
                if re.search(rf"\b{singular}\b", q_lower):
                    found.append(tname)
                    q_lower = re.sub(rf"\b{singular}\b", " ", q_lower)
        return found

    def _detect_columns(
        self, q: str, tables: List[str], operation: str
    ) -> Tuple[List[str], Optional[str]]:
        """Return (selected_columns, aggregate_column)."""
        aggregate_col = None

        # Collect all columns from relevant tables
        available_cols: List[Tuple[str, str]] = []  # (col_name, table_name)
        target_tables = tables if tables else self.schema.table_names
        for tname in target_tables:
            table = self.schema.table(tname)
            if table:
                for col in table.columns:
                    available_cols.append((col.name, tname))

        q_lower = q.lower()

        # Look for aggregate target ("sum of price", "avg salary", "total revenue", etc.)
        agg_target_pattern = re.compile(
            r"(?:sum|avg|average|max|maximum|min|minimum|total)\s+(?:of\s+)?(\w+)",
            re.IGNORECASE,
        )
        m = agg_target_pattern.search(q)
        if m:
            candidate = m.group(1).lower()
            for col_name, _ in available_cols:
                if col_name.lower() == candidate:
                    aggregate_col = col_name
                    break

        # Also check: any column name mentioned directly in the query that is numeric
        # Prefer columns that appear as standalone words in the query
        if not aggregate_col:
            for col_name, _ in available_cols:
                col = None
                for tname in (tables if tables else self.schema.table_names):
                    table = self.schema.table(tname)
                    if table:
                        c = table.column(col_name)
                        if c and c.is_numeric() and not c.primary_key:
                            col = c
                            break
                if col and re.search(rf"\b{re.escape(col_name.lower())}\b", q_lower):
                    aggregate_col = col_name
                    break

        # Look for explicit column mentions in SELECT context
        selected = []
        for col_name, tname in available_cols:
            if re.search(rf"\b{re.escape(col_name.lower())}\b", q_lower):
                # Don't include if it's only in a filter context
                if col_name not in selected:
                    selected.append(col_name)

        # For COUNT, we rarely want specific columns
        if operation == "COUNT":
            return [], aggregate_col

        # If the query asks for specific columns by name, return them
        # Otherwise return [] (meaning SELECT *)
        # Heuristic: if we found column names that aren't just filter values, include them
        # Keep only columns that appear to be "selected" (not just filter subjects)
        projection_keywords = re.compile(
            r"\b(?:show|list|get|return|display|give\s+me|what\s+(?:is|are))\b",
            re.IGNORECASE,
        )
        if projection_keywords.search(q) and selected:
            return selected, aggregate_col

        return [], aggregate_col

    def _detect_filters(self, q: str, tables: List[str]) -> List[Filter]:
        filters = []
        seen = set()

        # Gather all column names for context
        all_cols = set()
        target_tables = tables if tables else self.schema.table_names
        for tname in target_tables:
            table = self.schema.table(tname)
            if table:
                for col in table.columns:
                    all_cols.add(col.name.lower())

        # Pattern: "column > value", "column = 'value'", etc.
        comparison_re = re.compile(
            r"\b(\w+)\s*(>=|<=|!=|<>|>|<|=)\s*['\"]?(-?\d+(?:\.\d+)?|[\w][\w\s]*?)['\"]?"
            r"(?=\s*(?:and|or|order|group|limit|\Z))",
            re.IGNORECASE,
        )
        for m in comparison_re.finditer(q):
            col = m.group(1).lower()
            op = m.group(2).replace("<>", "!=")
            val_str = m.group(3).strip()
            if col not in all_cols:
                continue
            key = (col, op, val_str)
            if key in seen:
                continue
            seen.add(key)
            val = self._coerce_value(val_str)
            filters.append(Filter(column=col, operator=op, value=val))

        # Pattern: "where name is/equals X"
        equals_re = re.compile(
            r"\b(\w+)\s+(?:is|equals?|are)\s+['\"]?([\w][\w\s-]*?)['\"]?"
            r"(?=\s*(?:and|or|order|group|limit|\Z))",
            re.IGNORECASE,
        )
        for m in equals_re.finditer(q):
            col = m.group(1).lower()
            val_str = m.group(2).strip()
            if col not in all_cols:
                continue
            if col in ("what", "which", "where", "when", "who", "how"):
                continue
            key = (col, "=", val_str)
            if key in seen:
                continue
            seen.add(key)
            val = self._coerce_value(val_str)
            filters.append(Filter(column=col, operator="=", value=val))

        # Pattern: "containing X" / "with name X" / "named X"
        named_re = re.compile(
            r"\b(?:named?|called|with\s+name)\s+['\"]?([\w][\w\s-]*?)['\"]?"
            r"(?=\s*(?:and|or|order|group|limit|\Z))",
            re.IGNORECASE,
        )
        for m in named_re.finditer(q):
            val_str = m.group(1).strip()
            # Find a 'name' column in scope
            for tname in target_tables:
                table = self.schema.table(tname)
                if table and table.has_column("name"):
                    key = ("name", "=", val_str)
                    if key not in seen:
                        seen.add(key)
                        filters.append(Filter(column="name", operator="=", value=val_str))
                    break

        # Pattern: "status is/= X" already caught above; also "status X"
        # Detect "older than N" / "more than N"
        older_re = re.compile(
            r"\b(?:older|greater|more|higher|above)\s+than\s+(\d+(?:\.\d+)?)", re.IGNORECASE
        )
        m = older_re.search(q)
        if m:
            val = self._coerce_value(m.group(1))
            # Find likely numeric column
            for tname in target_tables:
                table = self.schema.table(tname)
                if not table:
                    continue
                for col in table.columns:
                    if col.is_numeric() and col.name.lower() not in ("id",):
                        key = (col.name.lower(), ">", str(val))
                        if key not in seen:
                            seen.add(key)
                            filters.append(Filter(column=col.name, operator=">", value=val))
                        break

        younger_re = re.compile(
            r"\b(?:younger|less|lower|below|cheaper|smaller)\s+than\s+(\d+(?:\.\d+)?)", re.IGNORECASE
        )
        m = younger_re.search(q)
        if m:
            val = self._coerce_value(m.group(1))
            for tname in target_tables:
                table = self.schema.table(tname)
                if not table:
                    continue
                for col in table.columns:
                    if col.is_numeric() and col.name.lower() not in ("id",):
                        key = (col.name.lower(), "<", str(val))
                        if key not in seen:
                            seen.add(key)
                            filters.append(Filter(column=col.name, operator="<", value=val))
                        break

        return filters

    def _detect_group_by(self, q: str, tables: List[str]) -> List[str]:
        m = self._GROUP_BY.search(q)
        if not m:
            return []
        candidate = m.group(1).lower()
        # Validate against schema
        for tname in (tables if tables else self.schema.table_names):
            table = self.schema.table(tname)
            if table and table.has_column(candidate):
                return [candidate]
        return []

    def _detect_order_by(
        self, q: str, tables: List[str], operation: str, aggregate_col: Optional[str]
    ) -> Optional[OrderBy]:
        # Explicit "order by col"
        m = self._ORDER_BY_COL.search(q)
        if m:
            col = m.group(1).lower()
            direction = "DESC" if self._ORDER_DESC.search(q) else "ASC"
            return OrderBy(column=col, direction=direction)

        # Implicit: "most expensive" / "highest price" etc.
        if self._ORDER_DESC.search(q):
            # Try to find a numeric column to sort by
            col = self._best_numeric_col(tables, aggregate_col)
            if col:
                return OrderBy(column=col, direction="DESC")
        if self._ORDER_ASC.search(q):
            col = self._best_numeric_col(tables, aggregate_col)
            if col:
                return OrderBy(column=col, direction="ASC")

        return None

    def _detect_limit(self, q: str) -> Optional[int]:
        m = self._LIMIT_RE.search(q)
        if m:
            return int(m.group(1))
        m = self._LIMIT_ALT_RE.search(q)
        if m:
            return int(m.group(1))
        return None

    def _infer_joins(self, tables: List[str]) -> List[Tuple[str, str]]:
        joins = []
        for i in range(len(tables) - 1):
            joins.append((tables[i], tables[i + 1]))
        return joins

    def _best_numeric_col(self, tables: List[str], aggregate_col: Optional[str]) -> Optional[str]:
        if aggregate_col:
            return aggregate_col
        for tname in (tables if tables else self.schema.table_names):
            table = self.schema.table(tname)
            if not table:
                continue
            for col in table.columns:
                if col.is_numeric() and not col.primary_key:
                    return col.name
        return None

    @staticmethod
    def _coerce_value(s: str) -> object:
        s = s.strip()
        try:
            if "." in s:
                return float(s)
            return int(s)
        except ValueError:
            return s
