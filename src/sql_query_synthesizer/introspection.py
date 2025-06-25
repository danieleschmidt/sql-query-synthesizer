from dataclasses import dataclass, field
from typing import List

from sqlalchemy import create_engine, inspect


@dataclass
class ColumnInfo:
    """Information about a database column."""

    name: str
    type: str
    nullable: bool
    default: str
    primary_key: bool


@dataclass
class ForeignKeyInfo:
    """Description of a foreign key relationship."""

    column: str
    referred_table: str
    referred_column: str


@dataclass
class TableInfo:
    """Metadata for a single database table."""

    name: str
    columns: List[ColumnInfo] = field(default_factory=list)
    foreign_keys: List[ForeignKeyInfo] = field(default_factory=list)


@dataclass
class SchemaInfo:
    """Collection of tables describing a database schema."""

    tables: List[TableInfo] = field(default_factory=list)


def introspect_database(url: str) -> SchemaInfo:
    """Return database schema information for the given connection URL."""

    engine = create_engine(url)
    inspector = inspect(engine)
    tables: List[TableInfo] = []

    for table_name in inspector.get_table_names():
        columns = [
            ColumnInfo(
                name=col["name"],
                type=str(col["type"]),
                nullable=col.get("nullable", False),
                default=str(col.get("default", "")),
                primary_key=col.get("primary_key", False),
            )
            for col in inspector.get_columns(table_name)
        ]

        foreign_keys = []
        for fk in inspector.get_foreign_keys(table_name):
            for constrained, referred in zip(
                fk.get("constrained_columns", []), fk.get("referred_columns", [])
            ):
                foreign_keys.append(
                    ForeignKeyInfo(
                        column=constrained,
                        referred_table=fk.get("referred_table"),
                        referred_column=referred,
                    )
                )

        tables.append(
            TableInfo(name=table_name, columns=columns, foreign_keys=foreign_keys)
        )

    engine.dispose()
    return SchemaInfo(tables=tables)
