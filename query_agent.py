"""Command-line interface for the :class:`QueryAgent`."""

import argparse
import os

from dotenv import load_dotenv

import yaml
from sql_synthesizer import QueryAgent
import csv


def load_env_config(config: str, env: str) -> dict:
    """Load a single environment configuration from a YAML file."""
    if not os.path.exists(config):
        return {}
    with open(config, "r") as fh:
        data = yaml.safe_load(fh) or {}
    try:
        return data["databases"][env]
    except KeyError as exc:
        raise KeyError(f"Environment '{env}' not found in {config}") from exc


def save_csv(filename: str, rows: list[dict]) -> None:
    """Write ``rows`` to ``filename`` in CSV format."""
    if not rows:
        return
    with open(filename, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``query-agent`` command.

    *argv* can be provided to supply command-line arguments for testing.
    """
    load_dotenv()
    parser = argparse.ArgumentParser(description="Interactive NL to SQL agent")
    parser.add_argument("--database-url")
    parser.add_argument("--config")
    parser.add_argument("--env")
    parser.add_argument("--schema-cache-ttl", type=int)
    parser.add_argument("--clear-cache", action="store_true", help="Clear cached schema and query results")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--cache-ttl",
        type=int,
        help="Cache query results for the given number of seconds",
    )
    parser.add_argument("--describe-table")
    parser.add_argument("--execute-sql", help="Execute raw SQL instead of generating from a question")
    parser.add_argument("--output-csv", help="Write query results to a CSV file")
    parser.add_argument("--openai-api-key", help="OpenAI API key for LLM-based generation")
    parser.add_argument("--openai-model", help="OpenAI model to use for generation")
    parser.add_argument(
        "--openai-timeout",
        type=float,
        help="Timeout in seconds for OpenAI API requests",
    )
    parser.add_argument("--explain", action="store_true", help="Display EXPLAIN plan instead of rows")
    parser.add_argument("--sql-only", action="store_true", help="Print generated SQL without executing")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--list-tables", action="store_true", help="List available tables and exit")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set log level")
    parser.add_argument("--log-format", choices=["standard", "json"], help="Set log format")
    parser.add_argument("--enable-structured-logging", action="store_true", help="Enable structured logging with trace IDs")
    parser.add_argument("question", nargs="?")
    args = parser.parse_args(argv)

    config_path = args.config or os.environ.get("QUERY_AGENT_CONFIG", "config/databases.yaml")
    env_name = args.env or os.environ.get("QUERY_AGENT_ENV", "default")
    config = load_env_config(config_path, env_name)
    db_url = (
        args.database_url
        or os.environ.get("DATABASE_URL")
        or config.get("url")
    )
    schema_ttl = args.schema_cache_ttl
    if schema_ttl is None:
        env_ttl = os.environ.get("QUERY_AGENT_SCHEMA_CACHE_TTL")
        if env_ttl is not None:
            schema_ttl = int(env_ttl)
        else:
            schema_ttl = config.get("schema_cache_ttl", 0)
    max_rows = args.max_rows
    if max_rows is None:
        max_rows = int(os.environ.get("QUERY_AGENT_MAX_ROWS", 5))
    cache_ttl = args.cache_ttl
    if cache_ttl is None:
        cache_ttl = int(os.environ.get("QUERY_AGENT_CACHE_TTL", 0))
        cache_ttl = config.get("query_cache_ttl", cache_ttl)
    openai_timeout = args.openai_timeout
    if openai_timeout is None:
        env_to = os.environ.get("QUERY_AGENT_OPENAI_TIMEOUT")
        if env_to is not None:
            openai_timeout = float(env_to)
    # Configure logging before creating agent
    enable_structured_logging = (
        args.enable_structured_logging 
        or os.environ.get("QUERY_AGENT_STRUCTURED_LOGGING", "").lower() in ("true", "1", "yes")
    )
    
    if enable_structured_logging or args.log_level or args.log_format:
        from sql_synthesizer.logging_utils import configure_logging
        configure_logging(
            level=args.log_level,
            format_type=args.log_format,
            enable_json=enable_structured_logging
        )
    
    agent = QueryAgent(
        db_url,
        schema_cache_ttl=schema_ttl,
        max_rows=max_rows,
        query_cache_ttl=cache_ttl,
        openai_api_key=args.openai_api_key,
        openai_model=args.openai_model or os.environ.get("QUERY_AGENT_OPENAI_MODEL", "gpt-3.5-turbo"),
        openai_timeout=openai_timeout,
        enable_structured_logging=enable_structured_logging,
    )

    if args.clear_cache:
        agent.clear_cache()
        print("Caches cleared")
        return

    if args.list_tables:
        for table, count in agent.list_table_counts():
            if count >= 0:
                print(f"{table} ({count})")
            else:
                print(table)
        return

    if args.describe_table:
        for name, typ in agent.table_columns(args.describe_table):
            print(f"{name}: {typ}")
        return

    if args.execute_sql:
        result = agent.execute_sql(args.execute_sql, explain=args.explain)
        print(result.sql)
        if result.explanation:
            print(result.explanation)
        if result.data:
            print(result.data)
            if args.output_csv:
                save_csv(args.output_csv, result.data)
        return

    if args.interactive:
        try:
            while True:
                question = input("question> ")
                if question.strip().lower() in {"quit", "exit"}:
                    break
                if args.sql_only:
                    sql = agent.generate_sql(question)
                    print(sql)
                else:
                    result = agent.query(question, explain=args.explain)
                    print(result.sql)
                    if result.explanation:
                        print(result.explanation)
                    if result.data:
                        print(result.data)
                        if args.output_csv:
                            save_csv(args.output_csv, result.data)
        except KeyboardInterrupt:
            pass
    elif args.question:
        if args.sql_only:
            sql = agent.generate_sql(args.question)
            print(sql)
        else:
            result = agent.query(args.question, explain=args.explain)
            print(result.sql)
            if result.explanation:
                print(result.explanation)
            if result.data:
                print(result.data)
                if args.output_csv:
                    save_csv(args.output_csv, result.data)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
