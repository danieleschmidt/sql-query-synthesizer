"""Command-line interface for the :class:`QueryAgent`."""

import argparse
import csv
import os
import sys

import yaml
from dotenv import load_dotenv
from sqlalchemy.exc import DatabaseError, OperationalError

from sql_synthesizer.config import config as app_config
from sql_synthesizer.sync_query_agent import QueryAgent
from sql_synthesizer.user_experience import format_cli_error

# Import quantum components if available
try:
    from sql_synthesizer.quantum import QuantumSQLSynthesizer

    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False


def load_env_config(config: str, env: str) -> dict:
    """Load a single environment configuration from a YAML file."""
    if not os.path.exists(config):
        return {}
    with open(config) as fh:
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
    try:
        _main_impl(argv)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)  # Standard exit code for SIGINT
    except (ImportError, ModuleNotFoundError) as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError as e:
        print(f"Permission denied: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Invalid configuration: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(format_cli_error(e), file=sys.stderr)
        sys.exit(1)


def _main_impl(argv: list[str] | None = None) -> None:
    """Internal implementation of main function with error handling."""
    load_dotenv()

    # Enhanced description with examples
    description = """Interactive Natural Language to SQL Agent
    
Ask questions about your data in plain English and get SQL queries automatically generated.

Examples:
  %(prog)s 'How many users are there?'
  %(prog)s 'Show me orders from last week'
  %(prog)s --list-tables
  %(prog)s --interactive

Common question patterns:
  • How many [table] are there?
  • Show me all [table]
  • What are the latest [table]?
  • Find [table] with [condition]
"""

    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--database-url")
    parser.add_argument("--config")
    parser.add_argument("--env")
    parser.add_argument("--schema-cache-ttl", type=int)
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached schema and query results",
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--cache-ttl",
        type=int,
        help="Cache query results for the given number of seconds",
    )
    parser.add_argument("--describe-table")
    parser.add_argument(
        "--execute-sql", help="Execute raw SQL instead of generating from a question"
    )
    parser.add_argument("--output-csv", help="Write query results to a CSV file")
    parser.add_argument(
        "--openai-api-key", help="OpenAI API key for LLM-based generation"
    )
    parser.add_argument("--openai-model", help="OpenAI model to use for generation")
    parser.add_argument(
        "--openai-timeout",
        type=float,
        help="Timeout in seconds for OpenAI API requests",
    )
    parser.add_argument(
        "--explain", action="store_true", help="Display EXPLAIN plan instead of rows"
    )
    parser.add_argument(
        "--sql-only", action="store_true", help="Print generated SQL without executing"
    )
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--list-tables", action="store_true", help="List available tables and exit"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set log level",
    )
    parser.add_argument(
        "--log-format", choices=["standard", "json"], help="Set log format"
    )
    parser.add_argument(
        "--enable-structured-logging",
        action="store_true",
        help="Enable structured logging with trace IDs",
    )

    # Quantum optimization arguments
    if QUANTUM_AVAILABLE:
        parser.add_argument(
            "--enable-quantum",
            action="store_true",
            help="Enable quantum-inspired query optimization",
        )
        parser.add_argument(
            "--quantum-qubits",
            type=int,
            default=16,
            help="Number of qubits for quantum optimization",
        )
        parser.add_argument(
            "--quantum-temp",
            type=float,
            default=1000.0,
            help="Initial temperature for quantum annealing",
        )
        parser.add_argument(
            "--quantum-stats",
            action="store_true",
            help="Display quantum optimization statistics",
        )

    parser.add_argument("question", nargs="?")
    args = parser.parse_args(argv)

    config_path = args.config or os.environ.get(
        "QUERY_AGENT_CONFIG", "config/databases.yaml"
    )
    env_name = args.env or os.environ.get("QUERY_AGENT_ENV", "default")
    config = load_env_config(config_path, env_name)
    db_url = args.database_url or os.environ.get("DATABASE_URL") or config.get("url")
    schema_ttl = args.schema_cache_ttl
    if schema_ttl is None:
        env_ttl = os.environ.get("QUERY_AGENT_SCHEMA_CACHE_TTL")
        if env_ttl is not None:
            schema_ttl = int(env_ttl)
        else:
            schema_ttl = config.get("schema_cache_ttl", 0)
    max_rows = args.max_rows
    if max_rows is None:
        max_rows = int(
            os.environ.get("QUERY_AGENT_MAX_ROWS", app_config.default_max_rows)
        )
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
    enable_structured_logging = args.enable_structured_logging or os.environ.get(
        "QUERY_AGENT_STRUCTURED_LOGGING", ""
    ).lower() in ("true", "1", "yes")

    if enable_structured_logging or args.log_level or args.log_format:
        from sql_synthesizer.logging_utils import configure_logging

        configure_logging(
            level=args.log_level,
            format_type=args.log_format,
            enable_json=enable_structured_logging,
        )

    # Create base agent
    base_agent = QueryAgent(
        db_url,
        schema_cache_ttl=schema_ttl,
        max_rows=max_rows,
        query_cache_ttl=cache_ttl,
        openai_api_key=args.openai_api_key,
        openai_model=args.openai_model
        or os.environ.get("QUERY_AGENT_OPENAI_MODEL", "gpt-3.5-turbo"),
        openai_timeout=openai_timeout,
        enable_structured_logging=enable_structured_logging,
    )

    # Wrap with quantum optimization if enabled
    enable_quantum = QUANTUM_AVAILABLE and (
        getattr(args, "enable_quantum", False)
        or os.environ.get("QUERY_AGENT_ENABLE_QUANTUM", "").lower()
        in ("true", "1", "yes")
    )

    if enable_quantum:
        from sql_synthesizer.quantum.core import QuantumQueryOptimizer

        quantum_optimizer = QuantumQueryOptimizer(
            num_qubits=getattr(args, "quantum_qubits", 16),
            temperature=getattr(args, "quantum_temp", 1000.0),
        )
        agent = QuantumSQLSynthesizer(base_agent, enable_quantum=True)
        print("🚀 Quantum optimization enabled")
    else:
        agent = base_agent

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
        print("Interactive SQL Query Agent")
        print("Ask questions about your data in natural language.")
        print("Type 'quit' or 'exit' to leave, or press Ctrl+C.")
        print()

        try:
            while True:
                question = input("question> ")
                if question.strip().lower() in {"quit", "exit"}:
                    break

                try:
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

                        # Display quantum statistics if enabled
                        if enable_quantum and getattr(args, "quantum_stats", False):
                            if (
                                hasattr(result, "quantum_metrics")
                                and result.quantum_metrics
                            ):
                                print("\n🔬 Quantum Optimization Stats:")
                                for key, value in result.quantum_metrics.items():
                                    print(f"  {key}: {value}")
                                if hasattr(result, "quantum_cost_reduction"):
                                    print(
                                        f"  Cost Reduction: {result.quantum_cost_reduction:.1%}"
                                    )
                except DatabaseError as e:
                    print(f"Database error: {format_cli_error(e)}")
                    print()  # Add blank line for readability
                except OperationalError as e:
                    print(f"Database connection error: {format_cli_error(e)}")
                    print()  # Add blank line for readability
                except ValueError as e:
                    print(f"Invalid input: {format_cli_error(e)}")
                    print()  # Add blank line for readability
                except FileNotFoundError as e:
                    print(f"File not found: {format_cli_error(e)}")
                    print()  # Add blank line for readability
                except Exception as e:
                    print(format_cli_error(e))
                    print()  # Add blank line for readability

        except KeyboardInterrupt:
            print("\nGoodbye!")
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

            # Display quantum statistics if enabled
            if enable_quantum and getattr(args, "quantum_stats", False):
                if hasattr(result, "quantum_metrics") and result.quantum_metrics:
                    print("\n🔬 Quantum Optimization Stats:")
                    for key, value in result.quantum_metrics.items():
                        print(f"  {key}: {value}")
                    if hasattr(result, "quantum_cost_reduction"):
                        print(f"  Cost Reduction: {result.quantum_cost_reduction:.1%}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
