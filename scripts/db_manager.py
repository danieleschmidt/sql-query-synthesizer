#!/usr/bin/env python3
"""Database management CLI for SQL Query Synthesizer."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sql_synthesizer.database.connection import ConnectionConfig, initialize_database
from sql_synthesizer.database.migrations import MigrationManager
from sql_synthesizer.database.repositories import (
    QueryHistoryRepository,
)


async def init_database(database_url: str):
    """Initialize database with schema."""
    print(f"Initializing database: {database_url}")

    config = ConnectionConfig(database_url)
    db_manager = await initialize_database(config)

    migration_manager = MigrationManager()

    # Apply all migrations
    success = await migration_manager.migrate_to_latest(db_manager)

    if success:
        print("✅ Database initialized successfully")

        # Show migration status
        status = await migration_manager.get_migration_status(db_manager)
        print(f"Applied {status['applied_count']} migrations")
        print(f"Current schema version: {status['current_version']}")
    else:
        print("❌ Database initialization failed")
        return False

    await db_manager.close()
    return True


async def create_migration(name: str, description: str = ""):
    """Create a new migration file."""
    migration_manager = MigrationManager()

    file_path = migration_manager.create_migration(name, description)
    print(f"✅ Created migration: {file_path}")

    return file_path


async def migration_status(database_url: str):
    """Show migration status."""
    print(f"Checking migration status for: {database_url}")

    config = ConnectionConfig(database_url)
    db_manager = await initialize_database(config)

    migration_manager = MigrationManager()
    status = await migration_manager.get_migration_status(db_manager)

    if 'error' in status:
        print(f"❌ Error: {status['error']}")
        await db_manager.close()
        return False

    print(f"Current schema version: {status['current_version']}")
    print(f"Total migrations: {status['total_migrations']}")
    print(f"Applied migrations: {status['applied_count']}")
    print(f"Pending migrations: {status['pending_count']}")

    if status['pending_migrations']:
        print("\nPending migrations:")
        for migration in status['pending_migrations']:
            print(f"  - {migration['version']}: {migration['name']}")

    if status['applied_migrations']:
        print("\nLast applied migration:")
        last_migration = status['applied_migrations'][-1]
        print(f"  {last_migration['version']}: {last_migration['name']}")

    await db_manager.close()
    return True


async def run_migrations(database_url: str):
    """Run pending migrations."""
    print(f"Running migrations for: {database_url}")

    config = ConnectionConfig(database_url)
    db_manager = await initialize_database(config)

    migration_manager = MigrationManager()

    # Check status first
    status = await migration_manager.get_migration_status(db_manager)

    if status['pending_count'] == 0:
        print("✅ No pending migrations")
        await db_manager.close()
        return True

    print(f"Found {status['pending_count']} pending migrations")

    # Apply migrations
    success = await migration_manager.migrate_to_latest(db_manager)

    if success:
        print("✅ All migrations applied successfully")
    else:
        print("❌ Migration failed")

    await db_manager.close()
    return success


async def health_check(database_url: str):
    """Check database health."""
    print(f"Checking database health: {database_url}")

    try:
        config = ConnectionConfig(database_url)
        db_manager = await initialize_database(config)

        health = await db_manager.health_check()
        stats = await db_manager.get_connection_stats()

        if health['healthy']:
            print("✅ Database is healthy")
            print(f"   Response time: {health['response_time_ms']}ms")
            print(f"   Pool size: {stats.pool_size}")
            print(f"   Active connections: {stats.checked_out}")
            print(f"   Health score: {stats.health_score:.2f}")
        else:
            print("❌ Database is unhealthy")
            if health.get('error'):
                print(f"   Error: {health['error']}")

        await db_manager.close()
        return health['healthy']

    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


async def query_stats(database_url: str):
    """Show query execution statistics."""
    print(f"Getting query statistics from: {database_url}")

    try:
        config = ConnectionConfig(database_url)
        db_manager = await initialize_database(config)

        repo = QueryHistoryRepository(db_manager)
        stats = await repo.get_query_statistics()

        if stats:
            print("📊 Query Statistics:")
            print(f"   Total queries: {stats.get('total_queries', 0):,}")
            print(f"   Successful: {stats.get('successful_queries', 0):,}")
            print(f"   Failed: {stats.get('failed_queries', 0):,}")
            print(f"   Success rate: {stats.get('success_rate', 0):.1f}%")
            print(f"   Cache hit rate: {stats.get('cache_hit_rate', 0):.1f}%")
            print(f"   Avg execution time: {stats.get('avg_execution_time_ms', 0):.1f}ms")
        else:
            print("📊 No query statistics available")

        await db_manager.close()
        return True

    except Exception as e:
        print(f"❌ Failed to get statistics: {e}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="SQL Query Synthesizer Database Manager")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Database URL argument for most commands
    db_arg_kwargs = {
        'help': 'Database URL (default: from DATABASE_URL env var)',
        'default': os.getenv('DATABASE_URL', 'sqlite:///./sql_synthesizer.db')
    }

    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize database schema')
    init_parser.add_argument('--database-url', **db_arg_kwargs)

    # Create migration command
    create_parser = subparsers.add_parser('create-migration', help='Create new migration')
    create_parser.add_argument('name', help='Migration name')
    create_parser.add_argument('--description', default='', help='Migration description')

    # Migration status command
    status_parser = subparsers.add_parser('status', help='Show migration status')
    status_parser.add_argument('--database-url', **db_arg_kwargs)

    # Run migrations command
    migrate_parser = subparsers.add_parser('migrate', help='Run pending migrations')
    migrate_parser.add_argument('--database-url', **db_arg_kwargs)

    # Health check command
    health_parser = subparsers.add_parser('health', help='Check database health')
    health_parser.add_argument('--database-url', **db_arg_kwargs)

    # Query statistics command
    stats_parser = subparsers.add_parser('stats', help='Show query statistics')
    stats_parser.add_argument('--database-url', **db_arg_kwargs)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == 'init':
            success = asyncio.run(init_database(args.database_url))
        elif args.command == 'create-migration':
            asyncio.run(create_migration(args.name, args.description))
            success = True
        elif args.command == 'status':
            success = asyncio.run(migration_status(args.database_url))
        elif args.command == 'migrate':
            success = asyncio.run(run_migrations(args.database_url))
        elif args.command == 'health':
            success = asyncio.run(health_check(args.database_url))
        elif args.command == 'stats':
            success = asyncio.run(query_stats(args.database_url))
        else:
            print(f"Unknown command: {args.command}")
            return 1

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
