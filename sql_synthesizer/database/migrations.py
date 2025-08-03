"""Database migration management system."""

import logging
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """Represents a database migration."""
    version: str
    name: str
    description: str
    up_sql: str
    down_sql: str
    created_at: datetime
    
    @property
    def filename(self) -> str:
        """Generate filename for this migration."""
        return f"{self.version}_{self.name.replace(' ', '_').lower()}.sql"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert migration to dictionary."""
        return {
            'version': self.version,
            'name': self.name,
            'description': self.description,
            'up_sql': self.up_sql,
            'down_sql': self.down_sql,
            'created_at': self.created_at.isoformat()
        }


class MigrationManager:
    """Manages database migrations and schema versions."""
    
    def __init__(self, migrations_path: str = "sql_synthesizer/database/migrations"):
        """Initialize migration manager."""
        self.migrations_path = Path(migrations_path)
        self.migrations_path.mkdir(parents=True, exist_ok=True)
        self._migrations: List[Migration] = []
        self._loaded = False
    
    def load_migrations(self) -> None:
        """Load all migrations from the migrations directory."""
        if self._loaded:
            return
        
        self._migrations = []
        
        # Load migration files
        for migration_file in sorted(self.migrations_path.glob("*.sql")):
            try:
                migration = self._parse_migration_file(migration_file)
                if migration:
                    self._migrations.append(migration)
            except Exception as e:
                logger.error(f"Failed to parse migration {migration_file}: {e}")
        
        self._loaded = True
        logger.info(f"Loaded {len(self._migrations)} migrations")
    
    def _parse_migration_file(self, file_path: Path) -> Optional[Migration]:
        """Parse a migration file and return Migration object."""
        content = file_path.read_text(encoding='utf-8')
        
        # Extract metadata from comments
        version_match = re.search(r'-- Version:\s*(\S+)', content)
        name_match = re.search(r'-- Name:\s*(.+)', content)
        desc_match = re.search(r'-- Description:\s*(.+)', content)
        
        if not all([version_match, name_match]):
            logger.warning(f"Migration {file_path} missing required metadata")
            return None
        
        # Split UP and DOWN sections
        up_match = re.search(r'-- UP\s*\n(.*?)(?=-- DOWN|\Z)', content, re.DOTALL)
        down_match = re.search(r'-- DOWN\s*\n(.*)', content, re.DOTALL)
        
        if not up_match:
            logger.warning(f"Migration {file_path} missing UP section")
            return None
        
        return Migration(
            version=version_match.group(1),
            name=name_match.group(1).strip(),
            description=desc_match.group(1).strip() if desc_match else "",
            up_sql=up_match.group(1).strip(),
            down_sql=down_match.group(1).strip() if down_match else "",
            created_at=datetime.fromtimestamp(file_path.stat().st_mtime)
        )
    
    def create_migration(self, name: str, description: str = "") -> str:
        """Create a new migration file template."""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{version}_{name.replace(' ', '_').lower()}.sql"
        file_path = self.migrations_path / filename
        
        template = f"""-- Version: {version}
-- Name: {name}
-- Description: {description}
-- Created: {datetime.now().isoformat()}

-- UP
-- Add your schema changes here


-- DOWN
-- Add rollback statements here

"""
        
        file_path.write_text(template, encoding='utf-8')
        logger.info(f"Created migration: {filename}")
        
        return str(file_path)
    
    def get_migrations(self) -> List[Migration]:
        """Get all loaded migrations sorted by version."""
        if not self._loaded:
            self.load_migrations()
        
        return sorted(self._migrations, key=lambda m: m.version)
    
    def get_migration(self, version: str) -> Optional[Migration]:
        """Get a specific migration by version."""
        migrations = self.get_migrations()
        return next((m for m in migrations if m.version == version), None)
    
    async def get_schema_version(self, db_manager) -> Optional[str]:
        """Get current schema version from database."""
        try:
            # Create schema_migrations table if it doesn't exist
            await self._ensure_schema_table(db_manager)
            
            # Get latest version
            result = await db_manager.execute_query(
                "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
            )
            
            if result:
                return result[0][0]
            return None
            
        except Exception as e:
            logger.error(f"Failed to get schema version: {e}")
            return None
    
    async def _ensure_schema_table(self, db_manager) -> None:
        """Ensure schema_migrations table exists."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version VARCHAR(20) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        await db_manager.execute_query(create_table_sql)
    
    async def apply_migration(self, db_manager, migration: Migration) -> bool:
        """Apply a single migration to the database."""
        try:
            logger.info(f"Applying migration {migration.version}: {migration.name}")
            
            # Execute migration SQL
            if migration.up_sql.strip():
                # Split by semicolon and execute each statement
                statements = [s.strip() for s in migration.up_sql.split(';') if s.strip()]
                for statement in statements:
                    await db_manager.execute_query(statement)
            
            # Record migration in schema table
            await db_manager.execute_query(
                """
                INSERT INTO schema_migrations (version, name, description)
                VALUES (:version, :name, :description)
                """,
                {
                    'version': migration.version,
                    'name': migration.name,
                    'description': migration.description
                }
            )
            
            logger.info(f"Successfully applied migration {migration.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply migration {migration.version}: {e}")
            return False
    
    async def rollback_migration(self, db_manager, migration: Migration) -> bool:
        """Rollback a single migration."""
        try:
            logger.info(f"Rolling back migration {migration.version}: {migration.name}")
            
            # Execute rollback SQL
            if migration.down_sql.strip():
                statements = [s.strip() for s in migration.down_sql.split(';') if s.strip()]
                for statement in statements:
                    await db_manager.execute_query(statement)
            
            # Remove migration record
            await db_manager.execute_query(
                "DELETE FROM schema_migrations WHERE version = :version",
                {'version': migration.version}
            )
            
            logger.info(f"Successfully rolled back migration {migration.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback migration {migration.version}: {e}")
            return False
    
    async def migrate_to_latest(self, db_manager) -> bool:
        """Apply all pending migrations."""
        try:
            current_version = await self.get_schema_version(db_manager)
            migrations = self.get_migrations()
            
            pending_migrations = []
            for migration in migrations:
                if not current_version or migration.version > current_version:
                    pending_migrations.append(migration)
            
            if not pending_migrations:
                logger.info("No pending migrations")
                return True
            
            logger.info(f"Applying {len(pending_migrations)} pending migrations")
            
            for migration in pending_migrations:
                success = await self.apply_migration(db_manager, migration)
                if not success:
                    logger.error(f"Migration failed at {migration.version}")
                    return False
            
            logger.info("All migrations applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Migration process failed: {e}")
            return False
    
    async def get_migration_status(self, db_manager) -> Dict[str, Any]:
        """Get current migration status."""
        try:
            current_version = await self.get_schema_version(db_manager)
            migrations = self.get_migrations()
            
            applied_migrations = []
            pending_migrations = []
            
            for migration in migrations:
                if not current_version or migration.version <= current_version:
                    applied_migrations.append(migration)
                else:
                    pending_migrations.append(migration)
            
            return {
                'current_version': current_version,
                'total_migrations': len(migrations),
                'applied_count': len(applied_migrations),
                'pending_count': len(pending_migrations),
                'applied_migrations': [m.to_dict() for m in applied_migrations],
                'pending_migrations': [m.to_dict() for m in pending_migrations]
            }
            
        except Exception as e:
            logger.error(f"Failed to get migration status: {e}")
            return {
                'current_version': None,
                'error': str(e)
            }