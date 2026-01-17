"""
Database Connection Management for Landseer Pipeline

Provides connection pooling and context management for MySQL database operations.
"""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass

import mysql.connector
from mysql.connector import pooling, Error as MySQLError

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "localhost"
    port: int = 3306
    user: str = "landseer"
    password: str = ""
    database: str = "landseer_pipeline"
    pool_name: str = "landseer_pool"
    pool_size: int = 5
    pool_reset_session: bool = True
    charset: str = "utf8mb4"
    collation: str = "utf8mb4_unicode_ci"
    autocommit: bool = False
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create configuration from environment variables."""
        return cls(
            host=os.getenv("LANDSEER_DB_HOST", "localhost"),
            port=int(os.getenv("LANDSEER_DB_PORT", "3306")),
            user=os.getenv("LANDSEER_DB_USER", "landseer"),
            password=os.getenv("LANDSEER_DB_PASSWORD", ""),
            database=os.getenv("LANDSEER_DB_NAME", "landseer_pipeline"),
            pool_size=int(os.getenv("LANDSEER_DB_POOL_SIZE", "5")),
        )
    
    def to_connection_dict(self) -> Dict[str, Any]:
        """Convert to mysql.connector connection parameters."""
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "charset": self.charset,
            "collation": self.collation,
            "autocommit": self.autocommit,
        }


class DatabaseConnection:
    """
    MySQL database connection manager with connection pooling.
    
    Usage:
        db = DatabaseConnection()
        db.connect()
        
        with db.get_cursor() as cursor:
            cursor.execute("SELECT * FROM pipeline_runs")
            results = cursor.fetchall()
        
        db.close()
    
    Or as context manager:
        with DatabaseConnection() as db:
            with db.get_cursor() as cursor:
                cursor.execute("SELECT * FROM datasets")
    """
    
    _instance: Optional["DatabaseConnection"] = None
    _pool: Optional[pooling.MySQLConnectionPool] = None
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """Initialize database connection manager."""
        self.config = config or DatabaseConfig.from_env()
        self._connection: Optional[mysql.connector.MySQLConnection] = None
    
    @classmethod
    def get_instance(cls, config: Optional[DatabaseConfig] = None) -> "DatabaseConnection":
        """Get singleton instance of DatabaseConnection."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance
    
    def connect(self) -> None:
        """Establish connection pool to the database."""
        if self._pool is not None:
            logger.debug("Connection pool already exists")
            return
        
        try:
            pool_config = self.config.to_connection_dict()
            pool_config["pool_name"] = self.config.pool_name
            pool_config["pool_size"] = self.config.pool_size
            pool_config["pool_reset_session"] = self.config.pool_reset_session
            
            self._pool = pooling.MySQLConnectionPool(**pool_config)
            logger.info(f"Database connection pool created: {self.config.host}:{self.config.port}/{self.config.database}")
            
        except MySQLError as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise
    
    def close(self) -> None:
        """Close all connections in the pool."""
        if self._pool is not None:
            # MySQL Connector/Python doesn't have a direct pool close method
            # Connections are returned to pool when released
            self._pool = None
            logger.info("Database connection pool closed")
    
    def get_connection(self) -> mysql.connector.MySQLConnection:
        """Get a connection from the pool."""
        if self._pool is None:
            self.connect()
        
        try:
            return self._pool.get_connection()
        except MySQLError as e:
            logger.error(f"Failed to get connection from pool: {e}")
            raise
    
    @contextmanager
    def get_cursor(self, dictionary: bool = True, buffered: bool = True):
        """
        Context manager for database cursor.
        
        Args:
            dictionary: If True, return results as dictionaries
            buffered: If True, fetch all results immediately
        
        Yields:
            MySQL cursor object
        """
        connection = self.get_connection()
        cursor = None
        try:
            cursor = connection.cursor(dictionary=dictionary, buffered=buffered)
            yield cursor
            connection.commit()
        except MySQLError as e:
            connection.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if cursor is not None:
                cursor.close()
            connection.close()  # Returns connection to pool
    
    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.
        
        Usage:
            with db.transaction() as (connection, cursor):
                cursor.execute("INSERT INTO ...")
                cursor.execute("UPDATE ...")
                # Commits on exit, rolls back on exception
        """
        connection = self.get_connection()
        cursor = None
        try:
            connection.autocommit = False
            cursor = connection.cursor(dictionary=True, buffered=True)
            yield connection, cursor
            connection.commit()
        except MySQLError as e:
            connection.rollback()
            logger.error(f"Transaction failed, rolled back: {e}")
            raise
        finally:
            if cursor is not None:
                cursor.close()
            connection.close()
    
    def execute(self, query: str, params: tuple = None) -> Optional[int]:
        """
        Execute a single query.
        
        Args:
            query: SQL query string
            params: Query parameters
        
        Returns:
            Last inserted ID for INSERT, or number of affected rows
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            if query.strip().upper().startswith("INSERT"):
                return cursor.lastrowid
            return cursor.rowcount
    
    def execute_many(self, query: str, params_list: list) -> int:
        """
        Execute a query with multiple parameter sets.
        
        Args:
            query: SQL query string
            params_list: List of parameter tuples
        
        Returns:
            Number of affected rows
        """
        with self.get_cursor() as cursor:
            cursor.executemany(query, params_list)
            return cursor.rowcount
    
    def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict[str, Any]]:
        """Fetch a single row as dictionary."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()
    
    def fetch_all(self, query: str, params: tuple = None) -> list:
        """Fetch all rows as list of dictionaries."""
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        query = """
            SELECT COUNT(*) as count
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        """
        result = self.fetch_one(query, (self.config.database, table_name))
        return result and result["count"] > 0
    
    def initialize_schema(self, schema_file: str) -> None:
        """
        Initialize database schema from SQL file.
        
        Args:
            schema_file: Path to SQL schema file
        """
        logger.info(f"Initializing database schema from {schema_file}")
        
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        # Split by delimiter changes and statement terminators
        # Handle DELIMITER changes for stored procedures
        statements = []
        current_delimiter = ";"
        current_statement = []
        
        lines = schema_sql.split('\n')
        for line in lines:
            stripped = line.strip()
            
            # Check for delimiter change
            if stripped.upper().startswith("DELIMITER"):
                parts = stripped.split()
                if len(parts) >= 2:
                    # Execute any pending statement
                    if current_statement:
                        stmt = '\n'.join(current_statement).strip()
                        if stmt:
                            statements.append(stmt)
                        current_statement = []
                    current_delimiter = parts[1]
                continue
            
            current_statement.append(line)
            
            # Check if statement ends
            if stripped.endswith(current_delimiter):
                stmt = '\n'.join(current_statement)
                # Remove the delimiter
                stmt = stmt[:stmt.rfind(current_delimiter)].strip()
                if stmt:
                    statements.append(stmt)
                current_statement = []
        
        # Add any remaining statement
        if current_statement:
            stmt = '\n'.join(current_statement).strip()
            if stmt:
                statements.append(stmt)
        
        # Execute statements
        connection = self.get_connection()
        cursor = connection.cursor()
        try:
            for stmt in statements:
                if stmt.strip():
                    try:
                        cursor.execute(stmt)
                        connection.commit()
                    except MySQLError as e:
                        if "already exists" not in str(e).lower():
                            logger.warning(f"Statement failed: {e}")
                        connection.rollback()
            
            logger.info("Database schema initialized successfully")
        finally:
            cursor.close()
            connection.close()
    
    def __enter__(self) -> "DatabaseConnection":
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


# Convenience function for getting database connection
def get_db_connection(config: Optional[DatabaseConfig] = None) -> DatabaseConnection:
    """Get the singleton database connection instance."""
    return DatabaseConnection.get_instance(config)
