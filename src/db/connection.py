"""
Database connection and session management for Landseer.

Provides SQLAlchemy database connection with support for MySQL and SQLite.
Uses connection pooling and session management for efficient database access.
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ..common import get_logger
from .models import Base

logger = get_logger(__name__)

# Load environment variables
load_dotenv()


@dataclass
class DatabaseConfig:
    """
    Database configuration.
    
    Supports MySQL (production) and SQLite (development/testing).
    """
    # Connection type: 'mysql' or 'sqlite'
    db_type: str = field(default_factory=lambda: os.getenv("LANDSEER_DB_TYPE", "sqlite"))
    
    # MySQL settings
    mysql_host: str = field(default_factory=lambda: os.getenv("LANDSEER_DB_HOST", "localhost"))
    mysql_port: int = field(default_factory=lambda: int(os.getenv("LANDSEER_DB_PORT", "3306")))
    mysql_user: str = field(default_factory=lambda: os.getenv("LANDSEER_DB_USER", "landseer"))
    mysql_password: str = field(default_factory=lambda: os.getenv("LANDSEER_DB_PASSWORD", ""))
    mysql_database: str = field(default_factory=lambda: os.getenv("LANDSEER_DB_NAME", "landseer"))
    
    # SQLite settings
    sqlite_path: str = field(default_factory=lambda: os.getenv("LANDSEER_DB_PATH", "landseer.db"))
    
    # Connection pool settings
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 1800  # 30 minutes
    
    # Echo SQL statements (for debugging)
    echo: bool = field(default_factory=lambda: os.getenv("LANDSEER_DB_ECHO", "false").lower() == "true")
    
    def get_database_url(self) -> str:
        """
        Get the SQLAlchemy database URL.
        
        Returns:
            Database URL string
        """
        if self.db_type == "mysql":
            return (
                f"mysql+mysqlconnector://{self.mysql_user}:{self.mysql_password}"
                f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
            )
        elif self.db_type == "sqlite":
            return f"sqlite:///{self.sqlite_path}"
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")


class Database:
    """
    Database connection manager.
    
    Provides:
    - Connection pooling
    - Session management
    - Schema creation/migration
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize database connection.
        
        Args:
            config: Database configuration (uses defaults if not provided)
        """
        self.config = config or DatabaseConfig()
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._initialized = False
    
    @property
    def engine(self) -> Engine:
        """Get the SQLAlchemy engine, creating it if necessary."""
        if self._engine is None:
            self._create_engine()
        return self._engine
    
    @property
    def session_factory(self) -> sessionmaker:
        """Get the session factory, creating it if necessary."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
        return self._session_factory
    
    def _create_engine(self) -> None:
        """Create the SQLAlchemy engine."""
        url = self.config.get_database_url()
        
        if self.config.db_type == "sqlite":
            # SQLite-specific settings
            self._engine = create_engine(
                url,
                echo=self.config.echo,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool
            )
            
            # Enable foreign keys for SQLite
            @event.listens_for(self._engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        else:
            # MySQL settings
            self._engine = create_engine(
                url,
                echo=self.config.echo,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=True  # Verify connection health
            )
        
        logger.info(f"Database engine created: {self.config.db_type}")
    
    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        self._initialized = True
        logger.info("Database tables created")
    
    def drop_tables(self) -> None:
        """Drop all database tables."""
        Base.metadata.drop_all(bind=self.engine)
        logger.info("Database tables dropped")
    
    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.
        
        Automatically commits on success and rolls back on exception.
        
        Yields:
            Database session
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_session(self) -> Session:
        """
        Get a new database session.
        
        Caller is responsible for committing/closing the session.
        
        Returns:
            New database session
        """
        return self.session_factory()
    
    def is_initialized(self) -> bool:
        """Check if database is initialized."""
        return self._initialized
    
    def close(self) -> None:
        """Close the database connection."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database connection closed")


# Global database instance
_database: Optional[Database] = None


def get_database() -> Database:
    """
    Get the global database instance.
    
    Returns:
        Database instance
        
    Raises:
        RuntimeError: If database not initialized
    """
    global _database
    if _database is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _database


def init_database(config: Optional[DatabaseConfig] = None, create_tables: bool = True) -> Database:
    """
    Initialize the global database instance.
    
    Args:
        config: Database configuration
        create_tables: Whether to create tables on initialization
        
    Returns:
        Database instance
    """
    global _database
    
    _database = Database(config)
    
    if create_tables:
        _database.create_tables()
    
    logger.info("Database initialized successfully")
    return _database


def get_session() -> Session:
    """
    Get a new database session from the global database.
    
    Returns:
        New database session
    """
    return get_database().get_session()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Context manager for database sessions using the global database.
    
    Yields:
        Database session
    """
    with get_database().session() as session:
        yield session
