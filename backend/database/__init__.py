"""
Voxify Database Package
Hybrid Storage System: SQLite + Chroma Vector Database

This package provides the complete database layer for the Voxify platform,
including relational data management and vector similarity search capabilities.
"""

from .models import (
    # Database Manager
    DatabaseManager,
    get_database_manager,
    
    # ORM Models
    Base,
    User,
    VoiceSample,
    VoiceModel,
    SynthesisJob,
    SynthesisCache,
    PhonemeAlignment,
    UsageStat,
    SystemSetting,
    SchemaVersion,
    
    # Utility functions
    generate_uuid,
    TimestampMixin
)

from .vector_config import (
    # Vector Database
    ChromaVectorDB,
    VectorDBConfig,
    create_vector_db
)

# Package version
__version__ = "1.0.0"

# Main components for easy import
__all__ = [
    # Database Management
    "DatabaseManager",
    "get_database_manager",
    "ChromaVectorDB", 
    "create_vector_db",
    
    # ORM Models
    "Base",
    "User",
    "VoiceSample", 
    "VoiceModel",
    "SynthesisJob",
    "SynthesisCache",
    "PhonemeAlignment",
    "UsageStat",
    "SystemSetting",
    "SchemaVersion",
    
    # Configuration
    "VectorDBConfig",
    
    # Utilities
    "generate_uuid",
    "TimestampMixin"
]

# Quick setup function
def initialize_database(database_url: str = None, vector_db_path: str = None):
    """
    Initialize both SQLite and Vector databases
    
    Parameters
    ----------
    database_url : str, optional
        SQLite database URL (default: sqlite:///data/voxify.db)
    vector_db_path : str, optional  
        Vector database path (default: data/chroma_db)
        
    Returns
    -------
    tuple
        (DatabaseManager, ChromaVectorDB) instances
    """
    if not database_url:
        import os
        database_url = os.getenv('DATABASE_URL', 'sqlite:///data/voxify.db')
        print("Using DATABASE_URL:", database_url)

    # Initialize SQLite database
    db_manager = get_database_manager(database_url)
    db_manager.create_tables()
    db_manager.init_default_data()
    # Initialize vector database
    vector_db = create_vector_db()
    
    return db_manager, vector_db 