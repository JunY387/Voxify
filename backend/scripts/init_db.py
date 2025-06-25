#!/usr/bin/env python3
"""
Voxify Database Initialization Script
Used to create database table structure and populate initial data.
"""

import os
import sys

# Add the project root directory to the Python path to import the database module
# This assumes init_db.py is in the backend directory, and the database module is in its subdirectory
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# If init_db.py is in the backend/scripts/ directory, adjust to SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Or, a more robust way is to ensure PYTHONPATH is set correctly when running this script, or run from the project root
# For the current structure (init_db.py in backend/ directory):
# sys.path.append(SCRIPT_DIR) 
# If database is a subdirectory of backend, the above sys.path.append(SCRIPT_DIR) is not needed

# Try to import the database initialization function
try:
    # Assume the database package is in backend/database/
    # And this script (init_db.py) is in the backend/ directory
    from database import initialize_database, get_database_manager, ChromaVectorDB
except ImportError as e:
    print(f"❌ Error: Could not import the database module: {e}")
    print("   Please ensure this script is run from the project's 'backend' directory,")
    print("   or that the Python path (PYTHONPATH) is set correctly to find the 'database' package.")
    print(f"   Current working directory: {os.getcwd()}")
    print(f"   Python search path: {sys.path}")
    sys.exit(1)

def main():
    """Execute the database initialization process"""
    print("🚀 Starting Voxify database initialization...")
    
    # Get database URL from environment variables, use default if not set
    # This is consistent with get_database_manager in models.py
    sqlite_db_url = os.getenv('DATABASE_URL', 'sqlite:///data/voxify.db')
    vector_db_path = os.getenv('VECTOR_DB_PATH', 'data/chroma_db')

    print(f"ℹ️  SQLite database location: {sqlite_db_url}")
    print(f"ℹ️  Vector database location: {vector_db_path}")

    # Ensure the data directory exists (SQLite and ChromaDB usually create files in this directory)
    try:
        os.makedirs("/data", exist_ok=True)
        print("✅ '/data' directory confirmed/created.")
    except OSError as e:
        print(f"⚠️  Could not create 'data' directory: {e}. This might not be an issue if database files are not in this directory.")

    try:
        print("🔧 Initializing SQLite database and ChromaDB...")
        db_manager, vector_db = initialize_database(
            database_url=sqlite_db_url,
            # vector_db_path parameter is not used in initialize_database, ChromaVectorDB uses its default or environment variable directly
        )
        print("✅ SQLite database tables created and initial data populated.")
        print(f"   - SQLite database engine: {db_manager.engine}")
        
        print("✅ ChromaDB vector database initialized.")
        print(f"   - ChromaDB client: {vector_db.client}")
        print(f"   - ChromaDB persistence directory: {vector_db.persist_directory}")
        active_collections = vector_db.client.list_collections()
        if active_collections:
            print("   - Active collections:")
            for coll in active_collections:
                print(f"     * {coll.name} (ID: {coll.id}, Count: {coll.count()})")
        else:
            print("   - No active vector collections currently (this might be normal if just initialized).")

        print("🎉 Database initialization successful!")
        
    except Exception as e:
        print(f"❌ A critical error occurred during database initialization: {e}")
        import traceback
        print("Detailed error information:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Ensure we are running from the backend directory, or the database package can be found
    # A simple check: see if the database subdirectory exists
    if not os.path.isdir(os.path.join(SCRIPT_DIR, "database")):
        print("❌ Error: This script does not seem to be in the correct 'backend' directory, or the 'database' subdirectory was not found.")
        print(f"   Script location: {SCRIPT_DIR}")
        print("   Please run this script from the 'backend' directory, e.g.: python init_db.py")
        # sys.exit(1) # Temporarily commented out to allow running from other locations, but import might fail

    main() 