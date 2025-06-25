#!/usr/bin/env python3
"""
Voxify Startup Script
Automatically initializes database and starts Flask application server
"""

import os
import sys
import argparse
from api import create_app

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def init_file_storage():
    """Initialize file storage directories"""
    print("Initializing file storage...")
    
    # Define storage paths
    storage_paths = {
        'data': 'data',
        'files': 'data/files',
        'synthesis': 'data/files/synthesis',
        'samples': 'data/files/samples',
        'temp': 'data/files/temp'
    }
    
    # Create directories
    for path_name, path in storage_paths.items():
        try:
            os.makedirs(path, exist_ok=True)
            print(f"✓ {path_name} directory: {path}")
        except OSError as e:
            print(f"Error creating {path_name} directory ({path}): {e}")
            return False
    
    # Set environment variables for file storage
    os.environ['VOXIFY_FILE_STORAGE'] = storage_paths['files']
    os.environ['VOXIFY_SYNTHESIS_STORAGE'] = storage_paths['synthesis']
    os.environ['VOXIFY_SAMPLES_STORAGE'] = storage_paths['samples']
    os.environ['VOXIFY_TEMP_STORAGE'] = storage_paths['temp']
    
    print("File storage initialization successful!")
    return True

def init_database():
    """Initialize database"""
    print("Initializing database...")
    
    try:
        from database import initialize_database, get_database_manager, ChromaVectorDB
    except ImportError as e:
        print(f"Error: Could not import database module: {e}")
        print("Please ensure this script is run from the project's 'backend' directory")
        return False

    # Get database URL from environment variables, use default if not set
    sqlite_db_url = os.getenv('DATABASE_URL', 'sqlite:///data/voxify.db')
    vector_db_path = os.getenv('VECTOR_DB_PATH', 'data/chroma_db')

    print(f"SQLite database location: {sqlite_db_url}")
    print(f"Vector database location: {vector_db_path}")

    # Ensure the data directory exists
    try:
        os.makedirs("data", exist_ok=True)
        print("Data directory confirmed/created")
    except OSError as e:
        print(f"Could not create 'data' directory: {e}")
        return False

    try:
        print("Initializing SQLite database and ChromaDB...")
        db_manager, vector_db = initialize_database(
            database_url=sqlite_db_url,
            vector_db_path=vector_db_path
        )
        print("SQLite database tables created and initial data populated")
        print(f"   - SQLite database engine: {db_manager.engine}")
        
        print("ChromaDB vector database initialized")
        print(f"   - ChromaDB client: {vector_db.client}")
        print(f"   - ChromaDB persistence directory: {vector_db.persist_directory}")
        
        active_collections = vector_db.client.list_collections()
        if active_collections:
            print("   - Active collections:")
            for coll in active_collections:
                print(f"     * {coll.name} (ID: {coll.id}, Count: {coll.count()})")
        else:
            print("   - No active vector collections (normal after initialization)")

        print("Database initialization successful!")
        return True
        
    except Exception as e:
        print(f"Critical error during database initialization: {e}")
        import traceback
        print("Detailed error information:")
        traceback.print_exc()
        return False

def seed_database():
    """Seed database with test data"""
    print("Seeding database with test data...")
    
    try:
        # Import seed script functions
        from scripts.seed_db import (
            create_test_users, create_voice_samples, create_voice_models,
            create_synthesis_jobs, create_usage_stats
        )
        from database import get_database_manager
        
        # Get database connection
        db = get_database_manager()
        session = db.get_session()
        
        try:
            # Create test data
            print("👤 Creating test users...")
            users = create_test_users(session)
            session.commit()
            
            print("🎤 Creating voice samples...")
            samples = create_voice_samples(session, users)
            session.commit()
            
            print("🤖 Creating voice models...")
            models = create_voice_models(session, samples)
            session.commit()
            
            print("🎯 Creating synthesis jobs...")
            jobs = create_synthesis_jobs(session, users, models)
            session.commit()
            
            print("📊 Creating usage statistics...")
            create_usage_stats(session, users)
            session.commit()
            
            print("✅ Test data creation successful!")
            
            # Print summary
            print("\n📝 Data Summary:")
            print(f"- Users: {len(users)}")
            print(f"- Voice Samples: {len(samples)}")
            print(f"- Voice Models: {len(models)}")
            print(f"- Synthesis Jobs: {len(jobs)}")
            print(f"- Usage Stats per User: 7 days")
            
            return True
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error during seeding: {str(e)}")
            return False
        finally:
            session.close()
            
    except ImportError as e:
        print(f"Error: Could not import seed module: {e}")
        return False
    except Exception as e:
        print(f"Error during database seeding: {e}")
        return False

def start_flask_app(skip_db_init=False, skip_file_init=False, seed_data=False):
    """Start Flask application"""
    
    # Initialize file storage (unless skipped)
    if not skip_file_init:
        if not init_file_storage():
            print("File storage initialization failed, cannot start application")
            return
        print()  # Empty line separator
    
    # Initialize database (unless skipped)
    if not skip_db_init:
        if not init_database():
            print("Database initialization failed, cannot start application")
            return
        print()  # Empty line separator
    
    # Seed database if requested
    if seed_data:
        if not seed_database():
            print("Database seeding failed, but continuing with server startup")
        print()  # Empty line separator
    
    # Create Flask app
    app = create_app()
    
    # Get configuration from environment
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"Starting Voxify API Server...")
    print(f"Server: http://{host}:{port}")
    print(f"Auth endpoints: http://{host}:{port}/api/v1/auth")
    print(f"Voice endpoints: http://{host}:{port}/api/v1/voice")
    print(f"Job endpoints: http://{host}:{port}/api/v1/jobs")
    print(f"File endpoints: http://{host}:{port}/api/v1/file")
    print("\nServer started! Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Run the Flask app
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped")
    except Exception as e:
        print(f"\nError occurred while running server: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Voxify One-Click Startup Script')
    parser.add_argument('--skip-db-init', action='store_true', 
                       help='Skip database initialization, start server directly')
    parser.add_argument('--skip-file-init', action='store_true',
                       help='Skip file storage initialization')
    parser.add_argument('--init-only', action='store_true',
                       help='Initialize database and file storage only, do not start server')
    parser.add_argument('--seed', action='store_true',
                       help='Seed database with test data during startup')
    parser.add_argument('--seed-only', action='store_true',
                       help='Only seed the database with test data, do not start server')
    
    args = parser.parse_args()
    
    print("Welcome to Voxify!")
    print("=" * 50)
    
    if args.seed_only:
        # Only seed the database
        if not seed_database():
            print("Database seeding failed")
            sys.exit(1)
        else:
            print("Database seeding completed successfully")
    elif args.init_only:
        # Initialize database and file storage only
        success = True
        if not args.skip_file_init:
            success = success and init_file_storage()
        if not args.skip_db_init:
            success = success and init_database()
        if args.seed:
            success = success and seed_database()
        
        if success:
            print("Initialization completed successfully")
        else:
            print("Initialization failed")
            sys.exit(1)
    else:
        # Start complete application
        start_flask_app(
            skip_db_init=args.skip_db_init,
            skip_file_init=args.skip_file_init,
            seed_data=args.seed
        )

if __name__ == '__main__':
    main() 