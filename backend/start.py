#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voxify Startup Script
Automatically initializes database and starts Flask application server
"""

import os
import sys
import ssl
import argparse
from api import create_app

DEFAULT_F5_TTS_REMOTE_URL = (
    "https://nnnnnjun-yang--f5-tts-voxify-fastapi-app.modal.run/synthesize"
)
DEFAULT_VOICE_EMBED_REMOTE_URL = (
    "https://nnnnnjun-yang--voxify-voice-embed-fastapi-app.modal.run/extract_embedding"
)

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def init_file_storage():
    """Initialize file storage directories"""
    print("Initializing file storage...")

    # Define storage paths
    storage_paths = {
        "data": "data",
        "files": "data/files",
        "synthesis": "data/files/synthesis",
        "samples": "data/files/samples",
        "temp": "data/files/temp",
    }

    # Create directories
    for path_name, path in storage_paths.items():
        try:
            os.makedirs(path, exist_ok=True)
            print(f"[OK] {path_name} directory: {path}")
        except OSError as e:
            print(f"Error creating {path_name} directory ({path}): {e}")
            return False

    # Set environment variables for file storage
    os.environ["VOXIFY_FILE_STORAGE"] = storage_paths["files"]
    os.environ["VOXIFY_SYNTHESIS_STORAGE"] = storage_paths["synthesis"]
    os.environ["VOXIFY_SAMPLES_STORAGE"] = storage_paths["samples"]
    os.environ["VOXIFY_TEMP_STORAGE"] = storage_paths["temp"]

    print("File storage initialization successful!")
    return True


def configure_remote_f5_tts():
    """Ensure F5-TTS requests are routed to the Modal-hosted service by default."""
    os.environ.setdefault("F5_TTS_REMOTE_URL", DEFAULT_F5_TTS_REMOTE_URL)
    os.environ.setdefault("F5_TTS_USE_REMOTE", "true")


def configure_remote_voice_embed():
    """Ensure voice embedding requests are routed to the Modal-hosted service by default."""
    os.environ.setdefault("VOICE_EMBED_REMOTE_URL", DEFAULT_VOICE_EMBED_REMOTE_URL)
    os.environ.setdefault("VOICE_EMBED_USE_REMOTE", "true")
    print(f"Configured VOICE_EMBED_REMOTE_URL to: {os.getenv('VOICE_EMBED_REMOTE_URL')}")
    print(f"Configured VOICE_EMBED_USE_REMOTE to: {os.getenv('VOICE_EMBED_USE_REMOTE')}")


def init_database():
    """Initialize database"""
    print("Initializing database...")

    try:
        from database import initialize_database
    except ImportError as e:
        print(f"Error: Could not import database module: {e}")
        print("Please ensure this script is run from the project's 'backend' directory")
        return False

    # Get database URL from environment variables, use default if not set
    sqlite_db_url = os.getenv("DATABASE_URL", "sqlite:///data/voxify.db")
    vector_db_path = os.getenv("VECTOR_DB_PATH", "data/chroma_db")

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
        db_manager, vector_db = initialize_database(database_url=sqlite_db_url, vector_db_path=vector_db_path)
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
            create_test_users,
            create_voice_samples,
            create_voice_models,
            create_synthesis_jobs,
            create_usage_stats,
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
            print("- Usage Stats per User: 7 days")

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


def setup_ssl_context():
    """Setup SSL context for HTTPS"""
    # Check for SSL certificate environment variables first
    cert_file = os.getenv("SSL_CERT_FILE", "/etc/letsencrypt/live/milaniez-cheetah.duckdns.org/fullchain.pem")
    key_file = os.getenv("SSL_KEY_FILE", "/etc/letsencrypt/live/milaniez-cheetah.duckdns.org/privkey.pem")

    # Check if certificate files exist
    if not os.path.exists(cert_file):
        print(f"⚠️  SSL certificate file not found: {cert_file}")
        return None

    if not os.path.exists(key_file):
        print(f"⚠️  SSL private key file not found: {key_file}")
        return None

    try:
        # Create SSL context
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_file, key_file)
        print("🔒 SSL certificates loaded successfully")
        print(f"   - Certificate: {cert_file}")
        print(f"   - Private Key: {key_file}")
        return context
    except Exception as e:
        print(f"❌ Error loading SSL certificates: {e}")
        print("   Make sure the certificate files are readable by the current user")
        print("   You may need to run with sudo or copy certificates to a user-accessible location")
        return None


def start_flask_app(skip_db_init=False, skip_file_init=False, seed_data=False, use_https=False):
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

    # Configure remote Modal services by default
    configure_remote_f5_tts()
    configure_remote_voice_embed()

    # Create Flask app
    app = create_app()

    # Get configuration from environment (cloud platform compatible)
    host = os.getenv("FLASK_HOST", "0.0.0.0")  # Bind to all interfaces for cloud deployment
    port = int(os.getenv("PORT", os.getenv("FLASK_PORT", 8000)))  # Use PORT for cloud platforms
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    # Setup SSL if requested
    ssl_context = None
    protocol = "http"
    if use_https:
        ssl_context = setup_ssl_context()
        if ssl_context:
            protocol = "https"
        else:
            print("🔄 Falling back to HTTP due to SSL setup issues")

    print("Starting Voxify API Server...")
    print(f"Server: {protocol}://{host}:{port}")
    print(f"Auth endpoints: {protocol}://{host}:{port}/api/v1/auth")
    print(f"Voice endpoints: {protocol}://{host}:{port}/api/v1/voice")
    print(f"Job endpoints: {protocol}://{host}:{port}/api/v1/jobs")
    print(f"File endpoints: {protocol}://{host}:{port}/api/v1/file")

    if use_https and ssl_context:
        print("🔒 HTTPS encryption enabled")
    elif use_https:
        print("⚠️  HTTPS requested but SSL setup failed - running HTTP")

    print("\nServer started! Press Ctrl+C to stop the server")
    print("=" * 50)

    # Run the Flask app
    try:
        # Check if we're in production environment
        is_production = os.getenv("FLASK_ENV") == "production"

        if is_production:
            # Use production-ready server (if available)
            try:
                from waitress import serve

                print("🚀 Starting with Waitress WSGI server (production)")
                if ssl_context:
                    # Note: Waitress doesn't directly support SSL context
                    # In production, you'd typically use a reverse proxy (nginx) for SSL
                    print("⚠️  Production mode with Waitress - SSL should be handled by reverse proxy")
                    serve(app, host=host, port=port, threads=4)
                else:
                    serve(app, host=host, port=port, threads=4)
            except ImportError:
                print("⚠️  Waitress not available, falling back to Flask dev server")
                app.run(host=host, port=port, debug=False, threaded=True, ssl_context=ssl_context)
        else:
            # Development server
            print("🔧 Starting with Flask development server")
            app.run(host=host, port=port, debug=debug, threaded=True, ssl_context=ssl_context)

    except KeyboardInterrupt:
        print("\n\nServer stopped")
    except Exception as e:
        print(f"\nError occurred while running server: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Voxify One-Click Startup Script")
    parser.add_argument(
        "--skip-db-init",
        action="store_true",
        help="Skip database initialization, start server directly",
    )
    parser.add_argument("--skip-file-init", action="store_true", help="Skip file storage initialization")
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Initialize database and file storage only, do not start server",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed database with test data during startup",
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Only seed the database with test data, do not start server",
    )
    parser.add_argument(
        "--https",
        action="store_true",
        help="Enable HTTPS using SSL certificates",
    )

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
            seed_data=args.seed,
            use_https=args.https,
        )


if __name__ == "__main__":
    main()
