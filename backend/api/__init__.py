"""
Voxify API Package
Flask REST API for the Voxify platform

This package provides the REST API endpoints for the Voxify platform,
including user authentication, voice sample management, and TTS synthesis.
"""

from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from flask_jwt_extended import JWTManager
import os


def create_app(test_config=None):
    """
    Create and configure the Flask application

    Parameters
    ----------
    test_config : dict, optional
        Test configuration to override default configuration

    Returns
    -------
    Flask
        Configured Flask application
    """

    # Load .env file from the backend directory (parent of api/)
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

    # Create Flask app
    app = Flask(__name__, instance_relative_config=True)

    # Default configuration
    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY", "Majick"),
        JWT_SECRET_KEY=os.getenv("JWT_SECRET_KEY", "Majick"),
        JWT_ACCESS_TOKEN_EXPIRES=int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES", 3600)),
        JWT_REFRESH_TOKEN_EXPIRES=int(os.getenv("JWT_REFRESH_TOKEN_EXPIRES", 2592000)),
        JWT_TOKEN_LOCATION=["headers"],
        JWT_HEADER_NAME="Authorization",
        JWT_HEADER_TYPE="Bearer",
        DATABASE_URL=os.getenv("DATABASE_URL", "sqlite:///data/voxify.db"),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # limit file upload size to 16MB
    )

    jwt = JWTManager(app)  # noqa: F841

    # Override with test config if provided
    if test_config is not None:
        app.config.from_mapping(test_config)

    # Configure CORS to allow frontend access
    CORS(
        app,
        origins=[
            "http://localhost:3000",  # React development server
            "http://127.0.0.1:3000",
            "http://localhost:3001",  # Alternative ports
            "http://127.0.0.1:3001",
            "https://voxify.vercel.app",  # Vercel production
            "https://*.vercel.app",  # All Vercel preview deployments
            "https://test-ho72cndbz-jun-yangs-projects-f7853876.vercel.app",
            "https://test-lemon-eight-27.vercel.app",
            "https://voxify-prod.vercel.app",  # Production Vercel deployment
            "https://voxify-dev.vercel.app",  # Development Vercel deployment
            "https://voxify-livid.vercel.app",
            "https://voxify-hea9ojq5p-jun-yangs-projects-f7853876.vercel.app",
        ],
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "Accept"],
        supports_credentials=True,
    )

    # Import blueprints
    from .v1.auth import auth_bp

    # from .v1.admin import admin_bp
    from .v1.voice import voice_bp
    from .v1.job import job_bp
    from .v1.file import file_bp

    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix="/api/v1/auth")
    # app.register_blueprint(admin_bp, url_prefix='/api/v1/admin')
    app.register_blueprint(voice_bp, url_prefix="/api/v1/voice")
    app.register_blueprint(job_bp, url_prefix="/api/v1/job")
    app.register_blueprint(file_bp, url_prefix="/api/v1/file")

    # Simple index route
    @app.route("/")
    def index():
        return {"message": "Welcome to Voxify API"}

    # Health check endpoint
    @app.route("/health")
    def health_check():
        return {"status": "healthy"}, 200

    return app
