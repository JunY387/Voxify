"""
Voxify API Package
Flask REST API for the Voxify platform

This package provides the REST API endpoints for the Voxify platform,
including user authentication, voice sample management, and TTS synthesis.
"""

from flask import Flask
from flask_cors import CORS
from flask_restx import Api
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv
from datetime import datetime

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
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

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

    # Initialize Swagger/OpenAPI documentation
    api = Api(
        app,
        version="1.0.0",
        title="Voxify API",
        description="""
        ## RESTful API for Voice Cloning and TTS Synthesis

        Voxify provides advanced voice cloning and text-to-speech synthesis capabilities powered by F5-TTS technology.

        ### Key Features
        - 🎙️ **Voice Sample Management**: Upload and process voice samples
        - 🧬 **Voice Cloning**: Create custom voice models from samples
        - 🗣️ **Speech Synthesis**: Generate speech using cloned voices
        - 📊 **Job Monitoring**: Track synthesis progress in real-time
        - 📁 **File Management**: Download and manage generated audio files

        ### Supported Languages
        - **Native Support** (🔵): Chinese (zh-CN, zh-TW), English (en-US, en-GB)
        - **Specialized Models** (🟡): Japanese, French, German, Spanish, Italian, Russian, Hindi
        - **Fallback Support** (🟠): Korean, Portuguese, Arabic, Thai, Vietnamese

        ### Authentication
        All endpoints (except public info) require JWT authentication. Use the `/auth/login` endpoint to obtain tokens.
        """,
        doc="/docs/",  # Swagger UI endpoint
        prefix="/api/v1",
        contact="Voxify API Team",
        contact_email="support@voxify.app",
        contact_url="https://voxify.app/support",
        license="MIT License",
        license_url="https://opensource.org/licenses/MIT",
        terms_url="https://voxify.app/terms",
        validate=True,  # Validate requests against schema
        ordered=True,  # Maintain operation order
        authorizations={
            "Bearer": {
                "type": "apiKey",
                "in": "header",
                "name": "Authorization",
                "description": "JWT token in format: Bearer {token}",
            }
        },
        security=["Bearer"],
    )

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
            "https://voxify-jun-self-deploy-version.vercel.app",  # Current Vercel deployment
        ],
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "Accept"],
        supports_credentials=True,
    )

    # Register legacy blueprints first (for actual functionality)
    from .v1.auth import auth_bp
    from .v1.voice import voice_bp
    from .v1.job import job_bp
    from .v1.file import file_bp

    # Register blueprints with their original functionality
    app.register_blueprint(auth_bp, url_prefix="/api/v1/auth")
    app.register_blueprint(voice_bp, url_prefix="/api/v1/voice")
    app.register_blueprint(job_bp, url_prefix="/api/v1/job")
    app.register_blueprint(file_bp, url_prefix="/api/v1/file")

    # Import and register Swagger namespaces with full functionality
    try:
        from .v1.auth.swagger_routes import auth_ns
        from .v1.voice.swagger_routes_full import voice_ns
        from .v1.job.swagger_routes_full import job_ns
        from .v1.file.swagger_routes_full import file_ns

        # Register namespaces with API (these provide both docs and functionality)
        api.add_namespace(auth_ns, path="/auth")
        api.add_namespace(voice_ns, path="/voice")
        api.add_namespace(job_ns, path="/job")
        api.add_namespace(file_ns, path="/file")

        print("✅ Swagger documentation with full functionality loaded successfully")
        print("📋 API endpoints: /api/v1/* | Swagger endpoints: /auth, /voice, /job, /file | Docs: /docs/")
    except ImportError as e:
        print(f"⚠️ Swagger routes not available: {e}")
        print("📋 Falling back to blueprint routes only")

    # Simple index route
    @app.route("/")
    def index():
        return {
            "message": "Welcome to Voxify API",
            "version": "1.0.0",
            "documentation": "/docs/",
            "health": "/health",
            "api_prefix": "/api/v1",
        }

    # Health check endpoint
    @app.route("/health")
    def health_check():
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "services": {"database": "operational", "f5_tts": "operational", "storage": "operational"},
        }, 200

    # API information endpoint
    @app.route("/api/info")
    def api_info():
        return {
            "api": {
                "name": "Voxify API",
                "version": "1.0.0",
                "description": "Voice cloning and TTS synthesis platform",
                "documentation": "/docs/",
                "openapi_spec": "/docs/swagger.json",
            },
            "features": {
                "voice_cloning": True,
                "tts_synthesis": True,
                "real_time_monitoring": True,
                "file_management": True,
                "multi_language_support": True,
            },
            "supported_formats": {"input": ["wav", "mp3"], "output": ["wav", "mp3", "flac", "ogg"]},
            "rate_limits": {"authenticated": "1000/hour", "anonymous": "100/hour"},
        }

    # Redirect old docs URLs
    @app.route("/api/docs")
    @app.route("/api/documentation")
    def redirect_to_docs():
        from flask import redirect

        return redirect("/docs/", code=301)

    return app
