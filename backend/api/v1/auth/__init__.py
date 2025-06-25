"""
Authentication API Blueprint
Handles user registration, login, and token management
"""

from flask import Blueprint

auth_bp = Blueprint('auth', __name__)

# Import routes after blueprint creation to avoid circular imports
from . import routes