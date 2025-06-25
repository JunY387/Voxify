"""
File API Blueprint
Handles file upload, download, and management
"""

from flask import Blueprint

file_bp = Blueprint('file', __name__)

# Import routes after blueprint creation to avoid circular imports
from . import routes

__all__ = ["file_bp"] 