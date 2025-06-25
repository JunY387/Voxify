"""
Voice API Blueprint
Handles voice sample management and cloning
"""

from flask import Blueprint

# Create the blueprint
voice_bp = Blueprint('voice', __name__, url_prefix='/api/v1/voice')

# Import routes to register them with the blueprint
from . import samples
from . import clones 