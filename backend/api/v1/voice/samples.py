"""
Voice Sample Management Routes
Handles voice sample upload, processing, and management
"""

import os
import uuid
import tempfile
import soundfile as sf
from pathlib import Path
from datetime import datetime, timezone
from flask import request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from database import get_database_manager
from database.models import VoiceSample
from .embeddings import generate_voice_embedding, delete_voice_embedding

# Import the blueprint from __init__.py
from . import voice_bp

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio_metadata(file_path: str) -> dict:
    """Extract metadata from audio file."""
    with sf.SoundFile(file_path) as audio_file:
        return {
            'duration': len(audio_file) / audio_file.samplerate,
            'sample_rate': audio_file.samplerate,
            'channels': audio_file.channels,
            'format': audio_file.format
        }

@voice_bp.route('/samples', methods=['POST'])
@jwt_required()
def upload_voice_sample():
    """
    Upload and process a voice sample.
    
    Request:
        - name: Sample name (required)
        - file: Audio file (required, WAV or MP3)
    
    Returns:
        JSON response with sample_id and processing status
    """
    # Get the current user's ID
    user_id = get_jwt_identity()
    
    print(f"[DEBUG] Voice sample upload request from user: {user_id}")
    print(f"[DEBUG] Request files: {request.files}")
    print(f"[DEBUG] Request form: {request.form}")
    
    # Validate name parameter
    name = request.form.get('name')
    print(f"[DEBUG] Extracted name from form: '{name}'")
    if not name or not name.strip():
        return jsonify({
            'success': False,
            'error': 'Name is required for voice samples'
        }), 400
    
    # Validate file
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided'
        }), 400
    
    file = request.files['file']
    print(f"[DEBUG] Uploaded file: {file.filename if file else 'None'}")
    if not file or not file.filename or not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid file type. Only WAV and MP3 files are allowed'
        }), 400
    
    try:
        # Generate unique sample ID
        sample_id = str(uuid.uuid4())
        
        # Create permanent storage directory
        storage_dir = Path(f"data/files/samples/{user_id}")
        storage_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Created storage directory: {storage_dir}")
        
        # Define permanent file path
        file_extension = Path(file.filename).suffix.lower() or '.wav'
        permanent_path = storage_dir / f"{sample_id}{file_extension}"
        print(f"[DEBUG] Saving file to: {permanent_path}")
        
        # Save file to permanent location
        file.save(str(permanent_path))
        print(f"[DEBUG] File saved successfully, size: {os.path.getsize(str(permanent_path))} bytes")
        
        # Extract audio metadata
        metadata = extract_audio_metadata(str(permanent_path))
        
        # Generate voice embedding
        embedding_id, embedding = generate_voice_embedding(str(permanent_path))
        
        # Get database session
        db = get_database_manager()
        
        # Store metadata in SQLite
        with db.get_session() as session:
            voice_sample = VoiceSample(
                id=sample_id,
                name=name,
                user_id=user_id,
                file_path=str(permanent_path),  # Permanent path
                file_size=os.path.getsize(str(permanent_path)),
                original_filename=file.filename,
                format=metadata['format'],
                duration=metadata['duration'],
                sample_rate=metadata['sample_rate'],
                channels=metadata['channels'],
                status='ready',  # Changed from 'uploaded' to 'ready' since we generate embedding immediately
                processing_start_time=datetime.now(timezone.utc),
                processing_end_time=datetime.now(timezone.utc),
                voice_embedding_id=embedding_id  # Store the embedding ID
            )
            session.add(voice_sample)
            session.commit()
        
        return jsonify({
            'success': True,
            'data': {
                'sample_id': sample_id,
                'name': name,
                'duration': metadata['duration'],
                'format': metadata['format'],
                'status': 'ready',
                'message': 'Voice sample uploaded and processed successfully.'
            }
        }), 201
        
    except Exception as e:
        # Clean up permanent file if it exists and was created
        if 'permanent_path' in locals() and permanent_path.exists():
            try:
                permanent_path.unlink()
            except:
                pass
        
        # Clean up embedding if it was created
        if 'embedding_id' in locals():
            try:
                delete_voice_embedding(embedding_id)
            except:
                pass
        
        return jsonify({
            'success': False,
            'error': f'Error processing voice sample: {str(e)}'
        }), 500

@voice_bp.route('/samples', methods=['GET'])
@jwt_required()
def list_voice_samples():
    """
    List all voice samples for the authenticated user.
    
    Query Parameters:
        - page: Page number (default: 1)
        - page_size: Items per page (default: 20)
        - status: Filter by status (optional)
    
    Returns:
        JSON response with list of voice samples and pagination info
    """
    # Get the current user's ID
    user_id = get_jwt_identity()
    
    page = request.args.get('page', 1, type=int)
    page_size = request.args.get('page_size', 20, type=int)
    status = request.args.get('status')
    
    db = get_database_manager()
    with db.get_session() as session:
        query = session.query(VoiceSample).filter_by(user_id=user_id)
        
        # Filter by status if provided
        if status:
            query = query.filter_by(status=status)
        
        samples = query.order_by(VoiceSample.created_at.desc())\
            .offset((page - 1) * page_size)\
            .limit(page_size)\
            .all()
        
        total = query.count()
        
        return jsonify({
            'success': True,
            'data': {
                'samples': [sample.to_dict() for sample in samples],
                'pagination': {
                    'page': page,
                    'page_size': page_size,
                    'total_count': total,
                    'total_pages': (total + page_size - 1) // page_size
                }
            }
        })

@voice_bp.route('/samples/<sample_id>', methods=['GET'])
@jwt_required()
def get_voice_sample(sample_id: str):
    """
    Get details of a specific voice sample.
    
    Args:
        sample_id: The unique identifier of the voice sample
    
    Returns:
        JSON response with sample details
    """
    # Get the current user's ID
    user_id = get_jwt_identity()
    
    db = get_database_manager()
    with db.get_session() as session:
        sample = session.query(VoiceSample).filter_by(id=sample_id, user_id=user_id).first()
        if not sample:
            return jsonify({
                'success': False,
                'error': 'Voice sample not found'
            }), 404
        
        return jsonify({
            'success': True,
            'data': sample.to_dict()
        })

@voice_bp.route('/samples/<sample_id>', methods=['DELETE'])
@jwt_required()
def delete_voice_sample(sample_id: str):
    """
    Delete a voice sample and its associated data.
    
    Args:
        sample_id: The unique identifier of the voice sample
    
    Returns:
        JSON response with deletion status
    """
    # Get the current user's ID
    user_id = get_jwt_identity()
    
    db = get_database_manager()
    with db.get_session() as session:
        sample = session.query(VoiceSample).filter_by(id=sample_id, user_id=user_id).first()
        if not sample:
            return jsonify({
                'success': False,
                'error': 'Voice sample not found'
            }), 404
        
        # Delete the voice embedding from ChromaDB
        if sample.voice_embedding_id:
            delete_voice_embedding(sample.voice_embedding_id)
        
        # Delete from SQLite
        session.delete(sample)
        session.commit()
        
        return jsonify({
            'success': True,
            'data': {
                'message': 'Voice sample deleted successfully'
            }
        }) 