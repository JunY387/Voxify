"""
File Management API Routes
Implements file download and management endpoints for synthesis results
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from flask import request, jsonify, send_file, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from . import file_bp
from database.models import SynthesisJob, SynthesisCache, VoiceModel, VoiceSample, get_database_manager

# Configure file storage paths
SYNTHESIS_OUTPUT_DIR = os.path.join(os.getenv('VOXIFY_SYNTHESIS_STORAGE', 'data/files/synthesis'), 'output')
SYNTHESIS_CACHE_DIR = os.path.join(os.getenv('VOXIFY_SYNTHESIS_STORAGE', 'data/files/synthesis'), 'cache')

# Ensure directories exist
os.makedirs(SYNTHESIS_OUTPUT_DIR, exist_ok=True)
os.makedirs(SYNTHESIS_CACHE_DIR, exist_ok=True)

def error_response(message: str, code: str = None, status_code: int = 400):
    """Standard error response format"""
    return jsonify({
        'success': False,
        'error': {
            'message': message,
            'code': code or f"ERROR_{status_code}",
            'timestamp': datetime.utcnow().isoformat()
        }
    }), status_code

def success_response(data=None, message: str = None, status_code: int = 200):
    """Standard success response format"""
    response = {
        'success': True,
        'timestamp': datetime.utcnow().isoformat()
    }
    if data is not None:
        response['data'] = data
    if message:
        response['message'] = message
    return jsonify(response), status_code

def get_synthesis_file(job_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Get synthesis job file path, filename and cache ID"""
    db = get_database_manager()
    with db.get_session() as session:
        job = session.query(SynthesisJob).filter_by(id=job_id).first()
        if not job:
            return None, None, None
            
        # Check cache first
        if job.cache_hit and job.cached_result_id:
            cache = session.query(SynthesisCache).filter_by(id=job.cached_result_id).first()
            if cache and cache.output_path:
                return cache.output_path, os.path.basename(cache.output_path), cache.id
        
        # Return direct output if no cache or cache miss
        if job.output_path:
            return job.output_path, os.path.basename(job.output_path), None
            
        return None, None, None

@file_bp.route('/synthesis/<job_id>', methods=['GET'])
@jwt_required()
def download_synthesis_file(job_id: str):
    """
    Download synthesized audio file
    
    Args:
        job_id: Synthesis job ID
        
    Returns:
        Audio file or error response
    """
    current_user_id = get_jwt_identity()
    
    # Get file information
    file_path, filename, cache_id = get_synthesis_file(job_id)
    if not file_path:
        return error_response("File not found", "FILE_NOT_FOUND", 404)
    
    # Check if file exists
    if not os.path.exists(file_path):
        return error_response("File not found on server", "FILE_NOT_FOUND", 404)
    
    # Check file ownership
    db = get_database_manager()
    with db.get_session() as session:
        job = session.query(SynthesisJob).filter_by(id=job_id).first()
        if not job or job.user_id != current_user_id:
            return error_response("Access denied", "ACCESS_DENIED", 403)
        
        # Update cache access time if using cached file
        if cache_id:
            cache = session.query(SynthesisCache).filter_by(id=cache_id).first()
            if cache:
                cache.hit_count += 1
                cache.last_accessed = datetime.utcnow()
                session.commit()
    
    try:
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype=f'audio/{os.path.splitext(filename)[1][1:]}'
        )
    except Exception as e:
        return error_response(f"Error downloading file: {str(e)}", "DOWNLOAD_ERROR", 500)

@file_bp.route('/synthesis/<job_id>', methods=['DELETE'])
@jwt_required()
def delete_synthesis_file(job_id: str):
    """
    Delete synthesized audio file
    
    Args:
        job_id: Synthesis job ID
        
    Returns:
        Success or error response
    """
    current_user_id = get_jwt_identity()
    
    # Get file information
    file_path, _, cache_id = get_synthesis_file(job_id)
    if not file_path:
        return error_response("File not found", "FILE_NOT_FOUND", 404)
    
    # Check file ownership
    db = get_database_manager()
    with db.get_session() as session:
        job = session.query(SynthesisJob).filter_by(id=job_id).first()
        if not job or job.user_id != current_user_id:
            return error_response("Access denied", "ACCESS_DENIED", 403)
        
        try:
            # Delete physical file if it exists
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Update database records
            if cache_id:
                # If using cached file, just remove the reference
                job.cache_hit = False
                job.cached_result_id = None
            else:
                # If using direct output, clear the output information
                job.output_path = None
                job.output_size = None
                job.duration = None
            
            session.commit()
            return success_response(message="File deleted successfully")
            
        except Exception as e:
            session.rollback()
            return error_response(f"Error deleting file: {str(e)}", "DELETE_ERROR", 500)

@file_bp.route('/voice-clone/<job_id>', methods=['GET'])
@jwt_required()
def download_voice_clone_synthesis(job_id: str):
    """
    Download voice clone synthesized audio file
    
    Args:
        job_id: Synthesis job ID from voice clone
        
    Returns:
        Audio file or error response
    """
    current_user_id = get_jwt_identity()
    
    try:
        # Get synthesis job information
        db = get_database_manager()
        with db.get_session() as session:
            job = session.query(SynthesisJob).filter_by(id=job_id).first()
            if not job:
                return error_response("Synthesis job not found", "JOB_NOT_FOUND", 404)
            
            # Check ownership
            if job.user_id != current_user_id:
                return error_response("Access denied", "ACCESS_DENIED", 403)
            
            # Get file path and ensure it exists
            if not job.output_path:
                return error_response("Audio file not found", "FILE_NOT_FOUND", 404)
            
            # Handle both relative and absolute paths
            if os.path.isabs(job.output_path):
                file_path = job.output_path
            else:
                # Database stores paths relative to backend directory
                # If running from api/ directory, need to go up one level
                # If running from backend/ directory, use as-is
                
                # Method 1: Try from current directory (if running from backend/)
                candidate1 = job.output_path
                # Method 2: Try from parent directory (if running from api/)
                candidate2 = os.path.join('..', job.output_path)
                
                if os.path.exists(candidate1):
                    file_path = os.path.abspath(candidate1)
                elif os.path.exists(candidate2):
                    file_path = os.path.abspath(candidate2)
                else:
                    return error_response("Audio file not found on server", "FILE_NOT_FOUND", 404)
            
            if not os.path.exists(file_path):
                return error_response("Audio file not found on server", "FILE_NOT_FOUND", 404)
            
            # Get filename for download
            filename = os.path.basename(file_path)
            if not filename:
                filename = f"synthesis_{job_id}.wav"
            
            # Send the file
            return send_file(
                file_path,
                as_attachment=False,  # Stream instead of download for audio playback
                download_name=filename,
                mimetype='audio/wav'
            )
            
    except Exception as e:
        return error_response(f"Error downloading file: {str(e)}", "DOWNLOAD_ERROR", 500)

@file_bp.route('/voice-clone/<job_id>/info', methods=['GET'])
@jwt_required()
def get_voice_clone_synthesis_info(job_id: str):
    """
    Get voice clone synthesis file information
    
    Args:
        job_id: Synthesis job ID
        
    Returns:
        File information or error response
    """
    current_user_id = get_jwt_identity()
    
    try:
        db = get_database_manager()
        with db.get_session() as session:
            job = session.query(SynthesisJob).filter_by(id=job_id).first()
            if not job:
                return error_response("Synthesis job not found", "JOB_NOT_FOUND", 404)
            
            # Check ownership
            if job.user_id != current_user_id:
                return error_response("Access denied", "ACCESS_DENIED", 403)
            
            # Get file info
            file_info = {
                'job_id': job.id,
                'text_content': job.text_content,
                'language': job.text_language,
                'status': job.status,
                'output_path': job.output_path,
                'duration': job.duration,
                'created_at': job.created_at.isoformat() if job.created_at else None,
                'completed_at': job.completed_at.isoformat() if job.completed_at else None
            }
            
            # Check if file exists
            if job.output_path and os.path.exists(job.output_path):
                file_info['file_size'] = os.path.getsize(job.output_path)
                file_info['file_exists'] = True
            else:
                file_info['file_exists'] = False
            
            return success_response(file_info)
            
    except Exception as e:
        return error_response(f"Error getting file info: {str(e)}", "INFO_ERROR", 500)