"""
Job Management API Routes
RESTful endpoints for synthesis job management with complete CRUD operations
"""

import json
import time
import hashlib
from datetime import datetime
from flask import request, jsonify, Response
from flask_jwt_extended import jwt_required, get_jwt_identity
from sqlalchemy.exc import IntegrityError
from sqlalchemy import desc, asc

from . import job_bp
from database.models import SynthesisJob, VoiceModel, get_database_manager

# Standard error response format
def error_response(message: str, code: str = None, details: dict = None, status_code: int = 400):
    """Create standardized error response"""
    response = {
        "success": False,
        "error": {
            "message": message,
            "code": code or f"ERROR_{status_code}",
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    if details:
        response["error"]["details"] = details
    return jsonify(response), status_code

# Standard success response format
def success_response(data=None, message: str = None, status_code: int = 200, meta: dict = None):
    """Create standardized success response"""
    response = {
        "success": True,
        "timestamp": datetime.utcnow().isoformat()
    }
    if data is not None:
        response["data"] = data
    if message:
        response["message"] = message
    if meta:
        response["meta"] = meta
    return jsonify(response), status_code

# Used to validate the data for a synthesis job.
def validate_synthesis_job_data(data: dict, is_update: bool = False) -> tuple:
    """Validate synthesis job data"""
    errors = {}
    
    # Required fields for creation
    if not is_update:
        if not data.get('text_content'):
            errors['text_content'] = 'Text content is required'
        if not data.get('voice_model_id'):
            errors['voice_model_id'] = 'Voice model ID is required'
    
    # Validate text content
    text_content = data.get('text_content', '')
    if text_content and len(text_content) > 1000000000000000000:
        errors['text_content'] = 'Text content cannot exceed 1000000000000000000 characters'
    
    # Validate synthesis parameters
    speed = data.get('speed')
    if speed is not None:
        try:
            speed = float(speed)
            if speed <= 0 or speed > 3.0:
                errors['speed'] = 'Speed must be between 0.1 and 3.0'
        except (ValueError, TypeError):
            errors['speed'] = 'Speed must be a valid number'
    
    pitch = data.get('pitch')
    if pitch is not None:
        try:
            pitch = float(pitch)
            if pitch <= 0 or pitch > 3.0:
                errors['pitch'] = 'Pitch must be between 0.1 and 3.0'
        except (ValueError, TypeError):
            errors['pitch'] = 'Pitch must be a valid number'
    
    volume = data.get('volume')
    if volume is not None:
        try:
            volume = float(volume)
            if volume < 0 or volume > 2.0:
                errors['volume'] = 'Volume must be between 0.0 and 2.0'
        except (ValueError, TypeError):
            errors['volume'] = 'Volume must be a valid number'
    
    # Validate output format
    output_format = data.get('output_format')
    if output_format and output_format not in ['wav', 'mp3', 'flac', 'ogg']:
        errors['output_format'] = 'Output format must be one of: wav, mp3, flac, ogg'
    
    # Validate sample rate
    sample_rate = data.get('sample_rate')
    if sample_rate is not None:
        try:
            sample_rate = int(sample_rate)
            if sample_rate not in [8000, 16000, 22050, 44100, 48000]:
                errors['sample_rate'] = 'Sample rate must be one of: 8000, 16000, 22050, 44100, 48000'
        except (ValueError, TypeError):
            errors['sample_rate'] = 'Sample rate must be a valid integer'
    
    return len(errors) == 0, errors

# Used to generate a hash for the text content and configuration so that we can cache the results.
def generate_text_hash(text: str, config: dict = None) -> str:
    """Generate hash for text content and configuration"""
    content = text + json.dumps(config or {}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()

# this method may need to be changed for different roles, like admin, user, etc. I am envisioning that admin can see
# all jobs, while user can only see their own jobs.
@job_bp.route('', methods=['GET'])
@jwt_required()
def list_jobs():
    """
    List synthesis jobs with filtering, sorting, and pagination
    
    Query Parameters:
    - status: Filter by job status (pending, processing, completed, failed, cancelled)
    - user_id: Filter by user ID (admin only)
    - voice_model_id: Filter by voice model ID
    - text_search: Search in text content
    - limit: Number of items per page (1-100, default: 20)
    - offset: Number of items to skip (default: 0)
    - sort_by: Sort field (created_at, updated_at, status, progress, duration)
    - sort_order: Sort order (asc, desc, default: desc)
    - include_text: Include full text content in response (default: false)
    
    Returns:
    - 200: List of jobs with pagination metadata
    - 400: Invalid query parameters
    - 500: Internal server error
    """
    try:
        # Parse and validate query parameters
        status = request.args.get('status')
        user_id = request.args.get('user_id')
        voice_model_id = request.args.get('voice_model_id')
        text_search = request.args.get('text_search')
        include_text = request.args.get('include_text', 'false').lower() == 'true'
        
        # Pagination parameters
        try:
            limit = min(max(int(request.args.get('limit', 20)), 1), 100)
            offset = max(int(request.args.get('offset', 0)), 0)
        except ValueError:
            return error_response("Invalid pagination parameters", "INVALID_PAGINATION")
        
        # Sorting parameters
        sort_by = request.args.get('sort_by', 'created_at')
        sort_order = request.args.get('sort_order', 'desc').lower()
        
        # Validate parameters
        valid_statuses = ['pending', 'processing', 'completed', 'failed', 'cancelled']
        if status and status not in valid_statuses:
            return error_response(f"Invalid status. Must be one of: {', '.join(valid_statuses)}", "INVALID_STATUS")
        
        valid_sort_fields = ['created_at', 'updated_at', 'status', 'progress', 'duration']
        if sort_by not in valid_sort_fields:
            return error_response(f"Invalid sort_by field. Must be one of: {', '.join(valid_sort_fields)}", "INVALID_SORT_FIELD")
        
        if sort_order not in ['asc', 'desc']:
            return error_response("Sort order must be 'asc' or 'desc'", "INVALID_SORT_ORDER")
        
        # Get database session
        db_manager = get_database_manager()
        session = db_manager.get_session()
        
        try:
            # Build query with filters
            query = session.query(SynthesisJob)
            
            # Apply filters
            current_user_id = get_jwt_identity()
            if not user_id:  # Regular users can only see their own jobs
                query = query.filter(SynthesisJob.user_id == current_user_id)
            elif user_id != current_user_id:
                # TODO: Add admin role check here
                query = query.filter(SynthesisJob.user_id == user_id)
            
            if status:
                query = query.filter(SynthesisJob.status == status)
            if voice_model_id:
                query = query.filter(SynthesisJob.voice_model_id == voice_model_id)
            if text_search:
                query = query.filter(SynthesisJob.text_content.contains(text_search))
            
            # Get total count for pagination
            total_count = query.count()
            
            # Apply sorting
            sort_column = getattr(SynthesisJob, sort_by)
            if sort_order == 'desc':
                query = query.order_by(desc(sort_column))
            else:
                query = query.order_by(asc(sort_column))
            
            # Apply pagination
            jobs = query.offset(offset).limit(limit).all()
            
            # Convert to dictionaries
            job_dicts = []
            for job in jobs:
                job_dict = job.to_dict()
                if not include_text:
                    job_dict['text_content'] = job_dict['text_content'][:100] + '...' if len(job_dict['text_content']) > 100 else job_dict['text_content']
                job_dicts.append(job_dict)
            
            # Pagination metadata
            meta = {
                "pagination": {
                    "total": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_next": offset + limit < total_count,
                    "has_prev": offset > 0,
                    "page": (offset // limit) + 1,
                    "total_pages": (total_count + limit - 1) // limit
                },
                "filters": {
                    "status": status,
                    "voice_model_id": voice_model_id,
                    "text_search": text_search
                },
                "sorting": {
                    "sort_by": sort_by,
                    "sort_order": sort_order
                }
            }
            
            return success_response(job_dicts, meta=meta)
            
        finally:
            session.close()
            
    except Exception as e:
        return error_response(f"Failed to retrieve jobs: {str(e)}", "RETRIEVAL_ERROR", status_code=500)

# Used to create a new synthesis job. Text content is required, voice model id is required, and other parameters are optional.
@job_bp.route('', methods=['POST'])
@jwt_required()
def create_job():
    """
    Create a new synthesis job
    
    Request Body:
    - text_content: Text to synthesize (required)
    - voice_model_id: ID of the voice model to use (required)
    - text_language: Language code (default: en-US)
    - output_format: Output audio format (default: wav)
    - sample_rate: Audio sample rate (default: 22050)
    - speed: Speech speed multiplier (default: 1.0)
    - pitch: Pitch multiplier (default: 1.0)
    - volume: Volume multiplier (default: 1.0)
    - config: Additional synthesis configuration (optional)
    
    Returns:
    - 201: Job created successfully
    - 400: Invalid request data
    - 404: Voice model not found
    - 500: Internal server error
    """
    try:
        # Get and validate request data
        data = request.get_json()
        if not data:
            return error_response("Request body is required", "MISSING_BODY")
        
        is_valid, validation_errors = validate_synthesis_job_data(data)
        if not is_valid:
            return error_response("Validation failed", "VALIDATION_ERROR", validation_errors)
        
        # Get database session
        db_manager = get_database_manager()
        session = db_manager.get_session()
        
        try:
            # Verify voice model exists
            voice_model = session.query(VoiceModel).filter(
                VoiceModel.id == data['voice_model_id'],
                VoiceModel.is_active == True
            ).first()
            
            if not voice_model:
                return error_response("Voice model not found or inactive", "VOICE_MODEL_NOT_FOUND", status_code=404)
            
            # Generate text hash for caching
            config = data.get('config', {})
            text_hash = generate_text_hash(data['text_content'], config)
            existing_job = session.query(SynthesisJob).filter_by(
            user_id=get_jwt_identity(),
            text_hash=text_hash,
            voice_model_id=data['voice_model_id'],
            output_format=data.get('output_format', 'wav'),
            sample_rate=data.get('sample_rate', 22050)
        ).first()

            if existing_job:
                return success_response(
                    existing_job.to_dict(),
                    message="Duplicate synthesis job already exists",
                    status_code=200 
                )

            
            # Create synthesis job
            job = SynthesisJob(
                user_id=get_jwt_identity(),
                voice_model_id=data['voice_model_id'],
                text_content=data['text_content'],
                text_hash=text_hash,
                text_language=data.get('text_language', 'en-US'),
                text_length=len(data['text_content']),
                word_count=len(data['text_content'].split()),
                config=json.dumps(config) if config else None,
                output_format=data.get('output_format', 'wav'),
                sample_rate=data.get('sample_rate', 22050),
                speed=data.get('speed', 1.0),
                pitch=data.get('pitch', 1.0),
                volume=data.get('volume', 1.0),
                status='pending',
                progress=0.0,  # Initialize progress to 0.0
                # Initialize timestamp fields
                word_timestamps=None,
                syllable_timestamps=None,
                phoneme_timestamps=None
            )
            
            session.add(job)
            session.commit()
            
            return success_response(
                job.to_dict(),
                message="Synthesis job created successfully",
                status_code=201
            )
            
        finally:
            session.close()
            
    except Exception as e:
        return error_response(f"Failed to create job: {str(e)}", "CREATION_ERROR", status_code=500)

@job_bp.route('/<job_id>', methods=['GET'])
@jwt_required()
def get_job(job_id):
    """
    Get detailed information about a specific synthesis job
    
    Path Parameters:
    - job_id: Unique identifier of the job
    
    Returns:
    - 200: Job details
    - 404: Job not found
    - 403: Access denied
    - 500: Internal server error
    """
    try:
        db_manager = get_database_manager()
        session = db_manager.get_session()
        
        try:
            job = session.query(SynthesisJob).filter(SynthesisJob.id == job_id).first()
            
            if not job:
                return error_response("Job not found", "JOB_NOT_FOUND", status_code=404)
            
            # Check access permissions
            current_user_id = get_jwt_identity()
            if job.user_id != current_user_id:
                # TODO: Add admin role check here
                return error_response("Access denied", "ACCESS_DENIED", status_code=403)
            
            return success_response(job.to_dict())
            
        finally:
            session.close()
            
    except Exception as e:
        return error_response(f"Failed to retrieve job: {str(e)}", "RETRIEVAL_ERROR", status_code=500)

@job_bp.route('/<job_id>', methods=['PUT'])
@jwt_required()
def update_job(job_id):
    """
    Update a synthesis job (only allowed for pending jobs)
    
    Path Parameters:
    - job_id: Unique identifier of the job
    
    Request Body:
    - text_content: Updated text content (optional)
    - speed: Updated speech speed (optional)
    - pitch: Updated pitch (optional)
    - volume: Updated volume (optional)
    - output_format: Updated output format (optional)
    - sample_rate: Updated sample rate (optional)
    - config: Updated synthesis configuration (optional)
    
    Returns:
    - 200: Job updated successfully
    - 400: Invalid request data or job cannot be updated
    - 404: Job not found
    - 403: Access denied
    - 500: Internal server error
    """
    try:
        # Get and validate request data
        data = request.get_json()
        if not data:
            return error_response("Request body is required", "MISSING_BODY")
        
        is_valid, validation_errors = validate_synthesis_job_data(data, is_update=True)
        if not is_valid:
            return error_response("Validation failed", "VALIDATION_ERROR", validation_errors)
        
        db_manager = get_database_manager()
        session = db_manager.get_session()
        
        try:
            job = session.query(SynthesisJob).filter(SynthesisJob.id == job_id).first()
            
            if not job:
                return error_response("Job not found", "JOB_NOT_FOUND", status_code=404)
            
            # Check access permissions
            current_user_id = get_jwt_identity()
            if job.user_id != current_user_id:
                return error_response("Access denied", "ACCESS_DENIED", status_code=403)
            
            # Check if job can be updated
            if job.status not in ['pending']:
                return error_response(
                    f"Cannot update job with status '{job.status}'. Only pending jobs can be updated.",
                    "JOB_NOT_UPDATABLE"
                )
            
            # Update job fields
            updated_fields = []
            
            if 'text_content' in data:
                job.text_content = data['text_content']
                job.text_length = len(data['text_content'])
                job.word_count = len(data['text_content'].split())
                job.text_hash = generate_text_hash(data['text_content'], job.config_dict)
                updated_fields.append('text_content')
            
            if 'speed' in data:
                job.speed = float(data['speed'])
                updated_fields.append('speed')
            
            if 'pitch' in data:
                job.pitch = float(data['pitch'])
                updated_fields.append('pitch')
            
            if 'volume' in data:
                job.volume = float(data['volume'])
                updated_fields.append('volume')
            
            if 'output_format' in data:
                job.output_format = data['output_format']
                updated_fields.append('output_format')
            
            if 'sample_rate' in data:
                job.sample_rate = int(data['sample_rate'])
                updated_fields.append('sample_rate')
            
            if 'config' in data:
                job.config = json.dumps(data['config']) if data['config'] else None
                updated_fields.append('config')
            
            job.updated_at = datetime.utcnow()
            session.commit()
            
            return success_response(
                job.to_dict(),
                message=f"Job updated successfully. Updated fields: {', '.join(updated_fields)}"
            )
            
        finally:
            session.close()
            
    except Exception as e:
        return error_response(f"Failed to update job: {str(e)}", "UPDATE_ERROR", status_code=500)

@job_bp.route('/<job_id>', methods=['PATCH'])
@jwt_required()
def patch_job(job_id):
    """
    Partially update a synthesis job or change its status
    
    Path Parameters:
    - job_id: Unique identifier of the job
    
    Request Body:
    - status: New job status (pending, processing, completed, failed, cancelled)
    - error_message: Error message (when status is 'failed')
    - progress: Job progress (0.0 to 1.0)
    - output_path: Path to generated audio file
    - duration: Duration of generated audio
    - processing_time_ms: Processing time in milliseconds
    
    Returns:
    - 200: Job updated successfully
    - 400: Invalid request data
    - 404: Job not found
    - 403: Access denied
    - 500: Internal server error
    """
    try:
        data = request.get_json()
        if not data:
            return error_response("Request body is required", "MISSING_BODY")
        
        db_manager = get_database_manager()
        session = db_manager.get_session()
        
        try:
            job = session.query(SynthesisJob).filter(SynthesisJob.id == job_id).first()
            
            if not job:
                return error_response("Job not found", "JOB_NOT_FOUND", status_code=404)
            
            # Check access permissions
            current_user_id = get_jwt_identity()
            if job.user_id != current_user_id:
                return error_response("Access denied", "ACCESS_DENIED", status_code=403)
            
            updated_fields = []
            
            # Update status
            if 'status' in data:
                new_status = data['status']
                valid_statuses = ['pending', 'processing', 'completed', 'failed', 'cancelled']
                
                if new_status not in valid_statuses:
                    return error_response(f"Invalid status. Must be one of: {', '.join(valid_statuses)}", "INVALID_STATUS")
                
                # Validate status transitions
                current_status = job.status
                valid_transitions = {
                    'pending': ['processing', 'cancelled', 'failed'],
                    'processing': ['completed', 'failed', 'cancelled'],
                    'completed': [],  # Terminal state
                    'failed': ['pending'],  # Allow retry
                    'cancelled': ['pending']  # Allow restart
                }
                
                if new_status != current_status and new_status not in valid_transitions.get(current_status, []):
                    return error_response(
                        f"Invalid status transition from '{current_status}' to '{new_status}'",
                        "INVALID_STATUS_TRANSITION"
                    )
                
                job.status = new_status
                updated_fields.append('status')
                
                # Set timestamps based on status
                if new_status == 'processing' and not job.started_at:
                    job.started_at = datetime.utcnow()
                elif new_status in ['completed', 'failed', 'cancelled'] and not job.completed_at:
                    job.completed_at = datetime.utcnow()
            
            # Update other fields
            if 'error_message' in data:
                job.error_message = data['error_message']
                updated_fields.append('error_message')
            
            if 'progress' in data:
                progress = float(data['progress'])
                if 0.0 <= progress <= 1.0:
                    job.progress = progress
                    updated_fields.append('progress')
                else:
                    return error_response("Progress must be between 0.0 and 1.0", "INVALID_PROGRESS")
            
            if 'output_path' in data:
                job.output_path = data['output_path']
                updated_fields.append('output_path')
            
            if 'duration' in data:
                job.duration = float(data['duration'])
                updated_fields.append('duration')
            
            if 'processing_time_ms' in data:
                job.processing_time_ms = int(data['processing_time_ms'])
                updated_fields.append('processing_time_ms')
            
            job.updated_at = datetime.utcnow()
            session.commit()
            
            return success_response(
                job.to_dict(),
                message=f"Job updated successfully. Updated fields: {', '.join(updated_fields)}"
            )
            
        finally:
            session.close()
            
    except Exception as e:
        return error_response(f"Failed to update job: {str(e)}", "UPDATE_ERROR", status_code=500)

@job_bp.route('/<job_id>', methods=['DELETE'])
@jwt_required()
def delete_job(job_id):
    """
    Delete a synthesis job (only allowed for completed, failed, or cancelled jobs)
    
    Path Parameters:
    - job_id: Unique identifier of the job
    
    Returns:
    - 204: Job deleted successfully
    - 400: Job cannot be deleted
    - 404: Job not found
    - 403: Access denied
    - 500: Internal server error
    """
    try:
        db_manager = get_database_manager()
        session = db_manager.get_session()
        
        try:
            job = session.query(SynthesisJob).filter(SynthesisJob.id == job_id).first()
            
            if not job:
                return error_response("Job not found", "JOB_NOT_FOUND", status_code=404)
            
            # Check access permissions
            current_user_id = get_jwt_identity()
            if job.user_id != current_user_id:
                return error_response("Access denied", "ACCESS_DENIED", status_code=403)
            
            # Check if job can be deleted
            if job.status in ['pending', 'processing']:
                return error_response(
                    f"Cannot delete job with status '{job.status}'. Only completed, failed, or cancelled jobs can be deleted.",
                    "JOB_NOT_DELETABLE"
                )
            
            session.delete(job)
            session.commit()
            
            return '', 204  # No content response
            
        finally:
            session.close()
            
    except Exception as e:
        return error_response(f"Failed to delete job: {str(e)}", "DELETION_ERROR", status_code=500)

# Legacy endpoint for backwards compatibility
@job_bp.route('/<job_id>/cancel', methods=['POST'])
@jwt_required()
def cancel_job_legacy(job_id):
    """
    Cancel a synthesis job (legacy endpoint - use PATCH instead)
    
    This endpoint is deprecated. Use PATCH /<job_id> with status: "cancelled" instead.
    """
    try:
        db_manager = get_database_manager()
        session = db_manager.get_session()
        
        try:
            job = session.query(SynthesisJob).filter(SynthesisJob.id == job_id).first()
            
            if not job:
                return error_response("Job not found", "JOB_NOT_FOUND", status_code=404)
            
            # Check access permissions
            current_user_id = get_jwt_identity()
            if job.user_id != current_user_id:
                return error_response("Access denied", "ACCESS_DENIED", status_code=403)
            
            # Check if job can be cancelled
            if job.status not in ['pending', 'processing']:
                return error_response(
                    f"Cannot cancel job with status '{job.status}'. Only pending or processing jobs can be cancelled.",
                    "JOB_NOT_CANCELLABLE"
                )
            
            # Update job status
            job.status = 'cancelled'
            job.error_message = 'Job cancelled by user request'
            job.updated_at = datetime.utcnow()
            
            if not job.completed_at:
                job.completed_at = datetime.utcnow()
            
            session.commit()
            
            return success_response(
                job.to_dict(),
                message="Job cancelled successfully. Note: This endpoint is deprecated, use PATCH instead."
            )
            
        finally:
            session.close()
            
    except Exception as e:
        return error_response(f"Failed to cancel job: {str(e)}", "CANCELLATION_ERROR", status_code=500)

def generate_job_progress_events(job_id):
    """
    Generator function for Server-Sent Events job progress updates
    """
    try:
        db_manager = get_database_manager()
        
        while True:
            session = db_manager.get_session()
            
            try:
                job = session.query(SynthesisJob).filter(SynthesisJob.id == job_id).first()
                
                if not job:
                    yield f"event: error\ndata: {json.dumps({'error': 'Job not found', 'code': 'JOB_NOT_FOUND'})}\n\n"
                    break
                
                # Prepare progress data
                progress_data = {
                    "job_id": job.id,
                    "status": job.status,
                    "progress": job.progress or 0.0,
                    "progress_percentage": round((job.progress or 0.0) * 100, 2),
                    "message": job.error_message if job.status == "failed" else "Processing...",
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "updated_at": job.updated_at.isoformat() if job.updated_at else None,
                    "estimated_completion": None
                }
                
                # Add completion data for finished jobs
                if job.status in ["completed", "failed", "cancelled"]:
                    progress_data.update({
                        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                        "processing_time_ms": job.processing_time_ms,
                        "output_path": job.output_path if job.status == "completed" else None,
                        "duration": job.duration if job.status == "completed" else None
                    })
                
                # Send progress update
                yield f"event: progress\ndata: {json.dumps(progress_data)}\n\n"
                
                # Check if job is in terminal state
                if job.status in ["completed", "failed", "cancelled"]:
                    yield f"event: complete\ndata: {json.dumps({'status': job.status, 'job_id': job.id})}\n\n"
                    break
                
                # Wait before next update (1 second interval)
                time.sleep(1.0)
                
            finally:
                session.close()
                
    except Exception as e:
        error_data = {
            "error": "Stream error occurred",
            "code": "STREAM_ERROR",
            "details": str(e),
            "job_id": job_id
        }
        yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

@job_bp.route('/<job_id>/progress', methods=['GET'])
@jwt_required()
def stream_job_progress(job_id):
    """
    Stream real-time job progress updates using Server-Sent Events (SSE)
    
    Path Parameters:
    - job_id: Unique identifier of the job to monitor
    
    Returns:
    - 200: SSE stream with progress updates
    - 404: Job not found
    - 403: Access denied
    - 500: Internal server error
    """
    try:
        # Verify job exists and user has access before starting stream
        db_manager = get_database_manager()
        session = db_manager.get_session()
        
        try:
            job = session.query(SynthesisJob).filter(SynthesisJob.id == job_id).first()
            
            if not job:
                return error_response("Job not found", "JOB_NOT_FOUND", status_code=404)
            
            # Check access permissions
            current_user_id = get_jwt_identity()
            if job.user_id != current_user_id:
                return error_response("Access denied", "ACCESS_DENIED", status_code=403)
            
            return Response(
                generate_job_progress_events(job_id),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Cache-Control'
                }
            )
            
        finally:
            session.close()
            
    except Exception as e:
        return error_response(f"Failed to start progress stream: {str(e)}", "STREAM_ERROR", status_code=500) 