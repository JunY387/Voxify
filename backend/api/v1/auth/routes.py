"""
Authentication API Routes
Implements user registration, login, and token management endpoints
"""

from flask import request, jsonify, current_app
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_required, get_jwt_identity
from sqlalchemy.exc import IntegrityError
from . import auth_bp
from datetime import datetime

# Import database models
from database.models import User, get_database_manager

# Import utility functions
from api.utils.password import hash_password, verify_password, validate_password_strength, validate_email

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
def success_response(data=None, message: str = None, status_code: int = 200):
    """Create standardized success response"""
    response = {
        "success": True,
        "timestamp": datetime.utcnow().isoformat()
    }
    if data is not None:
        response["data"] = data
    if message:
        response["message"] = message
    return jsonify(response), status_code

@auth_bp.route('/register', methods=['POST'])
def register():
    """
    Register a new user

    Request Body:
    - email: User's email address
    - password: User's password
    - first_name: User's first name (optional)
    - last_name: User's last name (optional)

    Returns:
    - 201: User created successfully
    - 400: Invalid request data
    - 409: Email already exists
    """
    # Get request data
    data = request.get_json()
    if not data:
        return error_response("Request body is required", "MISSING_BODY")

    # Validate required fields
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return error_response("Email and password are required", "MISSING_FIELDS")

    # Validate email format
    is_valid, error_message = validate_email(email)
    if not is_valid:
        return error_response(error_message, "INVALID_EMAIL")

    # Validate password strength
    is_valid, error_message = validate_password_strength(password)
    if not is_valid:
        return error_response(error_message, "WEAK_PASSWORD")

    # Hash password
    password_hash = hash_password(password)

    # Create user object
    new_user = User(
        email=email,
        password_hash=password_hash,
        first_name=data.get('first_name'),
        last_name=data.get('last_name')
    )

    # Save to database
    db_manager = get_database_manager()
    session = db_manager.get_session()

    try:
        session.add(new_user)
        session.commit()

        # Return success response
        user_data = {
            "id": new_user.id,
            "email": new_user.email,
            "first_name": new_user.first_name,
            "last_name": new_user.last_name,
            "created_at": new_user.created_at.isoformat() if new_user.created_at else None
        }
        
        return success_response(
            data={"user": user_data},
            message="User registered successfully",
            status_code=201
        )

    except IntegrityError:
        session.rollback()
        return error_response("Email already exists", "EMAIL_EXISTS", status_code=409)
    except Exception as e:
        session.rollback()
        return error_response(f"Registration failed: {str(e)}", "REGISTRATION_ERROR", status_code=500)
    finally:
        session.close()

@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Authenticate a user and issue JWT tokens

    Request Body:
    - email: User's email address
    - password: User's password

    Returns:
    - 200: Authentication successful
    - 400: Invalid request data
    - 401: Invalid credentials
    """
    # Get request data
    data = request.get_json()
    if not data:
        return error_response("Request body is required", "MISSING_BODY")

    # Validate required fields
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return error_response("Email and password are required", "MISSING_FIELDS")

    # Get user from database
    db_manager = get_database_manager()
    session = db_manager.get_session()

    try:
        user = session.query(User).filter_by(email=email).first()

        # Check if user exists and password is correct
        if not user or not verify_password(password, user.password_hash):
            return error_response("Invalid email or password", "INVALID_CREDENTIALS", status_code=401)

        # Check if user is active
        if not user.is_active:
            return error_response("Account is disabled", "ACCOUNT_DISABLED", status_code=403)

        # Update last login timestamp
        user.last_login_at = datetime.utcnow()
        session.commit()

        # Create tokens
        access_token = create_access_token(identity=user.id)
        refresh_token = create_refresh_token(identity=user.id)

        # Return tokens and user data
        response_data = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "user": {
                "id": user.id,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "last_login_at": user.last_login_at.isoformat()
            }
        }
        
        return success_response(
            data=response_data,
            message="Login successful"
        )

    except Exception as e:
        session.rollback()
        return error_response(f"Login failed: {str(e)}", "LOGIN_ERROR", status_code=500)
    finally:
        session.close()

@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """
    Refresh the access token using a valid refresh token

    Requires a valid refresh token in the Authorization header.

    Returns:
    - 200: New access token (and optionally a new refresh token)
    - 401: Invalid or expired refresh token
    """
    try:
        # Get the identity of the user from the current refresh token
        current_user = get_jwt_identity()

        # Generate a new access token
        new_access_token = create_access_token(identity=current_user)

        # Optionally, generate a new refresh token (optional logic if needed)
        new_refresh_token = create_refresh_token(identity=current_user)

        # Return the new tokens
        response_data = {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token
        }
        
        return success_response(
            data=response_data,
            message="Token refreshed successfully"
        )

    except Exception as e:
        return error_response(f"Failed to refresh token: {str(e)}", "TOKEN_REFRESH_ERROR", status_code=500)

@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """
    Get current user's profile information

    Requires a valid access token in the Authorization header.

    Returns:
    - 200: User profile information
    - 401: Invalid or expired access token
    - 404: User not found
    """
    try:
        # Get the current user's ID from the JWT token
        current_user_id = get_jwt_identity()

        # Get user from database
        db_manager = get_database_manager()
        session = db_manager.get_session()

        try:
            user = session.query(User).filter_by(id=current_user_id).first()

            if not user:
                return error_response("User not found", "USER_NOT_FOUND", status_code=404)

            # Return user profile data
            user_data = {
                "id": user.id,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "is_active": user.is_active,
                "email_verified": user.email_verified,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "updated_at": user.updated_at.isoformat() if user.updated_at else None,
                "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None
            }
            
            return success_response(
                data={"user": user_data},
                message="Profile retrieved successfully"
            )

        finally:
            session.close()

    except Exception as e:
        return error_response(f"Failed to get profile: {str(e)}", "PROFILE_ERROR", status_code=500)

@auth_bp.route('/profile', methods=['PUT', 'PATCH'])
@jwt_required()
def update_profile():
    """
    Update current user's profile information

    Requires a valid access token in the Authorization header.

    Request Body:
    - first_name: User's first name (optional)
    - last_name: User's last name (optional)
    - email: User's email address (optional)

    Returns:
    - 200: Profile updated successfully
    - 400: Invalid request data
    - 401: Invalid or expired access token
    - 404: User not found
    - 409: Email already exists
    """
    try:
        # Get the current user's ID from the JWT token
        current_user_id = get_jwt_identity()

        # Get request data
        data = request.get_json()
        if not data:
            return error_response("Request body is required", "MISSING_BODY")

        # Get user from database
        db_manager = get_database_manager()
        session = db_manager.get_session()

        try:
            user = session.query(User).filter_by(id=current_user_id).first()

            if not user:
                return error_response("User not found", "USER_NOT_FOUND", status_code=404)

            # Update fields if provided
            updated_fields = []
            
            if 'first_name' in data:
                user.first_name = data['first_name']
                updated_fields.append('first_name')
            
            if 'last_name' in data:
                user.last_name = data['last_name']
                updated_fields.append('last_name')
            
            if 'email' in data:
                new_email = data['email']
                # Validate email format
                is_valid, error_message = validate_email(new_email)
                if not is_valid:
                    return error_response(error_message, "INVALID_EMAIL")
                
                # Check if email is already taken by another user
                existing_user = session.query(User).filter_by(email=new_email).filter(User.id != current_user_id).first()
                if existing_user:
                    return error_response("Email already exists", "EMAIL_EXISTS", status_code=409)
                
                user.email = new_email
                user.email_verified = False  # Reset email verification when email changes
                updated_fields.append('email')

            if not updated_fields:
                return error_response("No valid fields provided for update", "NO_FIELDS_TO_UPDATE")

            # Update timestamp
            user.updated_at = datetime.utcnow()
            session.commit()

            # Return updated user profile data
            user_data = {
                "id": user.id,
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "is_active": user.is_active,
                "email_verified": user.email_verified,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "updated_at": user.updated_at.isoformat() if user.updated_at else None,
                "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None
            }
            
            return success_response(
                data={"user": user_data, "updated_fields": updated_fields},
                message="Profile updated successfully"
            )

        except IntegrityError:
            session.rollback()
            return error_response("Email already exists", "EMAIL_EXISTS", status_code=409)
        finally:
            session.close()

    except Exception as e:
        return error_response(f"Failed to update profile: {str(e)}", "PROFILE_UPDATE_ERROR", status_code=500)
