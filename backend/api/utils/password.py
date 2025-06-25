"""
Password Utilities
Functions for password hashing, validation, and security
"""

import bcrypt
import re

def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt
    
    Parameters
    ----------
    password : str
        Plain text password
        
    Returns
    -------
    str
        Hashed password
    """
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against a hash
    
    Parameters
    ----------
    password : str
        Plain text password
    password_hash : str
        Hashed password
        
    Returns
    -------
    bool
        True if password matches hash, False otherwise
    """
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

def validate_password_strength(password: str) -> tuple:
    """
    Validate password strength
    
    Parameters
    ----------
    password : str
        Password to validate
        
    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    # Check for at least one uppercase letter
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    # Check for at least one lowercase letter
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    # Check for at least one digit
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    
    # Check for at least one special character
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    
    return True, ""

def validate_email(email: str) -> tuple:
    """
    Validate email format
    
    Parameters
    ----------
    email : str
        Email to validate
        
    Returns
    -------
    tuple
        (is_valid, error_message)
    """
    # Email validation regex
    email_regex = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    if not email_regex.match(email):
        return False, "Invalid email format"
    
    return True, ""