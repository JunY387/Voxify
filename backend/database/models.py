"""
Voxify Database ORM Models
SQLAlchemy models for the Voxify platform hybrid storage system
"""

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean, Text, 
    DateTime, ForeignKey, CheckConstraint, Index, func, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.sqlite import TEXT
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import json

Base = declarative_base()

def generate_uuid():
    """Generate UUID for primary keys"""
    return str(uuid.uuid4())

def utc_now():
    """Get current UTC time with timezone info"""
    return datetime.now(timezone.utc)

class TimestampMixin:
    """Mixin for created_at and updated_at timestamps"""
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

class User(Base, TimestampMixin):
    """User account model"""
    __tablename__ = 'users'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    first_name = Column(String)
    last_name = Column(String)
    storage_used_bytes = Column(Integer, default=0, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    last_login_at = Column(DateTime)
    
    # Relationships
    voice_samples = relationship("VoiceSample", back_populates="user", cascade="all, delete-orphan")
    synthesis_jobs = relationship("SynthesisJob", back_populates="user", cascade="all, delete-orphan")
    usage_stats = relationship("UsageStat", back_populates="user", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('storage_used_bytes >= 0', name='check_storage_used_bytes'),
        Index('idx_users_email', 'email'),
        Index('idx_users_active', 'is_active'),
    )
    
    @property
    def full_name(self) -> str:
        """Get user's full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.first_name or self.last_name or self.email
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'full_name': self.full_name,
            'storage_used_bytes': self.storage_used_bytes,
            'is_active': self.is_active,
            'email_verified': self.email_verified,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login_at': self.last_login_at.isoformat() if self.last_login_at else None
        }

class VoiceSample(Base, TimestampMixin):
    """Voice sample model for uploaded audio files"""
    __tablename__ = 'voice_samples'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    
    # File information
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    original_filename = Column(String)
    file_hash = Column(String)  # SHA256 hash for deduplication
    
    # Audio properties
    format = Column(String, nullable=False)
    duration = Column(Float, nullable=False)
    sample_rate = Column(Integer, nullable=False)
    channels = Column(Integer, default=1)
    bit_depth = Column(Integer)
    
    # Language and quality metrics
    language = Column(String, default='en-US')
    quality_score = Column(Float)  # 0-10 scale
    noise_level = Column(Float)
    clarity_score = Column(Float)
    signal_to_noise_ratio = Column(Float)
    
    # Processing status
    status = Column(String, default='uploaded', nullable=False)
    processing_error = Column(Text)
    processing_start_time = Column(DateTime)
    processing_end_time = Column(DateTime)
    
    # Vector database associations
    voice_embedding_id = Column(String)  # Reference to Chroma
    speaker_embedding_id = Column(String)  # Reference to speaker identity
    
    # Metadata and categorization
    tags = Column(TEXT)  # JSON array of user tags
    is_public = Column(Boolean, default=False)
    gender = Column(String)  # Detected gender
    age_group = Column(String)  # Detected age group
    accent = Column(String)  # Detected accent
    
    # Relationships
    user = relationship("User", back_populates="voice_samples")
    voice_models = relationship("VoiceModel", back_populates="voice_sample", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('status IN ("uploaded", "processing", "ready", "failed")', name='check_status'),
        CheckConstraint('duration > 0', name='check_duration_positive'),
        CheckConstraint('sample_rate > 0', name='check_sample_rate_positive'),
        CheckConstraint('channels > 0', name='check_channels_positive'),
        CheckConstraint('quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 10)', name='check_quality_score_range'),
        Index('idx_voice_samples_user_id', 'user_id'),
        Index('idx_voice_samples_status', 'status'),
        Index('idx_voice_samples_language', 'language'),
        Index('idx_voice_samples_quality', 'quality_score'),
        Index('idx_voice_samples_created', 'created_at'),
        Index('idx_voice_samples_file_hash', 'file_hash'),
    )
    
    @property
    def tags_list(self) -> List[str]:
        """Get tags as a list"""
        if self.tags:
            try:
                return json.loads(self.tags)
            except (json.JSONDecodeError, TypeError):
                return []
        return []
    
    @tags_list.setter
    def tags_list(self, value: List[str]):
        """Set tags from a list"""
        self.tags = json.dumps(value) if value else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description,
            'file_size': self.file_size,
            'original_filename': self.original_filename,
            'format': self.format,
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'language': self.language,
            'quality_score': self.quality_score,
            'noise_level': self.noise_level,
            'clarity_score': self.clarity_score,
            'status': self.status,
            'tags': self.tags_list,
            'is_public': self.is_public,
            'gender': self.gender,
            'age_group': self.age_group,
            'accent': self.accent,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class VoiceModel(Base, TimestampMixin):
    """Voice model for trained AI models"""
    __tablename__ = 'voice_models'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    voice_sample_id = Column(String, ForeignKey('voice_samples.id', ondelete='CASCADE'), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    
    # Model file information
    model_path = Column(String, nullable=False)
    model_type = Column(String, default='tacotron2')
    model_size = Column(Integer)
    model_version = Column(String, default='1.0')
    model_hash = Column(String)
    
    # Training configuration
    training_config = Column(TEXT)  # JSON training parameters
    training_epochs = Column(Integer)
    learning_rate = Column(Float)
    batch_size = Column(Integer)
    dataset_size = Column(Integer)
    
    # Training status and progress
    training_status = Column(String, default='pending', nullable=False)
    training_progress = Column(Float, default=0.0)
    training_start_time = Column(DateTime)
    training_end_time = Column(DateTime)
    training_error = Column(Text)
    training_logs_path = Column(String)
    
    # Quality metrics and evaluation
    quality_metrics = Column(TEXT)  # JSON quality metrics
    mel_loss = Column(Float)
    validation_score = Column(Float)
    mos_score = Column(Float)  # Mean Opinion Score
    similarity_score = Column(Float)
    
    # Model status and management
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    deployment_status = Column(String, default='offline')
    
    # Relationships
    voice_sample = relationship("VoiceSample", back_populates="voice_models")
    synthesis_jobs = relationship("SynthesisJob", back_populates="voice_model")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('training_status IN ("pending", "training", "completed", "failed")', name='check_training_status'),
        CheckConstraint('training_progress >= 0.0 AND training_progress <= 1.0', name='check_training_progress'),
        CheckConstraint('deployment_status IN ("offline", "online", "deploying")', name='check_deployment_status'),
        Index('idx_voice_models_sample_id', 'voice_sample_id'),
        Index('idx_voice_models_status', 'training_status'),
        Index('idx_voice_models_active', 'is_active'),
        Index('idx_voice_models_deployment', 'deployment_status'),
    )
    
    @property
    def training_config_dict(self) -> Dict[str, Any]:
        """Get training config as dictionary"""
        if self.training_config:
            try:
                return json.loads(self.training_config)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}
    
    @training_config_dict.setter
    def training_config_dict(self, value: Dict[str, Any]):
        """Set training config from dictionary"""
        self.training_config = json.dumps(value) if value else None
    
    @property
    def quality_metrics_dict(self) -> Dict[str, Any]:
        """Get quality metrics as dictionary"""
        if self.quality_metrics:
            try:
                return json.loads(self.quality_metrics)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}
    
    @quality_metrics_dict.setter
    def quality_metrics_dict(self, value: Dict[str, Any]):
        """Set quality metrics from dictionary"""
        self.quality_metrics = json.dumps(value) if value else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'voice_sample_id': self.voice_sample_id,
            'name': self.name,
            'description': self.description,
            'model_type': self.model_type,
            'model_size': self.model_size,
            'model_version': self.model_version,
            'training_config': self.training_config_dict,
            'training_status': self.training_status,
            'training_progress': self.training_progress,
            'quality_metrics': self.quality_metrics_dict,
            'mel_loss': self.mel_loss,
            'validation_score': self.validation_score,
            'mos_score': self.mos_score,
            'similarity_score': self.similarity_score,
            'is_active': self.is_active,
            'is_default': self.is_default,
            'deployment_status': self.deployment_status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class SynthesisJob(Base, TimestampMixin):
    """TTS synthesis job model"""
    __tablename__ = 'synthesis_jobs'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    voice_model_id = Column(String, ForeignKey('voice_models.id', ondelete='CASCADE'), nullable=False)
    
    # Input text information
    text_content = Column(Text, nullable=False)
    text_hash = Column(String, nullable=False)
    text_language = Column(String, default='en-US')
    text_length = Column(Integer)
    word_count = Column(Integer)
    
    # Synthesis configuration
    config = Column(TEXT)  # JSON synthesis parameters
    output_format = Column(String, default='wav')
    sample_rate = Column(Integer, default=22050)
    speed = Column(Float, default=1.0)
    pitch = Column(Float, default=1.0)
    volume = Column(Float, default=1.0)
    
    # Output information
    output_path = Column(String)
    output_size = Column(Integer)
    duration = Column(Float)
    
    # Timing alignment data (project requirement)
    word_timestamps = Column(TEXT)  # JSON array
    syllable_timestamps = Column(TEXT)  # JSON array
    phoneme_timestamps = Column(TEXT)  # JSON array
    
    # Job status and progress
    status = Column(String, default='pending', nullable=False)
    progress = Column(Float, default=0.0)
    error_message = Column(Text)
    processing_node = Column(String)
    
    # Caching information
    cache_hit = Column(Boolean, default=False)
    cached_result_id = Column(String)
    
    # Performance metrics
    processing_time_ms = Column(Integer)
    queue_time_ms = Column(Integer)
    
    # Timestamps
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="synthesis_jobs")
    voice_model = relationship("VoiceModel", back_populates="synthesis_jobs")
    phoneme_alignments = relationship("PhonemeAlignment", back_populates="synthesis_job", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('status IN ("pending", "processing", "completed", "failed", "cancelled")', name='check_synthesis_status'),
        CheckConstraint('progress >= 0.0 AND progress <= 1.0', name='check_synthesis_progress'),
        CheckConstraint('speed >= 0.1 AND speed <= 3.0', name='check_speed_range'),
        CheckConstraint('pitch >= 0.1 AND pitch <= 3.0', name='check_pitch_range'),
        CheckConstraint('volume >= 0.0 AND volume <= 3.0', name='check_volume_range'),
        Index('idx_synthesis_jobs_user_id', 'user_id'),
        Index('idx_synthesis_jobs_status', 'status'),
        Index('idx_synthesis_jobs_text_hash', 'text_hash'),
        Index('idx_synthesis_jobs_model_id', 'voice_model_id'),
        Index('idx_synthesis_jobs_created', 'created_at'),
    )
    
    @property
    def config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        if self.config:
            try:
                return json.loads(self.config)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}
    
    @config_dict.setter
    def config_dict(self, value: Dict[str, Any]):
        """Set configuration from dictionary"""
        self.config = json.dumps(value) if value else None
    
    @property
    def word_timestamps_list(self) -> List[Dict[str, Any]]:
        """Get word timestamps as list"""
        if self.word_timestamps:
            try:
                return json.loads(self.word_timestamps)
            except (json.JSONDecodeError, TypeError):
                return []
        return []
    
    @word_timestamps_list.setter
    def word_timestamps_list(self, value: List[Dict[str, Any]]):
        """Set word timestamps from list"""
        self.word_timestamps = json.dumps(value) if value else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'voice_model_id': self.voice_model_id,
            'text_content': self.text_content,
            'text_hash': self.text_hash,
            'text_language': self.text_language,
            'text_length': self.text_length,
            'word_count': self.word_count,
            'config': self.config_dict,
            'output_format': self.output_format,
            'sample_rate': self.sample_rate,
            'speed': self.speed,
            'pitch': self.pitch,
            'volume': self.volume,
            'output_path': self.output_path,
            'output_size': self.output_size,
            'duration': self.duration,
            'word_timestamps': self.word_timestamps_list,
            'status': self.status,
            'progress': self.progress,
            'error_message': self.error_message,
            'cache_hit': self.cache_hit,
            'processing_time_ms': self.processing_time_ms,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

class SynthesisCache(Base, TimestampMixin):
    """Synthesis cache for performance optimization"""
    __tablename__ = 'synthesis_cache'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    text_hash = Column(String, nullable=False)
    voice_model_id = Column(String, ForeignKey('voice_models.id', ondelete='CASCADE'), nullable=False)
    config_hash = Column(String, nullable=False)
    
    # Cached content
    output_path = Column(String, nullable=False)
    duration = Column(Float, nullable=False)
    word_timestamps = Column(TEXT)
    syllable_timestamps = Column(TEXT)
    phoneme_timestamps = Column(TEXT)
    
    # Cache metadata
    hit_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=utc_now)
    file_size = Column(Integer)
    
    # Cache management
    expires_at = Column(DateTime)
    is_permanent = Column(Boolean, default=False)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('duration > 0', name='check_cache_duration_positive'),
        CheckConstraint('hit_count >= 0', name='check_hit_count_positive'),
        UniqueConstraint('text_hash', 'voice_model_id', 'config_hash', name='unique_cache_key'),
        Index('idx_synthesis_cache_hash', 'text_hash'),
        Index('idx_synthesis_cache_model', 'voice_model_id'),
        Index('idx_synthesis_cache_expires', 'expires_at'),
    )

class PhonemeAlignment(Base):
    """Detailed phoneme alignments for synthesis jobs"""
    __tablename__ = 'phoneme_alignments'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    synthesis_job_id = Column(String, ForeignKey('synthesis_jobs.id', ondelete='CASCADE'), nullable=False)
    sequence_number = Column(Integer, nullable=False)
    
    # Text unit information
    text_unit = Column(String, nullable=False)
    unit_type = Column(String, nullable=False)  # 'word', 'syllable', 'phoneme'
    parent_unit_id = Column(String)
    
    # Timing information
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)
    
    # Audio features
    confidence = Column(Float)  # 0-1 alignment confidence
    pitch_mean = Column(Float)
    pitch_std = Column(Float)
    energy_mean = Column(Float)
    energy_std = Column(Float)
    
    created_at = Column(DateTime, default=utc_now)
    
    # Relationships
    synthesis_job = relationship("SynthesisJob", back_populates="phoneme_alignments")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('unit_type IN ("word", "syllable", "phoneme")', name='check_unit_type'),
        CheckConstraint('start_time >= 0', name='check_start_time_positive'),
        CheckConstraint('end_time > start_time', name='check_end_time_after_start'),
        CheckConstraint('duration > 0', name='check_alignment_duration_positive'),
        CheckConstraint('confidence IS NULL OR (confidence >= 0 AND confidence <= 1)', name='check_confidence_range'),
        Index('idx_phoneme_alignments_job', 'synthesis_job_id'),
        Index('idx_phoneme_alignments_time', 'start_time', 'end_time'),
        Index('idx_phoneme_alignments_type', 'unit_type'),
    )

class UsageStat(Base):
    """Daily usage statistics per user"""
    __tablename__ = 'usage_stats'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    date = Column(String, nullable=False)  # DATE in YYYY-MM-DD format
    
    # Usage metrics
    voice_samples_uploaded = Column(Integer, default=0)
    models_trained = Column(Integer, default=0)
    synthesis_requests = Column(Integer, default=0)
    synthesis_duration = Column(Float, default=0.0)
    storage_used = Column(Integer, default=0)
    
    # API call statistics
    api_calls_auth = Column(Integer, default=0)
    api_calls_voice = Column(Integer, default=0)
    api_calls_tts = Column(Integer, default=0)
    api_calls_admin = Column(Integer, default=0)
    
    # Performance metrics
    avg_synthesis_time = Column(Float, default=0.0)
    cache_hit_rate = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=utc_now)
    
    # Relationships
    user = relationship("User", back_populates="usage_stats")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('synthesis_duration >= 0', name='check_synthesis_duration_positive'),
        CheckConstraint('storage_used >= 0', name='check_storage_used_positive'),
        CheckConstraint('cache_hit_rate >= 0 AND cache_hit_rate <= 1', name='check_cache_hit_rate_range'),
        Index('idx_usage_stats_user_date', 'user_id', 'date', unique=True),
        Index('idx_usage_stats_date', 'date'),
    )

class SystemSetting(Base):
    """System settings and configuration"""
    __tablename__ = 'system_settings'
    
    key = Column(String, primary_key=True)
    value = Column(String, nullable=False)
    data_type = Column(String, default='string')
    description = Column(Text)
    is_public = Column(Boolean, default=False)
    updated_at = Column(DateTime, default=utc_now)
    updated_by = Column(String)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('data_type IN ("string", "integer", "float", "boolean", "json")', name='check_data_type'),
    )
    
    def get_typed_value(self) -> Any:
        """Get value converted to appropriate type"""
        if self.data_type == 'integer':
            return int(self.value)
        elif self.data_type == 'float':
            return float(self.value)
        elif self.data_type == 'boolean':
            return self.value.lower() in ('true', '1', 'yes', 'on')
        elif self.data_type == 'json':
            try:
                return json.loads(self.value)
            except (json.JSONDecodeError, TypeError):
                return None
        return self.value

class SchemaVersion(Base):
    """Schema version tracking for migrations"""
    __tablename__ = 'schema_version'
    
    version = Column(String, primary_key=True)
    applied_at = Column(DateTime, default=utc_now)
    description = Column(Text)

# Database connection and session management
class DatabaseManager:
    """Database manager for Voxify platform"""
    
    def __init__(self, database_url: str = "sqlite:///data/voxify.db"):
        """
        Initialize database manager
        
        Parameters
        ----------
        database_url : str
            SQLAlchemy database URL
        """
        self.engine = create_engine(
            database_url,
            echo=False,  # Set to True for SQL debugging
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {}
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        Base.metadata.drop_all(bind=self.engine)
    
    def get_session(self):
        """Get a database session"""
        return self.SessionLocal()
    
    def init_default_data(self):
        """Initialize default system data"""
        session = self.get_session()
        try:
            # Add default system settings
            default_settings = [
                SystemSetting(key='max_voice_sample_size_mb', value='50', data_type='integer', 
                            description='Maximum voice sample file size in MB', is_public=True),
                SystemSetting(key='max_synthesis_text_length', value='1000', data_type='integer',
                            description='Maximum text length for synthesis', is_public=True),
                SystemSetting(key='default_sample_rate', value='22050', data_type='integer',
                            description='Default audio sample rate', is_public=True),
                SystemSetting(key='cache_expiry_days', value='30', data_type='integer',
                            description='Cache expiration time in days', is_public=False),
                SystemSetting(key='max_concurrent_training_jobs', value='3', data_type='integer',
                            description='Maximum concurrent training jobs', is_public=False),
                SystemSetting(key='maintenance_mode', value='false', data_type='boolean',
                            description='System maintenance mode flag', is_public=True)
            ]
            
            for setting in default_settings:
                existing = session.query(SystemSetting).filter_by(key=setting.key).first()
                if not existing:
                    session.add(setting)
            
            # Add schema version
            existing_version = session.query(SchemaVersion).filter_by(version='1.0.0').first()
            if not existing_version:
                version = SchemaVersion(
                    version='1.0.0',
                    description='Initial database schema for Voxify platform'
                )
                session.add(version)
            
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

# Utility functions
def get_database_manager(database_url: str = None) -> DatabaseManager:
    """Get database manager instance"""
    if database_url is None:
        import os
        database_url = os.getenv('DATABASE_URL', 'sqlite:///data/voxify.db')
    
    return DatabaseManager(database_url) 