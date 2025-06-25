-- Voxify Platform Database Schema
-- SQLite + Vector Database Hybrid Storage Design
-- Version: 1.0

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- =============================================================================
-- Core User Management Tables
-- =============================================================================

-- Users table for authentication and account management
CREATE TABLE users (
    id TEXT PRIMARY KEY,                    -- UUID
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    first_name TEXT,
    last_name TEXT,
    storage_used_bytes INTEGER DEFAULT 0,   -- Current storage usage
    is_active BOOLEAN DEFAULT TRUE,
    email_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP,
    
    CHECK (storage_used_bytes >= 0)
);

-- Indexes for users table
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_active ON users(is_active);

-- =============================================================================
-- Voice Sample Management Tables
-- =============================================================================

-- Voice samples table for uploaded audio files and metadata
CREATE TABLE voice_samples (
    id TEXT PRIMARY KEY,                    -- UUID
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    
    -- File information
    file_path TEXT NOT NULL,               -- Storage path
    file_size INTEGER NOT NULL,            -- Size in bytes
    original_filename TEXT,
    file_hash TEXT,                        -- SHA256 hash for deduplication
    
    -- Audio properties
    format TEXT NOT NULL,                  -- wav, mp3, flac, etc.
    duration REAL NOT NULL,                -- Duration in seconds
    sample_rate INTEGER NOT NULL,          -- Sample rate in Hz
    channels INTEGER DEFAULT 1,            -- Number of audio channels
    bit_depth INTEGER,                     -- Bit depth
    
    -- Language and quality metrics
    language TEXT DEFAULT 'en-US',
    quality_score REAL,                    -- 0-10 audio quality score
    noise_level REAL,                      -- Noise level assessment
    clarity_score REAL,                    -- Speech clarity score
    signal_to_noise_ratio REAL,           -- SNR measurement
    
    -- Processing status
    status TEXT DEFAULT 'uploaded',        -- uploaded, processing, ready, failed
    processing_error TEXT,
    processing_start_time TIMESTAMP,
    processing_end_time TIMESTAMP,
    
    -- Vector database associations
    voice_embedding_id TEXT,               -- Reference to Chroma voice embedding
    speaker_embedding_id TEXT,             -- Reference to speaker identity embedding
    
    -- Metadata and categorization
    tags TEXT,                             -- JSON array of user tags
    is_public BOOLEAN DEFAULT FALSE,       -- Public sharing flag
    gender TEXT,                           -- Detected gender (optional)
    age_group TEXT,                        -- Detected age group (optional)
    accent TEXT,                           -- Detected accent (optional)
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    CHECK (status IN ('uploaded', 'processing', 'ready', 'failed')),
    CHECK (duration > 0),
    CHECK (sample_rate > 0),
    CHECK (channels > 0),
    CHECK (quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 10))
);

-- Indexes for voice_samples table
CREATE INDEX idx_voice_samples_user_id ON voice_samples(user_id);
CREATE INDEX idx_voice_samples_status ON voice_samples(status);
CREATE INDEX idx_voice_samples_language ON voice_samples(language);
CREATE INDEX idx_voice_samples_quality ON voice_samples(quality_score);
CREATE INDEX idx_voice_samples_created ON voice_samples(created_at);
CREATE INDEX idx_voice_samples_file_hash ON voice_samples(file_hash);

-- =============================================================================
-- Voice Model Management Tables  
-- =============================================================================

-- Voice models table for trained AI models
CREATE TABLE voice_models (
    id TEXT PRIMARY KEY,                    -- UUID
    voice_sample_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    
    -- Model file information
    model_path TEXT NOT NULL,              -- Model file storage path
    model_type TEXT DEFAULT 'tacotron2',   -- Model architecture type
    model_size INTEGER,                    -- Model file size in bytes
    model_version TEXT DEFAULT '1.0',      -- Model version
    model_hash TEXT,                       -- Model file hash
    
    -- Training configuration
    training_config TEXT,                  -- JSON training parameters
    training_epochs INTEGER,
    learning_rate REAL,
    batch_size INTEGER,
    dataset_size INTEGER,                  -- Training dataset size
    
    -- Training status and progress
    training_status TEXT DEFAULT 'pending', -- pending, training, completed, failed
    training_progress REAL DEFAULT 0.0,     -- 0.0-1.0 progress
    training_start_time TIMESTAMP,
    training_end_time TIMESTAMP,
    training_error TEXT,
    training_logs_path TEXT,               -- Training logs file path
    
    -- Quality metrics and evaluation
    quality_metrics TEXT,                   -- JSON quality metrics
    mel_loss REAL,                         -- Training loss
    validation_score REAL,                 -- Validation score
    mos_score REAL,                        -- Mean Opinion Score
    similarity_score REAL,                 -- Voice similarity score
    
    -- Model status and management
    is_active BOOLEAN DEFAULT TRUE,
    is_default BOOLEAN DEFAULT FALSE,       -- User's default model
    deployment_status TEXT DEFAULT 'offline', -- offline, online, deploying
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (voice_sample_id) REFERENCES voice_samples (id) ON DELETE CASCADE,
    CHECK (training_status IN ('pending', 'training', 'completed', 'failed')),
    CHECK (training_progress >= 0.0 AND training_progress <= 1.0),
    CHECK (deployment_status IN ('offline', 'online', 'deploying'))
);

-- Indexes for voice_models table
CREATE INDEX idx_voice_models_sample_id ON voice_models(voice_sample_id);
CREATE INDEX idx_voice_models_status ON voice_models(training_status);
CREATE INDEX idx_voice_models_active ON voice_models(is_active);
CREATE INDEX idx_voice_models_deployment ON voice_models(deployment_status);

-- =============================================================================
-- TTS Synthesis Management Tables
-- =============================================================================

-- Synthesis jobs table for TTS requests and results
CREATE TABLE synthesis_jobs (
    id TEXT PRIMARY KEY,                    -- UUID
    user_id TEXT NOT NULL,
    voice_model_id TEXT NOT NULL,
    
    -- Input text information
    text_content TEXT NOT NULL,
    text_hash TEXT NOT NULL,               -- Hash for caching
    text_language TEXT DEFAULT 'en-US',
    text_length INTEGER,                   -- Character count
    word_count INTEGER,                    -- Word count
    
    -- Synthesis configuration
    config TEXT,                           -- JSON synthesis parameters
    output_format TEXT DEFAULT 'wav',      -- wav, mp3, flac
    sample_rate INTEGER DEFAULT 22050,
    speed REAL DEFAULT 1.0,                -- Speed adjustment (0.5-2.0)
    pitch REAL DEFAULT 1.0,                -- Pitch adjustment (0.5-2.0)
    volume REAL DEFAULT 1.0,               -- Volume adjustment (0.1-2.0)
    
    -- Output information
    output_path TEXT,                      -- Generated audio file path
    output_size INTEGER,                   -- File size in bytes
    duration REAL,                         -- Audio duration in seconds
    
    -- Timing alignment data (project requirement)
    word_timestamps TEXT,                   -- JSON: [{"word": "hello", "start": 0.0, "end": 0.5}]
    syllable_timestamps TEXT,               -- JSON: [{"syllable": "hel", "start": 0.0, "end": 0.25}]
    phoneme_timestamps TEXT,                -- JSON: [{"phoneme": "h", "start": 0.0, "end": 0.1}]
    
    -- Job status and progress
    status TEXT DEFAULT 'pending',          -- pending, processing, completed, failed, cancelled
    progress REAL DEFAULT 0.0,             -- 0.0-1.0 progress
    error_message TEXT,
    processing_node TEXT,                  -- Which worker processed this job
    
    -- Caching information
    cache_hit BOOLEAN DEFAULT FALSE,        -- Whether this was served from cache
    cached_result_id TEXT,                  -- Source cache record ID
    
    -- Performance metrics
    processing_time_ms INTEGER,            -- Processing time in milliseconds
    queue_time_ms INTEGER,                 -- Time spent in queue
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    FOREIGN KEY (voice_model_id) REFERENCES voice_models (id) ON DELETE CASCADE,
    CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    CHECK (progress >= 0.0 AND progress <= 1.0),
    CHECK (speed >= 0.1 AND speed <= 3.0),
    CHECK (pitch >= 0.1 AND pitch <= 3.0),
    CHECK (volume >= 0.0 AND volume <= 3.0)
);

-- Indexes for synthesis_jobs table
CREATE INDEX idx_synthesis_jobs_user_id ON synthesis_jobs(user_id);
CREATE INDEX idx_synthesis_jobs_status ON synthesis_jobs(status);
CREATE INDEX idx_synthesis_jobs_text_hash ON synthesis_jobs(text_hash);
CREATE INDEX idx_synthesis_jobs_model_id ON synthesis_jobs(voice_model_id);
CREATE INDEX idx_synthesis_jobs_created ON synthesis_jobs(created_at);

-- =============================================================================
-- Synthesis Caching Tables
-- =============================================================================

-- Synthesis cache table for performance optimization
CREATE TABLE synthesis_cache (
    id TEXT PRIMARY KEY,                    -- UUID
    text_hash TEXT NOT NULL,               -- Text content hash
    voice_model_id TEXT NOT NULL,
    config_hash TEXT NOT NULL,             -- Configuration parameters hash
    
    -- Cached content
    output_path TEXT NOT NULL,
    duration REAL NOT NULL,
    word_timestamps TEXT,
    syllable_timestamps TEXT,
    phoneme_timestamps TEXT,
    
    -- Cache metadata
    hit_count INTEGER DEFAULT 0,           -- Number of times served from cache
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_size INTEGER,
    
    -- Cache management
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,                  -- Cache expiration time
    is_permanent BOOLEAN DEFAULT FALSE,    -- Whether this cache entry should persist
    
    FOREIGN KEY (voice_model_id) REFERENCES voice_models (id) ON DELETE CASCADE,
    CHECK (duration > 0),
    CHECK (hit_count >= 0)
);

-- Unique index for cache lookup
CREATE UNIQUE INDEX idx_synthesis_cache_unique ON synthesis_cache(text_hash, voice_model_id, config_hash);
CREATE INDEX idx_synthesis_cache_expires ON synthesis_cache(expires_at);
CREATE INDEX idx_synthesis_cache_accessed ON synthesis_cache(last_accessed);

-- =============================================================================
-- Phoneme Alignment Tables (Detailed Timing Data)
-- =============================================================================

-- Detailed phoneme alignments for synthesis jobs
CREATE TABLE phoneme_alignments (
    id TEXT PRIMARY KEY,                    -- UUID
    synthesis_job_id TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,      -- Position in sequence
    
    -- Text unit information
    text_unit TEXT NOT NULL,               -- Word, syllable, or phoneme
    unit_type TEXT NOT NULL,               -- 'word', 'syllable', 'phoneme'
    parent_unit_id TEXT,                   -- Reference to parent unit (e.g., syllable parent for phoneme)
    
    -- Timing information
    start_time REAL NOT NULL,              -- Start time in seconds
    end_time REAL NOT NULL,                -- End time in seconds
    duration REAL NOT NULL,                -- Duration in seconds
    
    -- Audio features
    confidence REAL,                       -- Alignment confidence score (0-1)
    pitch_mean REAL,                       -- Average pitch in Hz
    pitch_std REAL,                        -- Pitch standard deviation
    energy_mean REAL,                      -- Average energy
    energy_std REAL,                       -- Energy standard deviation
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (synthesis_job_id) REFERENCES synthesis_jobs (id) ON DELETE CASCADE,
    CHECK (unit_type IN ('word', 'syllable', 'phoneme')),
    CHECK (start_time >= 0),
    CHECK (end_time > start_time),
    CHECK (duration > 0),
    CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1))
);

-- Indexes for phoneme_alignments table
CREATE INDEX idx_phoneme_alignments_job ON phoneme_alignments(synthesis_job_id);
CREATE INDEX idx_phoneme_alignments_time ON phoneme_alignments(start_time, end_time);
CREATE INDEX idx_phoneme_alignments_type ON phoneme_alignments(unit_type);

-- =============================================================================
-- Usage Statistics and Analytics Tables
-- =============================================================================

-- Daily usage statistics per user
CREATE TABLE usage_stats (
    id TEXT PRIMARY KEY,                    -- UUID
    user_id TEXT NOT NULL,
    date DATE NOT NULL,
    
    -- Usage metrics
    voice_samples_uploaded INTEGER DEFAULT 0,
    models_trained INTEGER DEFAULT 0,
    synthesis_requests INTEGER DEFAULT 0,
    synthesis_duration REAL DEFAULT 0.0,   -- Total synthesis duration in seconds
    storage_used INTEGER DEFAULT 0,        -- Storage used in bytes
    
    -- API call statistics
    api_calls_auth INTEGER DEFAULT 0,
    api_calls_voice INTEGER DEFAULT 0,
    api_calls_tts INTEGER DEFAULT 0,
    api_calls_admin INTEGER DEFAULT 0,
    
    -- Performance metrics
    avg_synthesis_time REAL DEFAULT 0.0,   -- Average synthesis time
    cache_hit_rate REAL DEFAULT 0.0,       -- Cache hit rate (0-1)
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    CHECK (synthesis_duration >= 0),
    CHECK (storage_used >= 0),
    CHECK (cache_hit_rate >= 0 AND cache_hit_rate <= 1)
);

-- Unique constraint and indexes for usage_stats
CREATE UNIQUE INDEX idx_usage_stats_user_date ON usage_stats(user_id, date);
CREATE INDEX idx_usage_stats_date ON usage_stats(date);

-- =============================================================================
-- System Configuration and Settings Tables
-- =============================================================================

-- System settings and configuration
CREATE TABLE system_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    data_type TEXT DEFAULT 'string',       -- string, integer, float, boolean, json
    description TEXT,
    is_public BOOLEAN DEFAULT FALSE,       -- Whether this setting can be read by non-admin users
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by TEXT,                       -- User ID who updated this setting
    
    CHECK (data_type IN ('string', 'integer', 'float', 'boolean', 'json'))
);

-- Insert default system settings
INSERT INTO system_settings (key, value, data_type, description, is_public) VALUES
('max_voice_sample_size_mb', '50', 'integer', 'Maximum voice sample file size in MB', TRUE),
('max_synthesis_text_length', '1000', 'integer', 'Maximum text length for synthesis', TRUE),
('default_sample_rate', '22050', 'integer', 'Default audio sample rate', TRUE),
('cache_expiry_days', '30', 'integer', 'Cache expiration time in days', FALSE),
('max_concurrent_training_jobs', '3', 'integer', 'Maximum concurrent training jobs', FALSE),
('maintenance_mode', 'false', 'boolean', 'System maintenance mode flag', TRUE);

-- =============================================================================
-- Database Maintenance Views
-- =============================================================================

-- View for user statistics summary
CREATE VIEW user_summary AS
SELECT 
    u.id,
    u.email,
    COUNT(DISTINCT vs.id) as voice_samples_count,
    COUNT(DISTINCT vm.id) as models_count,
    COUNT(DISTINCT sj.id) as synthesis_jobs_count,
    u.storage_used_bytes,
    u.created_at,
    u.last_login_at
FROM users u
LEFT JOIN voice_samples vs ON u.id = vs.user_id AND vs.status = 'ready'
LEFT JOIN voice_models vm ON vs.id = vm.voice_sample_id AND vm.is_active = TRUE
LEFT JOIN synthesis_jobs sj ON u.id = sj.user_id AND sj.status = 'completed'
GROUP BY u.id;

-- View for model performance metrics
CREATE VIEW model_performance AS
SELECT 
    vm.id,
    vm.name,
    vm.voice_sample_id,
    vm.training_status,
    vm.quality_metrics,
    vm.similarity_score,
    COUNT(sj.id) as usage_count,
    AVG(sj.processing_time_ms) as avg_processing_time,
    vm.created_at
FROM voice_models vm
LEFT JOIN synthesis_jobs sj ON vm.id = sj.voice_model_id AND sj.status = 'completed'
WHERE vm.is_active = TRUE
GROUP BY vm.id;

-- =============================================================================
-- Database Triggers for Automatic Updates
-- =============================================================================

-- Trigger to update updated_at timestamp
CREATE TRIGGER update_users_timestamp 
    AFTER UPDATE ON users
    FOR EACH ROW
    WHEN NEW.updated_at = OLD.updated_at
BEGIN
    UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_voice_samples_timestamp 
    AFTER UPDATE ON voice_samples
    FOR EACH ROW
    WHEN NEW.updated_at = OLD.updated_at
BEGIN
    UPDATE voice_samples SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_voice_models_timestamp 
    AFTER UPDATE ON voice_models
    FOR EACH ROW
    WHEN NEW.updated_at = OLD.updated_at
BEGIN
    UPDATE voice_models SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Trigger to update cache access time
CREATE TRIGGER update_cache_access_time
    AFTER UPDATE ON synthesis_cache
    FOR EACH ROW
    WHEN NEW.hit_count > OLD.hit_count
BEGIN
    UPDATE synthesis_cache SET last_accessed = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- =============================================================================
-- Initial Data and Setup
-- =============================================================================

-- Create default admin user (password should be changed immediately)
INSERT INTO users (id, email, password_hash, first_name, last_name) VALUES
('admin-000-000-000-000', 'admin@voxify.com', '$2b$12$placeholder_hash', 'System', 'Administrator');

-- Add any additional initialization data here

-- Schema version for migration tracking
CREATE TABLE schema_version (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

INSERT INTO schema_version (version, description) VALUES 
('1.0.0', 'Initial database schema for Voxify platform'); 