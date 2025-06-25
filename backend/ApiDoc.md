# Voxify RESTful API Architecture Documentation

## Architecture Overview

Voxify adopts a pure RESTful architecture design, providing a unified, easy-to-use, and scalable HTTP/JSON API for all clients and internal services. The system features a three-layer voice processing architecture: Voice Clone Management for creating custom voice models, Voice Model Library for organizing and selecting voices, and Job Management for synthesis execution.

- **Flask REST API Gateway**: Provides HTTP/REST interfaces for web clients, mobile applications, admin interfaces, and internal service communication.
- **Voice Processing Pipeline**: Voice Sample → Voice Clone → Voice Model Library → Synthesis Job → Audio Output
- **RESTful Client Support**: All clients and internal services use RESTful API (HTTP/JSON) for communication.

### Architecture Layers
```
┌─────────────────────────────────────────────────────────────┐
│                    Client Layer                             │
│  Web Apps │  Mobile Apps │  Admin UI  │  Internal Services  │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                Flask REST API Gateway (HTTP/JSON)           │
│   - Authentication & Authorization                          │
│   - Request Validation                                      │
│   - Response Formatting                                    │
│   - Admin Interface                                        │
│   - Middleware Layer:                                      │
│     * JWT Authentication                                   │
│     * CORS & Rate Limiting                                 │
│     * Error Handling                                       │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                 Data Storage Layer                          │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   SQLite    │  │    Redis    │  │File Storage │         │
│  │User Data &  │  │ Cache &     │  │   MinIO/S3  │         │
│  │ Metadata    │  │ Sessions    │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐                          │
│  │Model Storage│  │Vector Database│                        │
│  │Voice Models │  │ Chroma/Qdrant │                        │
│  │ & AI Assets │  │Voice Embeddings│                       │
│  └─────────────┘  └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                 Message Queue Layer                         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐                          │
│  │   Celery    │  │  RabbitMQ   │                          │
│  │Async Tasks  │  │Message Broker│                         │
│  │ Worker      │  │             │                          │
│  └─────────────┘  └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Stack Details

### RESTful API Gateway Layer
```python
# Core Framework
Flask==2.3.3
Flask-RESTful==0.3.10
Flask-JWT-Extended==4.5.3
Flask-CORS==4.0.0
Flask-Limiter==3.5.0
```

### Data Storage & Infrastructure
```python
# Database
SQLite==3.40.0
sqlite3 (Python built-in)

# Vector Database
chromadb==0.4.15
sentence-transformers==2.2.2  # for voice embeddings

# File Storage
MinIO (S3-compatible)

# Message Queue
RabbitMQ==3.11
Celery==5.3.1

# Monitoring & Logging
prometheus-client==0.17.1
structlog==23.1.0
```

---

## RESTful API Endpoints 

### Authentication
- POST /api/v1/auth/register
- POST /api/v1/auth/login
- POST /api/v1/auth/refresh

### Voice Sample Management
- POST /api/v1/voice/samples
  - Upload voice sample for processing
  - Accepts audio files (WAV, MP3)
  - Stores file temporarily and metadata in SQLite
  - Returns sample_id and processing status

- GET /api/v1/voice/samples
  - List all voice samples for the authenticated user
  - Includes pagination support
  - Returns sample metadata and processing status

- GET /api/v1/voice/samples/{sample_id}
  - Get details of a specific voice sample
  - Includes processing status and metadata

- DELETE /api/v1/voice/samples/{sample_id}
  - Remove a voice sample
  - Deletes both metadata from SQLite and embedding from ChromaDB

### Voice Clone Management
- POST /api/v1/voice/clones
  - Generate a new voice clone from processed samples
  - Uses embeddings from ChromaDB
  - Accepts array of sample_ids
  - Returns clone_id and generation status

- GET /api/v1/voice/clones
  - List all voice clones for the user
  - Includes generation status and metadata

- GET /api/v1/voice/clones/{clone_id}
  - Get details of a specific voice clone
  - Includes generation status and sample references

- DELETE /api/v1/voice/clones/{clone_id}
  - Remove a voice clone

- POST /api/v1/voice/clones/{clone_id}/select
  - Set a voice clone as the active one for synthesis
  - Returns success status

### Storage Architecture
- **Temporary File Storage**: Audio files stored temporarily during processing
- **SQLite Database**: Stores voice sample and clone metadata
- **ChromaDB**: Stores voice embeddings for similarity search and voice matching

### Voice Model Library
- GET /api/v1/voice/models
  - Get all available voice models (preset + custom clones)
  - Supports filtering by tags, language, type, and owner
  - Includes preview samples and quality scores
  - Returns paginated list with metadata
- GET /api/v1/voice/models/{model_id}
  - Get detailed information about a specific voice model
  - Includes model metadata, tags, and sample audio
  - Returns model configuration and usage statistics
- GET /api/v1/voice/models/{model_id}/preview
  - Get preview audio sample for a voice model
  - Returns audio file or streaming URL
  - Used for voice model auditioning before selection
- POST /api/v1/voice/models/{model_id}/favorite
  - Add a voice model to user's favorites
  - Supports both preset and custom models
  - Returns updated favorite status

### Job Management
- GET /api/v1/job
  - List synthesis jobs with filtering, sorting, and pagination
  - Query parameters: status, user_id, voice_model_id, text_search, limit, offset, sort_by, sort_order, include_text
  - Supports filtering by status (pending, processing, completed, failed, cancelled)
  - Returns paginated list with metadata
- POST /api/v1/job
  - Create a new synthesis job
  - Requires text_content and voice_model_id
  - Supports configuration for speed, pitch, volume, output format, sample rate
  - Returns job details with unique job_id
- GET /api/v1/job/{job_id}
  - Get detailed information about a specific synthesis job
  - Includes job status, progress, configuration, and results
  - Access control: users can only see their own jobs
- PUT /api/v1/job/{job_id}
  - Update a synthesis job (only allowed for pending jobs)
  - Can update text_content, speed, pitch, volume, output_format, sample_rate, config
  - Returns updated job details

- PATCH /api/v1/job/{job_id}
  - Partial update of a synthesis job
  - Supports status updates and progress tracking
  - Can update status, progress, error_message, output_file_path, duration
- DELETE /api/v1/job/{job_id}
  - Delete a synthesis job
  - Only allowed for jobs with status: pending, completed, failed, cancelled
  - Cannot delete jobs that are currently processing
- GET /api/v1/job/{job_id}/progress
  - Stream real-time progress updates for a synthesis job
  - Returns Server-Sent Events (SSE) stream
  - Provides continuous progress updates until job completion

### File Management
- GET /api/v1/files/audio/{file_id}
- GET /api/v1/files/{file_id}/info

### Admin Endpoints
- GET /api/v1/admin/stats
- GET /api/v1/admin/users
- GET/PUT /api/v1/admin/config

---

## Job Management API Examples

### List Jobs with Filtering and Pagination
```http
GET /api/v1/job?status=completed&limit=10&offset=0&sort_by=created_at&sort_order=desc&include_text=false
Authorization: Bearer <access_token>
```

**Response:**
```json
{
    "success": true,
    "data": [
        {
            "id": "job_1234567890",
            "user_id": "user_123",
            "voice_model_id": "vm_456",
            "text_content": "Hello world...",
            "text_hash": "a1b2c3d4...",
            "text_language": "en-US",
            "text_length": 123,
            "word_count": 20,
            "status": "completed",
            "progress": 100.0,
            "config": {
                "speed": 1.0,
                "pitch": 1.0,
                "volume": 1.0
            },
            "output_format": "wav",
            "sample_rate": 22050,
            "output_file_path": "/audio/job_1234567890.wav",
            "duration": 5.2,
            "created_at": "2024-01-15T10:30:00Z",
            "updated_at": "2024-01-15T10:35:00Z"
        }
    ],
    "meta": {
        "pagination": {
            "total_count": 45,
            "limit": 10,
            "offset": 0,
            "has_more": true
        }
    },
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### Create New Synthesis Job
```http
POST /api/v1/job
Content-Type: application/json
Authorization: Bearer <access_token>

{
    "text_content": "Hello world, this is a test synthesis job.",
    "voice_model_id": "vm_456",
    "text_language": "en-US",
    "output_format": "wav",
    "sample_rate": 22050,
    "speed": 1.2,
    "pitch": 1.0,
    "volume": 1.0,
    "config": {
        "include_timestamps": true,
        "timestamp_granularity": "word"
    }
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "id": "job_1234567890",
        "user_id": "user_123",
        "voice_model_id": "vm_456",
        "text_content": "Hello world, this is a test synthesis job.",
        "text_hash": "a1b2c3d4e5f6...",
        "text_language": "en-US",
        "text_length": 43,
        "word_count": 8,
        "status": "pending",
        "progress": 0.0,
        "config": "{\"include_timestamps\": true, \"timestamp_granularity\": \"word\"}",
        "output_format": "wav",
        "sample_rate": 22050,
        "speed": 1.2,
        "pitch": 1.0,
        "volume": 1.0,
        "created_at": "2024-01-15T10:30:00Z",
        "updated_at": "2024-01-15T10:30:00Z"
    },
    "message": "Synthesis job created successfully",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### Update Job Configuration
```http
PUT /api/v1/job/job_1234567890
Content-Type: application/json
Authorization: Bearer <access_token>

{
    "speed": 1.5,
    "pitch": 1.1,
    "volume": 0.9,
    "output_format": "mp3",
    "sample_rate": 44100
}
```

### Stream Job Progress (Server-Sent Events)
```http
GET /api/v1/job/job_1234567890/progress
Authorization: Bearer <access_token>
Accept: text/event-stream
```

**Response Stream:**
```
data: {"progress": 0.0, "status": "pending", "message": "Job queued for processing"}

data: {"progress": 25.0, "status": "processing", "message": "Text preprocessing completed"}

data: {"progress": 75.0, "status": "processing", "message": "Audio synthesis in progress"}

data: {"progress": 100.0, "status": "completed", "message": "Synthesis completed", "output_file_path": "/audio/job_1234567890.wav", "duration": 5.2}

data: {"event": "complete"}
```

### Delete Job
```http
DELETE /api/v1/job/job_1234567890
Authorization: Bearer <access_token>
```

**Response:**
```json
{
    "success": true,
    "message": "Job deleted successfully",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## API Response Format

All API responses follow a unified format:
```json
{
    "success": true/false,
    "data": {},
    "message": "Operation description",
    "error": {
        "code": "ERROR_CODE",
        "message": "Error description",
        "details": "Detailed information"
    }
}
```

---

## Pagination
All list APIs support pagination:
```json
{
    "pagination": {
        "page": 1,
        "page_size": 20,
        "total_count": 100,
        "total_pages": 5,
        "has_next": true,
        "has_prev": false
    }
}
```

---

## Authentication
- REST API uses JWT Bearer Token for all user and service authentication.
- Admin endpoints require additional permission verification.

---

## Rate Limiting
- Maximum 100 requests per minute per user
- File upload limit: 50MB/file, maximum 100 files per day
- Speech synthesis limit: maximum 1000 syntheses per day

---

## Project Structure Example
```
backend/
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── auth.py          # Authentication endpoints
│   │   ├── voice.py         # Voice sample & clone endpoints
│   │   ├── models.py        # Voice model library endpoints
│   │   ├── jobs.py          # Job management endpoints
│   │   ├── files.py         # File management endpoints
│   │   └── admin.py         # Admin endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py      # Request models
│   │   └── responses.py     # Response models
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── auth.py          # JWT authentication middleware
│   │   └── error_handler.py # Error handling
│   └── utils/
│       ├── __init__.py
│       ├── validators.py    # Request validation
│       └── formatters.py    # Response formatting
├── tests/
├── requirements.txt
└── app.py
```

---

## Voice Model Library API Examples

### Get Available Voice Models with Filtering
```http
GET /api/v1/voice/models?tags=female,gentle&language=zh-CN&type=preset&limit=10&offset=0
Authorization: Bearer <access_token>
```

**Response:**
```json
{
    "success": true,
    "data": [
        {
            "voice_model_id": "preset_gentle_female_001",
            "name": "Gentle Female Voice",
            "type": "preset",
            "tags": ["female", "gentle", "sweet", "young"],
            "language": "zh-CN",
            "description": "A soft and warm female voice suitable for storytelling and meditation",
            "sample_audio_url": "/api/v1/files/audio/samples/gentle_female.wav",
            "preview_text": "Hello, this is a preview of the gentle female voice.",
            "quality_score": 0.95,
            "usage_count": 1250,
            "is_favorite": false,
            "created_at": "2024-01-01T00:00:00Z"
        },
        {
            "voice_model_id": "custom_user123_clone",
            "name": "My Personal Voice",
            "type": "custom",
            "tags": ["personal", "custom"],
            "language": "en-US",
            "description": "Personal voice clone created from user samples",
            "owner_id": "user123",
            "clone_id": "vc_456",
            "quality_score": 0.88,
            "usage_count": 25,
            "is_favorite": true,
            "created_at": "2024-01-15T10:30:00Z"
        }
    ],
    "meta": {
        "pagination": {
            "total_count": 45,
            "limit": 10,
            "offset": 0,
            "has_more": true
        },
        "filters": {
            "available_tags": ["female", "male", "gentle", "energetic", "mature", "young"],
            "available_languages": ["zh-CN", "en-US", "ja-JP", "ko-KR"],
            "available_types": ["preset", "custom"]
        }
    },
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### Get Voice Model Details
```http
GET /api/v1/voice/models/preset_gentle_female_001
Authorization: Bearer <access_token>
```

**Response:**
```json
{
    "success": true,
    "data": {
        "voice_model_id": "preset_gentle_female_001",
        "name": "Gentle Female Voice",
        "type": "preset",
        "tags": ["female", "gentle", "sweet", "young"],
        "language": "zh-CN",
        "description": "A soft and warm female voice suitable for storytelling and meditation",
        "sample_audio_url": "/api/v1/files/audio/samples/gentle_female.wav",
        "preview_text": "Hello, this is a preview of the gentle female voice.",
        "quality_score": 0.95,
        "usage_count": 1250,
        "is_favorite": false,
        "model_config": {
            "sample_rate": 22050,
            "supported_formats": ["wav", "mp3", "flac"],
            "max_text_length": 10000,
            "supports_emotions": ["neutral", "happy", "sad", "excited"],
            "supports_speed_range": [0.5, 3.0],
            "supports_pitch_range": [0.5, 2.0]
        },
        "statistics": {
            "average_synthesis_time": 2.3,
            "success_rate": 0.992,
            "user_rating": 4.7
        },
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-15T08:00:00Z"
    },
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### Get Voice Model Preview
```http
GET /api/v1/voice/models/preset_gentle_female_001/preview
Authorization: Bearer <access_token>
```

**Response:**
```json
{
    "success": true,
    "data": {
        "preview_audio_url": "/api/v1/files/audio/previews/preset_gentle_female_001.wav",
        "preview_text": "Hello, this is a preview of the gentle female voice.",
        "duration": 3.2,
        "format": "wav",
        "sample_rate": 22050
    },
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### Add Voice Model to Favorites
```http
POST /api/v1/voice/models/preset_gentle_female_001/favorite
Authorization: Bearer <access_token>
```

**Response:**
```json
{
    "success": true,
    "data": {
        "voice_model_id": "preset_gentle_female_001",
        "is_favorite": true,
        "favorites_count": 1
    },
    "message": "Voice model added to favorites",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### Create Synthesis Job with Selected Voice Model
```http
POST /api/v1/job
Content-Type: application/json
Authorization: Bearer <access_token>

{
    "text_content": "Hello world, this synthesis uses a selected voice model from the library.",
    "voice_model_id": "preset_gentle_female_001",
    "text_language": "en-US",
    "output_format": "wav",
    "sample_rate": 22050,
    "speed": 1.0,
    "pitch": 1.0,
    "volume": 1.0,
    "config": {
        "emotion": "happy",
        "include_timestamps": true
    }
}
```

---

## Voice Sample Training

### Train Voice Clone Model
```http
POST /api/v1/voice/samples/{sample_id}/train
Content-Type: application/json
Authorization: Bearer <access_token>

{
    "config": {
        "epochs": 100,
        "learning_rate": 0.001,
        "batch_size": 32,
        "enable_timestamp_alignment": true
    }
}
```

**Response**:
```json
{
    "success": true,
    "data": {
        "training_job_id": "job_train_1234567890",
        "estimated_duration": 1800,
        "status": "pending",
        "features": {
            "timestamp_alignment_enabled": true,
            "voice_quality_enhancement": true
        }
    },
    "message": "Voice training job created with timestamp alignment support"
}
```

---

## Complete API Workflows

### Workflow 1: Text-to-Speech with Preset Voice Models

**Overview**: User selects from pre-trained voice models in the library and synthesizes speech.

1. **User Registration**
```bash
curl -X POST http://localhost:5000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "SecurePass123!", "first_name": "John", "last_name": "Doe"}'
```

2. **User Login**
```bash
curl -X POST http://localhost:5000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "SecurePass123!"}'
```

3. **Create Synthesis Job (using default voice model)**
```bash
curl -X POST http://localhost:5000/api/v1/job \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "text_content": "Hello world, this uses the default voice model!",
    "voice_model_id": "default_voice_001",
    "output_format": "wav",
    "speed": 1.0,
    "pitch": 1.0,
    "volume": 1.0
  }'
```

> **Note**: Voice Model Library (`/api/v1/voice/models/`) for browsing and previewing voices will be implemented in future versions. Currently using default voice models.

4. **Monitor Synthesis Progress (Real-time)**
```bash
curl -X GET "http://localhost:5000/api/v1/job/job_1234567890/progress" \
  -H "Authorization: Bearer <access_token>" \
  -H "Accept: text/event-stream"
```

5. **Get Synthesis Result**
```bash
curl -X GET "http://localhost:5000/api/v1/job/job_1234567890" \
  -H "Authorization: Bearer <access_token>"
```

6. **Download Audio File via File Management**
```bash
curl -X GET "http://localhost:5000/api/v1/files/audio/job_1234567890" \
  -H "Authorization: Bearer <access_token>" \
  -o synthesized_audio.wav
```



---

### Workflow 2: Custom Voice Cloning and Synthesis

**Overview**: User uploads voice samples, creates a custom voice clone, and synthesizes speech using their cloned voice.

1. **User Registration**
```bash
curl -X POST http://localhost:5000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "SecurePass123!", "first_name": "Jane", "last_name": "Smith"}'
```

2. **User Login**
```bash
curl -X POST http://localhost:5000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "SecurePass123!"}'
```

3. **Upload Voice Sample**
```bash
curl -X POST http://localhost:5000/api/v1/voice/samples \
  -H "Authorization: Bearer <access_token>" \
  -F "file=@voice_sample.wav" \
  -F "name=My Personal Voice Sample" \
  -F "description=High quality recording for voice cloning" \
  -F "language=en-US"
```

4. **Process Voice Sample**
```bash
curl -X POST http://localhost:5000/api/v1/voice/samples/vs_1234567890/process \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{}'
```

5. **Monitor Processing Progress**
```bash
curl -X GET "http://localhost:5000/api/v1/voice/samples/vs_1234567890" \
  -H "Authorization: Bearer <access_token>"
```

6. **Create Voice Clone (after processing completes)**
```bash
curl -X POST http://localhost:5000/api/v1/voice/clones \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "sample_ids": ["vs_1234567890"],
    "name": "My Personal Voice Clone",
    "description": "Custom voice clone from my samples"
  }'
```

7. **Monitor Clone Creation**
```bash
curl -X GET "http://localhost:5000/api/v1/voice/clones/vc_1234567890" \
  -H "Authorization: Bearer <access_token>"
```

8. **Create Synthesis Job with Custom Cloned Voice**
```bash
curl -X POST http://localhost:5000/api/v1/job \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "text_content": "Hello world, this is my personal cloned voice speaking!",
    "voice_model_id": "vc_1234567890",
    "output_format": "wav",
    "speed": 1.0,
    "pitch": 1.0,
    "volume": 1.0
  }'
```

9. **Monitor Synthesis Progress (Real-time)**
```bash
curl -X GET "http://localhost:5000/api/v1/job/job_2345678901/progress" \
  -H "Authorization: Bearer <access_token>" \
  -H "Accept: text/event-stream"
```

10. **Get Synthesis Result**
```bash
curl -X GET "http://localhost:5000/api/v1/job/job_2345678901" \
  -H "Authorization: Bearer <access_token>"
```

11. **Download Custom Voice Audio via File Management**
```bash
curl -X GET "http://localhost:5000/api/v1/files/audio/job_2345678901" \
  -H "Authorization: Bearer <access_token>" \
  -o custom_voice_audio.wav
```


---

### Workflow Comparison

| Step | Preset Voice Workflow | Custom Voice Workflow |
|------|----------------------|----------------------|
| **Setup** | Register → Login | Register → Login |
| **Voice Selection** | Use Default Voice | Upload Samples → Process → Create Clone |
| **Synthesis** | Create Job → Monitor → Download | Create Job → Monitor → Download |
| **Time Required** | ~5 minutes | ~30+ minutes (includes processing & cloning) |
| **Quality** | Standard, reliable | Personalized, unique |
| **Use Case** | Quick prototyping, testing | Personal branding, custom applications |

---

## Deployment and Configuration

### Docker Compose Example
```yaml
version: '3.8'

services:
  # Flask REST Gateway
  voxify-backend:
    build: ./backend
    ports:
      - "5000:5000"
    environment:
      - JWT_SECRET=your-secret-key
      - DATABASE_URL=sqlite:///data/voxify.db
      - VECTOR_DB_PATH=/data/chroma_db
    volumes:
      - voxify_data:/data
      - voxify_uploads:/uploads
      - voxify_models:/models
  
  # Message Queue (for async processing)
  rabbitmq:
    image: rabbitmq:3.11-management
    environment:
      - RABBITMQ_DEFAULT_USER=voxify
      - RABBITMQ_DEFAULT_PASS=voxify_pass
    ports:
      - "5672:5672"
      - "15672:15672"  # Management UI
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
  
  # Celery Worker (for background tasks)
  celery-worker:
    build: ./backend
    command: celery -A app.celery worker --loglevel=info
    environment:
      - DATABASE_URL=sqlite:///data/voxify.db
      - CELERY_BROKER_URL=pyamqp://voxify:voxify_pass@rabbitmq:5672//
    volumes:
      - voxify_data:/data
      - voxify_uploads:/uploads
      - voxify_models:/models
    depends_on:
      - rabbitmq

volumes:
  voxify_data:        # SQLite database and Chroma vector DB
  voxify_uploads:     # Voice sample uploads
  voxify_models:      # Trained AI models
  rabbitmq_data:      # RabbitMQ data
```

---


```

## Key Features

- **Preset Voice Library**: Ready-to-use professional voice models with various characteristics
- **Custom Voice Cloning**: Create personalized voice models from user audio samples  
- **Unified Voice Selection**: Browse and select from both preset and custom voices in one interface
- **Advanced Synthesis Control**: Fine-tune speed, pitch, volume, and emotional expression
- **Real-time Progress Tracking**: Monitor synthesis jobs with Server-Sent Events
- **Quality Assurance**: Built-in quality scoring and validation for all voice models

## AI Acknowledgment
This file was completed with the help of Cursor, which significantly streamlined the process by handling repetitive formatting and ad hoc tasks efficiently. Tools like Cursor reduce the need for manual, repetitive work—something most people understandably want to avoid：）
