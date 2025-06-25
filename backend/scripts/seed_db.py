"""
Database Seed Script
Creates sample data for testing and development
"""

import os
import sys
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_database_manager
from database.models import (
    User, VoiceSample, VoiceModel, SynthesisJob,
    PhonemeAlignment, UsageStat, SystemSetting
)
from api.utils.password import hash_password

def utc_now():
    """Get current UTC time with timezone info"""
    return datetime.now(timezone.utc)

def create_test_users(session):
    """Create test users"""
    users = [
        {
            'email': 'test@example.com',
            'password': 'Test123!',
            'first_name': 'Test',
            'last_name': 'User',
            'is_active': True,
            'email_verified': True,
            'last_login_at': utc_now()
        },
        {
            'email': 'admin@example.com',
            'password': 'Admin123!',
            'first_name': 'Admin',
            'last_name': 'User',
            'is_active': True,
            'email_verified': True,
            'last_login_at': utc_now()
        }
    ]
    
    created_users = []
    for user_data in users:
        password = user_data.pop('password')
        user = User(
            password_hash=hash_password(password),
            **user_data
        )
        session.add(user)
        created_users.append(user)
    
    return created_users

def create_voice_samples(session, users):
    """Create voice samples for users"""
    samples = []
    for user in users:
        for i in range(2):  # 2 samples per user
            sample = VoiceSample(
                user_id=user.id,
                name=f'{user.first_name}\'s Voice Sample {i+1}',
                description='This is a test voice sample',
                file_path=f'data/files/samples/{user.id}/sample_{i+1}.wav',
                file_size=1024 * 1024,  # 1MB
                original_filename=f'original_sample_{i+1}.wav',
                format='wav',
                duration=5.0,
                sample_rate=22050,
                channels=1,
                status='ready',
                quality_score=9.5,
                language='en-US',
                is_public=True,
                gender='male' if i % 2 == 0 else 'female',
                tags_list=['test', 'english', 'high-quality'],
                processing_start_time=utc_now() - timedelta(days=1),
                processing_end_time=utc_now() - timedelta(days=1, minutes=5)
            )
            session.add(sample)
            samples.append(sample)
    
    return samples

def create_voice_models(session, samples):
    """Create voice models for samples"""
    models = []
    for sample in samples:
        model = VoiceModel(
            voice_sample_id=sample.id,
            name=f'Model for {sample.name}',
            description='Model trained from voice sample',
            model_path=f'data/models/{sample.id}/model.pth',
            model_type='tacotron2',
            model_size=256 * 1024 * 1024,  # 256MB
            model_version='1.0',
            training_config_dict={
                'epochs': 1000,
                'batch_size': 32,
                'learning_rate': 0.001
            },
            training_epochs=1000,
            learning_rate=0.001,
            batch_size=32,
            dataset_size=100,
            training_status='completed',
            training_progress=1.0,
            training_start_time=utc_now() - timedelta(days=2),
            training_end_time=utc_now() - timedelta(days=1),
            mel_loss=0.1,
            validation_score=0.95,
            mos_score=4.5,
            similarity_score=0.98,
            is_active=True,
            deployment_status='online'
        )
        session.add(model)
        models.append(model)
    
    return models

def create_synthesis_jobs(session, users, models):
    """Create synthesis jobs"""
    jobs = []
    texts = [
        "Hello, this is a test voice synthesis.",
        "Artificial Intelligence is changing our lives.",
        "Voice synthesis technology is becoming more natural.",
        "Have a great day!"
    ]
    
    for user in users:
        for model in models:
            for i, text in enumerate(texts):
                # Create jobs with different statuses
                status = ['completed', 'processing', 'pending', 'failed'][i % 4]
                progress = 1.0 if status == 'completed' else (
                    0.0 if status == 'pending' else 0.5
                )
                
                job = SynthesisJob(
                    user_id=user.id,
                    voice_model_id=model.id,
                    text_content=text,
                    text_hash=str(hash(text)),
                    text_language='en-US',
                    text_length=len(text),
                    word_count=len(text.split()),
                    config_dict={
                        'speed': 1.0,
                        'pitch': 1.0,
                        'volume': 1.0
                    },
                    output_format='wav',
                    sample_rate=22050,
                    status=status,
                    progress=progress,
                                    started_at=utc_now() - timedelta(hours=1) if status != 'pending' else None,
                completed_at=utc_now() if status == 'completed' else None
                )
                
                if status == 'completed':
                    job.output_path = f'data/files/synthesis/output/{job.id}.wav'
                    job.output_size = 1024 * 1024  # 1MB
                    job.duration = 5.0
                    job.processing_time_ms = 1500
                    job.queue_time_ms = 500
                elif status == 'failed':
                    job.error_message = "Model inference failed: Out of memory"
                
                session.add(job)
                jobs.append(job)
    
    return jobs

def create_usage_stats(session, users):
    """Create usage statistics"""
    for user in users:
        for i in range(7):  # Last 7 days of data
            date = (utc_now() - timedelta(days=i)).strftime('%Y-%m-%d')
            stat = UsageStat(
                user_id=user.id,
                date=date,
                voice_samples_uploaded=2,
                models_trained=1,
                synthesis_requests=10,
                synthesis_duration=50.0,
                storage_used=1024 * 1024 * 100,  # 100MB
                api_calls_auth=5,
                api_calls_voice=15,
                api_calls_tts=10,
                api_calls_admin=2,
                avg_synthesis_time=1.5,
                cache_hit_rate=0.8
            )
            session.add(stat)
            session.flush()  # Flush immediately to ensure relationships are established

def main():
    """Main function to seed the database"""
    print("🌱 Starting to seed test data...")
    
    # Get database connection
    db = get_database_manager()
    session = db.get_session()
    
    try:
        # Create test data
        print("👤 Creating test users...")
        users = create_test_users(session)
        session.commit()
        
        print("🎤 Creating voice samples...")
        samples = create_voice_samples(session, users)
        session.commit()
        
        print("🤖 Creating voice models...")
        models = create_voice_models(session, samples)
        session.commit()
        
        print("🎯 Creating synthesis jobs...")
        jobs = create_synthesis_jobs(session, users, models)
        session.commit()
        
        print("📊 Creating usage statistics...")
        create_usage_stats(session, users)
        session.commit()
        
        print("✅ Test data creation successful!")
        
        # Print summary
        print("\n📝 Data Summary:")
        print(f"- Users: {len(users)}")
        print(f"- Voice Samples: {len(samples)}")
        print(f"- Voice Models: {len(models)}")
        print(f"- Synthesis Jobs: {len(jobs)}")
        print(f"- Usage Stats per User: 7 days")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Error: {str(e)}")
        raise
    finally:
        session.close()

if __name__ == '__main__':
    main() 