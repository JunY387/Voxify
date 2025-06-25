import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import jobService from '../services/job.service';
import authService from '../services/auth.service';
import { createAudioUrl } from '../services/api';
import apiConfig from '../config/api.config';

const Tasks = () => {
  const navigate = useNavigate();
  const [jobs, setJobs] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [playingJobId, setPlayingJobId] = useState(null);
  const [currentlyPlaying, setCurrentlyPlaying] = useState(null);
  const audioRef = useRef(null);

  const loadJobs = useCallback(async () => {
    try {
      const jobsData = await jobService.getSynthesisJobs();
      setJobs(jobsData);
      setError('');
    } catch (err) {
      if (err.message === 'No authentication token found') {
        navigate('/login');
      } else {
        setError('Failed to load synthesis jobs. Please try again later.');
        console.error(err);
      }
    } finally {
      setIsLoading(false);
    }
  }, [navigate]);

  useEffect(() => {
    const initializePage = async () => {
      // Check if user is logged in
      const currentUser = authService.getCurrentUser();
      const token = localStorage.getItem('access_token');
      
      if (!currentUser || !token) {
        navigate('/login');
        return;
      }

      try {
        // Try to refresh the token first
        if (!authService.isAuthenticated()) {
          const refreshResult = await authService.refreshToken();
          if (!refreshResult.success) {
            navigate('/login');
            return;
          }
        }
        
        // Load jobs after ensuring authentication
        await loadJobs();
      } catch (err) {
        console.error('Error initializing page:', err);
        navigate('/login');
      }
    };

    initializePage();
  }, [navigate, loadJobs]);

  // Audio playback functions
  const handlePlayPause = async (job) => {
    if (!job.id || job.status !== 'completed') return;

    if (playingJobId === job.id) {
      // Pause current audio
      if (audioRef.current) {
        audioRef.current.pause();
      }
      setPlayingJobId(null);
      setCurrentlyPlaying(null);
    } else {
      try {
        // Stop any currently playing audio
        if (audioRef.current) {
          audioRef.current.pause();
        }

        // Determine if this is a voice clone job or traditional job
        const isVoiceClone = job.voice_model?.type === 'f5_tts' || job.output_path?.includes('voice_clones');
        
        // Fetch audio file
        const audioBlob = await fetchAudioBlob(job.id, isVoiceClone);
        if (audioBlob && audioRef.current) {
          const audioUrl = URL.createObjectURL(audioBlob);
          audioRef.current.src = audioUrl;
          await audioRef.current.play();
          setPlayingJobId(job.id);
          setCurrentlyPlaying(job);
        }
      } catch (error) {
        console.error('Failed to play audio:', error);
        setError('Failed to play audio file');
      }
    }
  };

  const fetchAudioBlob = async (jobId, isVoiceClone) => {
    try {
      const endpoint = isVoiceClone ? `/file/voice-clone/${jobId}` : `/file/synthesis/${jobId}`;
      const token = localStorage.getItem('access_token');
      
      const response = await fetch(`${apiConfig.apiBaseUrl}${endpoint}`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch audio: ${response.status}`);
      }

      return await response.blob();
    } catch (error) {
      console.error('Error fetching audio:', error);
      throw error;
    }
  };

  const handleAudioEnded = () => {
    setPlayingJobId(null);
    setCurrentlyPlaying(null);
  };

  const handleDownload = async (job) => {
    try {
      const isVoiceClone = job.voice_model?.type === 'f5_tts' || job.output_path?.includes('voice_clones');
      const audioBlob = await fetchAudioBlob(job.id, isVoiceClone);
      const audioUrl = URL.createObjectURL(audioBlob);
      
      const link = document.createElement('a');
      link.href = audioUrl;
      link.download = `speech_${job.id}.wav`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Clean up the URL after download
      setTimeout(() => URL.revokeObjectURL(audioUrl), 1000);
    } catch (err) {
      if (err.message === 'No authentication token found') {
        navigate('/login');
      } else {
        console.error('Failed to download synthesis output:', err);
        setError('Failed to download audio file');
      }
    }
  };

  const handleCancel = async (jobId) => {
    try {
      await jobService.cancelSynthesisJob(jobId);
      loadJobs(); // Reload jobs list
    } catch (err) {
      if (err.message === 'No authentication token found') {
        navigate('/login');
      } else {
        console.error('Failed to cancel synthesis job:', err);
      }
    }
  };

  const getStatusColor = (status) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return 'text-green-500';
      case 'processing':
        return 'text-blue-500';
      case 'failed':
        return 'text-red-500';
      case 'pending':
        return 'text-yellow-500';
      default:
        return 'text-gray-500';
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getJobType = (job) => {
    if (job.voice_model?.type === 'f5_tts' || job.output_path?.includes('voice_clones')) {
      return 'Voice Clone';
    }
    return 'Text to Speech';
  };

  const getVoiceName = (job) => {
    if (job.voice_model) {
      return job.voice_model.name || 'Custom Voice';
    }
    return 'Default Voice';
  };

  return (
    <div className="min-h-screen bg-black text-white p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold mb-12">Generated Voices</h1>
        
        {/* Hidden audio element for playback */}
        <audio 
          ref={audioRef} 
          onEnded={handleAudioEnded}
          className="hidden"
        />

        {isLoading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-white"></div>
          </div>
        ) : error ? (
          <div className="text-red-500 text-center bg-red-500/10 py-4 rounded border border-red-500/20">
            {error}
          </div>
        ) : jobs.length === 0 ? (
          <div className="text-center py-12 bg-zinc-900 rounded-lg border border-zinc-800">
            <div className="mb-4">
              <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
            </div>
            <p className="text-gray-400 text-lg">No generated voices found</p>
            <p className="text-gray-500 text-sm mt-2">Start by creating your first voice synthesis or voice clone</p>
          </div>
        ) : (
          <div className="space-y-6">
            {jobs.map((job) => (
              <div
                key={job.id}
                className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 hover:border-white transition-colors"
              >
                <div className="flex justify-between items-start mb-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="px-2 py-1 text-xs bg-zinc-800 text-gray-300 rounded">
                        {getJobType(job)}
                      </span>
                      <span className={`px-3 py-1 rounded-full text-sm ${getStatusColor(job.status)}`}>
                        {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                      </span>
                    </div>
                    <h3 className="text-xl font-semibold mb-2 line-clamp-2 break-words">
                      {job.text_content || "No text content"}
                    </h3>
                    <div className="text-gray-400 text-sm space-y-1">
                      <p>Voice: {getVoiceName(job)}</p>
                      <p>Created: {formatDate(job.created_at)}</p>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-4 mb-4 text-sm">
                  <div>
                    <p className="text-gray-400">Duration</p>
                    <p className="font-medium">{job.duration ? `${job.duration.toFixed(1)}s` : 'N/A'}</p>
                  </div>
                  <div>
                    <p className="text-gray-400">Format</p>
                    <p className="font-medium">{job.output_format || 'WAV'}</p>
                  </div>
                  <div>
                    <p className="text-gray-400">Sample Rate</p>
                    <p className="font-medium">{job.sample_rate ? `${job.sample_rate} Hz` : 'N/A'}</p>
                  </div>
                </div>

                {/* Progress bar for processing jobs */}
                {job.status === 'processing' && (
                  <div className="mb-4">
                    <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-white transition-all duration-500"
                        style={{ width: `${(job.progress || 0) * 100}%` }}
                      />
                    </div>
                  </div>
                )}

                <div className="flex justify-between items-center">
                  <div className="flex items-center space-x-4">
                    {job.status === 'completed' && (
                      <div className="flex items-center space-x-3">
                        <button
                          onClick={() => handlePlayPause(job)}
                          className="flex items-center space-x-2 px-4 py-2 text-sm border border-white rounded hover:bg-white hover:text-black transition-colors"
                        >
                          {playingJobId === job.id ? (
                            <>
                              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 002 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
                              </svg>
                              <span>Pause</span>
                            </>
                          ) : (
                            <>
                              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                              </svg>
                              <span>Play</span>
                            </>
                          )}
                        </button>
                        <button
                          onClick={() => handleDownload(job)}
                          className="flex items-center space-x-2 px-4 py-2 text-sm border border-zinc-600 text-gray-300 rounded hover:border-white hover:text-white transition-colors"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                          <span>Download</span>
                        </button>
                      </div>
                    )}
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    {job.status === 'processing' && (
                      <button
                        onClick={() => handleCancel(job.id)}
                        className="px-4 py-2 text-sm border border-red-500 text-red-500 rounded hover:bg-red-500 hover:text-white transition-colors"
                      >
                        Cancel
                      </button>
                    )}
                  </div>
                </div>

                {job.error_message && (
                  <div className="mt-4 p-3 bg-red-500/10 text-red-500 rounded text-sm">
                    Error: {job.error_message}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Tasks; 