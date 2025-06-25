import React, { useState, useEffect, useRef } from 'react';
import jobService from '../services/job.service';
import voiceCloneService from '../services/voiceClone.service';
import { createAudioUrl } from '../services/api';
import apiConfig from '../config/api.config';

const TextToSpeech = () => {
  const [text, setText] = useState('');
  const [voice, setVoice] = useState('');
  const [voiceType, setVoiceType] = useState('clone'); // 'clone' or 'system'
  const [voiceClones, setVoiceClones] = useState([]);
  const [loadingClones, setLoadingClones] = useState(true);
  const [config, setConfig] = useState({
    speed: 1.0,
    pitch: 1.0,
    volume: 1.0,
    outputFormat: 'wav',
    sampleRate: 22050,
    language: 'en-US'
  });
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [generatedAudio, setGeneratedAudio] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef(null);

  // Load voice clones on component mount
  useEffect(() => {
    const loadVoiceClones = async () => {
      try {
        setLoadingClones(true);
        const response = await voiceCloneService.listVoiceClones();
        if (response.success) {
          setVoiceClones(response.data.clones || []);
        }
      } catch (error) {
        console.error('Failed to load voice clones:', error);
        setVoiceClones([]);
      } finally {
        setLoadingClones(false);
      }
    };

    loadVoiceClones();
  }, []);

  // Audio control functions
  const handlePlayPause = async () => {
    if (!generatedAudio?.jobId) return;

    if (isPlaying) {
      if (audioRef.current) {
        audioRef.current.pause();
      }
      setIsPlaying(false);
    } else {
      try {
        // If audio source is not set, fetch it first
        if (!audioRef.current?.src || audioRef.current.src === '') {
          const audioBlob = await fetchAudioBlob(generatedAudio.jobId, generatedAudio.voiceType === 'clone');
          if (audioBlob && audioRef.current) {
            const audioUrl = URL.createObjectURL(audioBlob);
            audioRef.current.src = audioUrl;
          }
        }
        
        if (audioRef.current) {
          await audioRef.current.play();
          setIsPlaying(true);
        }
      } catch (error) {
        console.error('Failed to play audio:', error);
        alert('Unable to play audio file');
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
    setIsPlaying(false);
  };

  const handleAudioError = (e) => {
    console.error('Audio playback error:', e);
    setIsPlaying(false);
    alert('Unable to play audio file');
  };

  const handleDownload = async () => {
    if (!generatedAudio?.jobId) return;
    
    try {
      const audioBlob = await fetchAudioBlob(generatedAudio.jobId, generatedAudio.voiceType === 'clone');
      const audioUrl = URL.createObjectURL(audioBlob);
      
      const link = document.createElement('a');
      link.href = audioUrl;
      link.download = `speech_${generatedAudio.jobId}.wav`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Clean up the URL after download
      setTimeout(() => URL.revokeObjectURL(audioUrl), 1000);
    } catch (error) {
      console.error('Failed to download audio:', error);
      alert('Unable to download audio file');
    }
  };

  const handleNewGeneration = () => {
    setGeneratedAudio(null);
    setIsPlaying(false);
    setText('');
    setVoice('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim() || !voice) return;

    setIsGenerating(true);
    
    try {
      // Simulate progress while actually generating
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 500);

      let result;
      if (voiceType === 'clone') {
        // Use voice clone synthesis
        result = await voiceCloneService.synthesizeWithClone(voice, text, {
          language: config.language,
          outputFormat: config.outputFormat,
          sampleRate: config.sampleRate
        });
      } else {
        // Use traditional synthesis job
        result = await jobService.createSynthesisJob(voice, text, config);
      }
      
      setProgress(100);
      
      if (result.success) {
        // Store generated audio information
        setGeneratedAudio({
          text: text,
          audioPath: result.data?.output_path,
          jobId: result.data?.job_id,
          language: config.language,
          voiceType: voiceType,
          voiceName: voiceType === 'clone' 
            ? voiceClones.find(clone => clone.clone_id === voice)?.name 
            : voice,
          createdAt: new Date().toISOString()
        });
        
        // Don't reset form immediately, let user play the audio first
        // setText('');
        // setVoice('');
      }
    } catch (error) {
      console.error('Failed to generate speech:', error);
      alert(`Failed to generate speech: ${error.message || error}`);
    } finally {
      setIsGenerating(false);
      setProgress(0);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white p-8">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-4xl font-bold mb-12">Text to Speech</h1>

        <form onSubmit={handleSubmit} className="space-y-8">
          {/* Text Input */}
          <div>
            <label htmlFor="text" className="block text-sm font-medium text-gray-300 mb-2">
              Text
            </label>
            <textarea
              id="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="w-full rounded border border-zinc-800 bg-zinc-900 px-4 py-2 text-white placeholder-gray-400 focus:border-white focus:outline-none focus:ring-1 focus:ring-white transition-colors h-48 resize-none"
              placeholder="Enter the text you want to convert to speech"
              disabled={isGenerating}
            />
          </div>

          {/* Voice Type Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Voice Type
            </label>
            <div className="flex space-x-4 mb-4">
              <label className="flex items-center">
                <input
                  type="radio"
                  value="clone"
                  checked={voiceType === 'clone'}
                  onChange={(e) => {
                    setVoiceType(e.target.value);
                    setVoice(''); // Reset voice selection when type changes
                  }}
                  className="mr-2"
                  disabled={isGenerating}
                />
                <span className="text-white">Voice Clones</span>
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  value="system"
                  checked={voiceType === 'system'}
                  onChange={(e) => {
                    setVoiceType(e.target.value);
                    setVoice(''); // Reset voice selection when type changes
                  }}
                  className="mr-2"
                  disabled={isGenerating}
                />
                <span className="text-white">System Voices</span>
              </label>
            </div>
          </div>

          {/* Voice Selection */}
          <div>
            <label htmlFor="voice" className="block text-sm font-medium text-gray-300 mb-2">
              {voiceType === 'clone' ? 'Select Voice Clone' : 'Select System Voice'}
            </label>
            {voiceType === 'clone' ? (
              <select
                id="voice"
                value={voice}
                onChange={(e) => setVoice(e.target.value)}
                className="w-full rounded border border-zinc-800 bg-zinc-900 px-4 py-2 text-white placeholder-gray-400 focus:border-white focus:outline-none focus:ring-1 focus:ring-white transition-colors"
                disabled={isGenerating || loadingClones}
              >
                <option value="">
                  {loadingClones ? 'Loading voice clones...' : 'Select a voice clone'}
                </option>
                {voiceClones.map((clone) => (
                  <option key={clone.clone_id} value={clone.clone_id}>
                    {clone.name} ({clone.language}) - {clone.status}
                  </option>
                ))}
                {!loadingClones && voiceClones.length === 0 && (
                  <option value="" disabled>
                    No voice clones available. Create one in Voice Clone page.
                  </option>
                )}
              </select>
            ) : (
              <select
                id="voice"
                value={voice}
                onChange={(e) => setVoice(e.target.value)}
                className="w-full rounded border border-zinc-800 bg-zinc-900 px-4 py-2 text-white placeholder-gray-400 focus:border-white focus:outline-none focus:ring-1 focus:ring-white transition-colors"
                disabled={isGenerating}
              >
                <option value="">Select a system voice</option>
                <option value="voice1">Voice 1</option>
                <option value="voice2">Voice 2</option>
              </select>
            )}
          </div>

          {/* Selected Voice Clone Info */}
          {voiceType === 'clone' && voice && (
            <div className="bg-zinc-900 border border-zinc-800 rounded p-4">
              <h3 className="text-lg font-semibold mb-2">Selected Voice Clone</h3>
              {(() => {
                const selectedClone = voiceClones.find(clone => clone.clone_id === voice);
                if (selectedClone) {
                  return (
                    <div className="space-y-2 text-sm text-gray-300">
                      <div><strong>Name:</strong> {selectedClone.name}</div>
                      <div><strong>Language:</strong> {selectedClone.language}</div>
                      <div><strong>Status:</strong> <span className={`${selectedClone.status === 'ready' ? 'text-green-400' : 'text-yellow-400'}`}>{selectedClone.status}</span></div>
                      <div><strong>Created:</strong> {new Date(selectedClone.created_at).toLocaleDateString()}</div>
                    </div>
                  );
                }
                return null;
              })()}
            </div>
          )}

          {/* Settings Section */}
          <div>
            <h2 className="text-xl font-semibold mb-4">Settings</h2>
            <div className="space-y-4">
              {/* Language Selection */}
              <div>
                <label htmlFor="language" className="block text-sm font-medium text-gray-300 mb-2">
                  Language
                </label>
                <select
                  id="language"
                  value={config.language}
                  onChange={(e) => setConfig(prev => ({ ...prev, language: e.target.value }))}
                  className="w-full rounded border border-zinc-800 bg-zinc-900 px-4 py-2 text-white placeholder-gray-400 focus:border-white focus:outline-none focus:ring-1 focus:ring-white transition-colors"
                  disabled={isGenerating}
                >
                  <option value="en-US">English (US)</option>
                  <option value="en-GB">English (UK)</option>
                  <option value="zh-CN">Chinese (Simplified)</option>
                </select>
              </div>

              {/* Speed Control */}
              <div>
                <label htmlFor="speed" className="block text-sm font-medium text-gray-300 mb-2">
                  Speed
                </label>
                <select
                  id="speed"
                  value={config.speed}
                  onChange={(e) => setConfig(prev => ({ ...prev, speed: parseFloat(e.target.value) }))}
                  className="w-full rounded border border-zinc-800 bg-zinc-900 px-4 py-2 text-white placeholder-gray-400 focus:border-white focus:outline-none focus:ring-1 focus:ring-white transition-colors"
                  disabled={isGenerating}
                >
                  <option value="0.75">Slow</option>
                  <option value="1.0">Normal</option>
                  <option value="1.25">Fast</option>
                </select>
              </div>

              {/* Pitch Control */}
              <div>
                <label htmlFor="pitch" className="block text-sm font-medium text-gray-300 mb-2">
                  Pitch
                </label>
                <select
                  id="pitch"
                  value={config.pitch}
                  onChange={(e) => setConfig(prev => ({ ...prev, pitch: parseFloat(e.target.value) }))}
                  className="w-full rounded border border-zinc-800 bg-zinc-900 px-4 py-2 text-white placeholder-gray-400 focus:border-white focus:outline-none focus:ring-1 focus:ring-white transition-colors"
                  disabled={isGenerating}
                >
                  <option value="0.75">Low</option>
                  <option value="1.0">Normal</option>
                  <option value="1.25">High</option>
                </select>
              </div>

              {/* Volume Control */}
              <div>
                <label htmlFor="volume" className="block text-sm font-medium text-gray-300 mb-2">
                  Volume
                </label>
                <select
                  id="volume"
                  value={config.volume}
                  onChange={(e) => setConfig(prev => ({ ...prev, volume: parseFloat(e.target.value) }))}
                  className="w-full rounded border border-zinc-800 bg-zinc-900 px-4 py-2 text-white placeholder-gray-400 focus:border-white focus:outline-none focus:ring-1 focus:ring-white transition-colors"
                  disabled={isGenerating}
                >
                  <option value="0.75">Quiet</option>
                  <option value="1.0">Normal</option>
                  <option value="1.25">Loud</option>
                </select>
              </div>
            </div>
          </div>

          {/* Generation Progress */}
          {isGenerating && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Generating</span>
                <span>{progress}% complete</span>
              </div>
              <div className="h-2 bg-zinc-900 rounded-full overflow-hidden">
                <div
                  className="h-full bg-white transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {/* Submit Button */}
          <div>
            <button
              type="submit"
              disabled={!text.trim() || !voice || isGenerating}
              className={`w-full px-6 py-3 rounded text-center font-semibold border-2 border-white hover:bg-white hover:text-black transition-colors ${
                (!text.trim() || !voice || isGenerating) ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              {isGenerating 
                ? 'Generating...' 
                : voiceType === 'clone' 
                  ? 'Generate with Voice Clone' 
                  : 'Generate speech'
              }
            </button>
          </div>
        </form>

        {/* Generated Audio Player */}
        {generatedAudio && (
          <div className="mt-8 bg-zinc-900 border border-zinc-800 rounded-lg p-6">
            <h2 className="text-2xl font-semibold mb-4">Generated Audio</h2>
            
            {/* Audio Information */}
            <div className="mb-4 space-y-2 text-sm text-gray-300">
              <div><strong>Text:</strong> {generatedAudio.text.length > 100 ? `${generatedAudio.text.substring(0, 100)}...` : generatedAudio.text}</div>
              <div><strong>Voice Type:</strong> {generatedAudio.voiceType === 'clone' ? 'Voice Clone' : 'System Voice'}</div>
              <div><strong>Voice Name:</strong> {generatedAudio.voiceName}</div>
              <div><strong>Language:</strong> {generatedAudio.language}</div>
              <div><strong>Generated At:</strong> {new Date(generatedAudio.createdAt).toLocaleString('en-US')}</div>
            </div>

            {/* Audio Player Controls */}
            <div className="flex items-center space-x-4 mb-4">
              <button
                onClick={handlePlayPause}
                className="flex items-center space-x-2 px-4 py-2 bg-white text-black rounded hover:bg-gray-200 transition-colors"
              >
                <span>{isPlaying ? '⏸️' : '▶️'}</span>
                <span>{isPlaying ? 'Pause' : 'Play'}</span>
              </button>
              
              <button
                onClick={handleDownload}
                className="flex items-center space-x-2 px-4 py-2 border border-white text-white rounded hover:bg-white hover:text-black transition-colors"
              >
                <span>⬇️</span>
                <span>Download</span>
              </button>
              
              <button
                onClick={handleNewGeneration}
                className="flex items-center space-x-2 px-4 py-2 border border-zinc-600 text-gray-300 rounded hover:bg-zinc-800 transition-colors"
              >
                <span>🔄</span>
                <span>Generate New</span>
              </button>
            </div>

            {/* Hidden Audio Element */}
            <audio
              ref={audioRef}
              onEnded={handleAudioEnded}
              onError={handleAudioError}
              preload="none"
              className="hidden"
            />

            {/* Audio Waveform Visualization (placeholder) */}
            <div className="h-16 bg-zinc-800 rounded flex items-center justify-center text-gray-400">
              <span>🎵 Audio Waveform (Playing: {isPlaying ? 'Yes' : 'No'})</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TextToSpeech; 