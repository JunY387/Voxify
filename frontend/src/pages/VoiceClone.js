import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import voiceCloneService from '../services/voiceClone.service';
import LoadingSpinner from '../components/LoadingSpinner';
import TabSelector from '../components/TabSelector';
import VoiceRecorder from '../components/VoiceRecorder';
import { getRandomScript, isRecordingSupported } from '../config/recordingScripts';

const VoiceClone = () => {
  const navigate = useNavigate();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [refText, setRefText] = useState('');
  const [language, setLanguage] = useState('zh-CN');
  const [files, setFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  
  // Tab and recording related states
  const [activeTab, setActiveTab] = useState('upload');
  const [randomScript, setRandomScript] = useState('');
  const [recordedAudioFile, setRecordedAudioFile] = useState(null);

  // Generate new random script when language changes
  useEffect(() => {
    if (isRecordingSupported(language)) {
      const newScript = getRandomScript(language);
      setRandomScript(newScript);
      
      // Auto-update reference text if in record mode
      if (activeTab === 'record') {
        setRefText(newScript);
      }
    }
  }, [language, activeTab]);

  // Tab switching handler
  const handleTabChange = (tabId) => {
    setActiveTab(tabId);
    // Switch to English if current language doesn't support recording when switching to record tab
    if (tabId === 'record' && !isRecordingSupported(language)) {
      setLanguage('en-US');
    }
    
    // Auto-fill reference text with random script when switching to record mode
    if (tabId === 'record' && randomScript) {
      setRefText(randomScript);
    } else if (tabId === 'upload') {
      // Clear reference text when switching to upload mode
      setRefText('');
    }
  };

  // Recording completion handler
  const handleRecordingComplete = (audioFile) => {
    setRecordedAudioFile(audioFile);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files);
    setFiles(droppedFiles);
  };

  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(selectedFiles);
  };

  const handleGenerateClone = async (e) => {
    e.preventDefault();
    
    // Only proceed if this is a real form submission from the submit button
    if (e.type !== 'submit' || (e.nativeEvent?.submitter && e.nativeEvent.submitter.type !== 'submit')) {
      return;
    }
    
    // Custom validation to avoid browser default messages
    if (!name.trim()) {
      alert('Please enter a name for your voice clone');
      document.getElementById('name').focus();
      return;
    }

    if (!refText.trim()) {
      alert('Please enter reference text (must match audio content exactly)');
      document.getElementById('refText').focus();
      return;
    }

    // Check audio files (upload mode) or recorded file (record mode)
    if (activeTab === 'upload' && files.length === 0) {
      alert('Please select at least one audio file');
      return;
    }
    
    if (activeTab === 'record' && !recordedAudioFile) {
      alert('Please record your voice first');
      return;
    }

    setIsCreating(true);
    setIsUploading(true);
    try {
      const uploadedIds = [];
      
      // Determine files to process
      const filesToProcess = activeTab === 'upload' ? files : [recordedAudioFile];

      for (let i = 0; i < filesToProcess.length; i++) {
        const file = filesToProcess[i];
        try {
          const sampleName = file.name?.trim() || `Voice Sample ${Date.now()}`;
          const uploadResult = await voiceCloneService.uploadVoiceSample(
            file,
            sampleName
          );
          if (uploadResult.success) {
            uploadedIds.push(uploadResult.data.sample_id);
          } else {
            alert(`Failed to upload ${file.name}: ${uploadResult.error}`);
            return;
          }
        } catch (error) {
          console.error(`Failed to upload ${file.name}:`, error);
          alert(`Failed to upload ${file.name}: ${error.message || error}`);
          return;
        }
      }

      const result = await voiceCloneService.createVoiceClone({
        sample_ids: uploadedIds,
        name: name.trim(),
        description: description.trim(),
        ref_text: refText.trim(),
        language: language,
        clone_type: activeTab, // 'upload' or 'record'
      });

      if (result.success) {
        alert('Voice clone created successfully! 🎉');
        navigate('/dashboard');
      } else {
        alert(`Creation failed: ${result.error}`);
      }
    } catch (error) {
      console.error('Create clone failed:', error);
      alert(`Creation failed: ${error.message || error}`);
    } finally {
      setIsCreating(false);
      setIsUploading(false);
      // Clean up file states
      if (activeTab === 'upload') {
        setFiles([]);
      } else {
        setRecordedAudioFile(null);
      }
    }
  };

  return (
    <div className="min-h-screen bg-black text-white p-8">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-4xl font-bold mb-12">Clone your voice</h1>

        {/* Voice Input Section - Outside of form */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold mb-4">Voice Input</h2>
          
          {/* Tab Selector */}
          <TabSelector
            activeTab={activeTab}
            onTabChange={handleTabChange}
            tabs={[
              { id: 'upload', label: 'Upload voice sample' },
              { 
                id: 'record', 
                label: 'Record your voice',
                disabled: !isRecordingSupported(language) 
              }
            ]}
          />
        </div>

        <form onSubmit={handleGenerateClone} className="space-y-8">
          {/* Name Input */}
          <div>
            <label
              htmlFor="name"
              className="block text-sm font-medium text-gray-300 mb-2"
            >
              Clone Name *
            </label>
            <input
              id="name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full rounded border border-zinc-800 bg-zinc-900 px-4 py-2 text-white placeholder-gray-400 focus:border-white focus:outline-none focus:ring-1 focus:ring-white transition-colors"
              placeholder="My voice clone"
            />
          </div>

          {/* Description Input */}
          <div>
            <label
              htmlFor="description"
              className="block text-sm font-medium text-gray-300 mb-2"
            >
              Description
            </label>
            <textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full rounded border border-zinc-800 bg-zinc-900 px-4 py-2 text-white placeholder-gray-400 focus:border-white focus:outline-none focus:ring-1 focus:ring-white transition-colors h-24 resize-none"
              placeholder="This is my personal voice clone"
            />
          </div>

          {/* Reference Text Input */}
          <div>
            <label
              htmlFor="refText"
              className="block text-sm font-medium text-gray-300 mb-2"
            >
              Reference Text *
            </label>
            <div className="mb-2">
              <p className="text-sm text-gray-400">
                Enter the exact text spoken in your audio file, including
                punctuation
              </p>
            </div>
            <textarea
              id="refText"
              value={refText}
              onChange={(e) => setRefText(e.target.value)}
              className={`w-full rounded border border-zinc-800 px-4 py-2 text-white placeholder-gray-400 focus:border-white focus:outline-none focus:ring-1 focus:ring-white transition-colors h-32 resize-none ${
                activeTab === 'record' ? 'bg-zinc-800 cursor-not-allowed' : 'bg-zinc-900'
              }`}
              placeholder="e.g., Hello everyone, I'm John. Today is a beautiful day, perfect for taking a walk outside."
              readOnly={activeTab === 'record'}
            />
          </div>

          {/* Language Selection */}
          <div>
            <label
              htmlFor="language"
              className="block text-sm font-medium text-gray-300 mb-2"
            >
              Language
            </label>
            <select
              id="language"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="w-full rounded border border-zinc-800 bg-zinc-900 px-4 py-2 text-white focus:border-white focus:outline-none focus:ring-1 focus:ring-white transition-colors"
            >
              {/* Native multilingual support */}
              <optgroup label="Native Support (Best Quality)">
                <option value="zh-CN">Chinese (Simplified)</option>
                <option value="zh-TW">Chinese (Traditional)</option>
                <option value="en-US">English (US)</option>
                <option value="en-GB">English (UK)</option>
              </optgroup>

              {/* Specialized model support */}
              <optgroup label="Specialized Models (High Quality)">
                <option value="ja-JP">Japanese</option>
                <option value="fr-FR">French</option>
                <option value="de-DE">German</option>
                <option value="es-ES">Spanish</option>
                <option value="it-IT">Italian</option>
                <option value="ru-RU">Russian</option>
                <option value="hi-IN">Hindi</option>
                <option value="fi-FI">Finnish</option>
              </optgroup>

              {/* Fallback support */}
              <optgroup label="Basic Support (Limited Quality)">
                <option value="ko-KR">Korean</option>
                <option value="pt-BR">Portuguese</option>
                <option value="ar-SA">Arabic</option>
                <option value="th-TH">Thai</option>
                <option value="vi-VN">Vietnamese</option>
              </optgroup>
            </select>
          </div>

          {/* Voice Input Content */}
          <div>

            {/* Upload Tab Content */}
            {activeTab === 'upload' && (
              <div>
                <p className="text-gray-400 mb-4">
                  Upload 3-30 seconds of high-quality audio for F5-TTS. Clear speech
                  with minimal background noise works best. One good sample is
                  sufficient.
                </p>

            {/* Drop Zone */}
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              className="border-2 border-dashed border-zinc-800 rounded-lg p-12 text-center hover:border-white transition-colors cursor-pointer"
            >
              <div className="space-y-4">
                <p className="text-lg font-medium">Drag and drop files here</p>
                <p className="text-gray-400">Or click to browse</p>
                <input
                  type="file"
                  multiple
                  accept="audio/*"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="file-upload"
                  aria-label="Upload voice samples"
                />
                <button
                  type="button"
                  onClick={() => document.getElementById('file-upload').click()}
                  className="px-4 py-2 bg-zinc-900 border border-zinc-800 rounded hover:bg-zinc-800 transition-colors"
                >
                  Browse Files
                </button>
              </div>
            </div>

            {/* File List */}
            {files.length > 0 && (
              <div className="mt-4 space-y-2">
                {files.map((file, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between bg-zinc-900 p-3 rounded"
                  >
                    <span className="text-gray-300">{file.name}</span>
                    <span className="text-gray-400">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </span>
                  </div>
                ))}
                {/* Removed standalone upload button; upload occurs during generation */}
              </div>
            )}

            {/* Removed uploaded samples list because generation is a single-step flow */}
              </div>
            )}

            {/* Record Tab Content */}
            {activeTab === 'record' && (
              <div>
                {!isRecordingSupported(language) ? (
                  <div className="bg-yellow-900/20 border border-yellow-800 rounded-lg p-4 text-center">
                    <p className="text-yellow-300 mb-2">
                      Recording is not supported for the selected language.
                    </p>
                    <p className="text-yellow-400 text-sm">
                      Please switch to Chinese (Simplified) or English to use the recording feature.
                    </p>
                  </div>
                ) : (
                  <VoiceRecorder
                    randomScript={randomScript}
                    onRecordingComplete={handleRecordingComplete}
                    language={language}
                  />
                )}
              </div>
            )}

            {/* Loading Spinner */}
            {(isUploading || isCreating) && (
              <LoadingSpinner message="Generating... This might take a while" />
            )}
          </div>

          {/* Submit Button */}
          <div>
            <button
              type="submit"
              disabled={
                (activeTab === 'upload' && files.length === 0) ||
                (activeTab === 'record' && !recordedAudioFile) ||
                isCreating || isUploading ||
                !name.trim() ||
                !refText.trim()
              }
              className={`w-full px-6 py-3 rounded text-center font-semibold border-2 border-white hover:bg-white hover:text-black transition-colors ${
                (activeTab === 'upload' && files.length === 0) ||
                (activeTab === 'record' && !recordedAudioFile) ||
                isCreating || isUploading ||
                !name.trim() ||
                !refText.trim()
                  ? 'opacity-50 cursor-not-allowed'
                  : ''
              }`}
            >
              {isCreating || isUploading ? 'Generating...' : 'Generate Voice Clone'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default VoiceClone;
