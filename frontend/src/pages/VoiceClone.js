import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import voiceCloneService from '../services/voiceClone.service';

const VoiceClone = () => {
  const navigate = useNavigate();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [refText, setRefText] = useState('');
  const [language, setLanguage] = useState('zh-CN');
  const [files, setFiles] = useState([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedSamples, setUploadedSamples] = useState([]);
  const [isCreating, setIsCreating] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files);
    setFiles(droppedFiles);
  };

  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(selectedFiles);
  };

  const handleFileUpload = async () => {
    if (files.length === 0) return;
    
    setIsUploading(true);
    setUploadProgress(0);
    
    try {
      const uploadedIds = [];
      
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        try {
          // Ensure we have a valid sample name
          const sampleName = file.name?.trim() || `Voice Sample ${Date.now()}`;
          const result = await voiceCloneService.uploadVoiceSample(file, sampleName);
          if (result.success) {
            uploadedIds.push(result.data.sample_id);
            setUploadedSamples(prev => [...prev, result.data]);
          } else {
            console.error(`Failed to upload ${file.name}:`, result.error);
            alert(`Failed to upload ${file.name}: ${result.error}`);
          }
        } catch (error) {
          console.error(`Failed to upload ${file.name}:`, error);
          alert(`Failed to upload ${file.name}: ${error.message || error}`);
        }
        
        setUploadProgress(((i + 1) / files.length) * 100);
      }
      
      setFiles([]);
      if (uploadedIds.length > 0) {
        alert(`Successfully uploaded ${uploadedIds.length} file(s)`);
      }
    } catch (error) {
      console.error('Upload failed:', error);
      alert('Upload failed, please try again');
    } finally {
      setIsUploading(false);
    }
  };

  const handleCreateClone = async (e) => {
    e.preventDefault();
    
    if (uploadedSamples.length === 0) {
      alert('Please upload voice samples first');
      return;
    }
    
    if (!refText.trim()) {
      alert('Please enter reference text (must match audio content exactly)');
      return;
    }
    
    if (!name.trim()) {
      alert('Please enter a name for your voice clone');
      return;
    }
    
    setIsCreating(true);
    
    try {
      const cloneData = {
        sample_ids: uploadedSamples.map(s => s.sample_id),
        name: name.trim(),
        description: description.trim(),
        ref_text: refText.trim(),
        language: language
      };
      
      const result = await voiceCloneService.createVoiceClone(cloneData);
      
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
    }
  };

  return (
    <div className="min-h-screen bg-black text-white p-8">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-4xl font-bold mb-12">Clone your voice</h1>

        <form onSubmit={handleCreateClone} className="space-y-8">
          {/* Name Input */}
          <div>
            <label htmlFor="name" className="block text-sm font-medium text-gray-300 mb-2">
              Clone Name *
            </label>
            <input
              id="name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full rounded border border-zinc-800 bg-zinc-900 px-4 py-2 text-white placeholder-gray-400 focus:border-white focus:outline-none focus:ring-1 focus:ring-white transition-colors"
              placeholder="My voice clone"
              required
            />
          </div>

          {/* Description Input */}
          <div>
            <label htmlFor="description" className="block text-sm font-medium text-gray-300 mb-2">
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
            <label htmlFor="refText" className="block text-sm font-medium text-gray-300 mb-2">
              Reference Text *
            </label>
            <div className="mb-2">
              <p className="text-sm text-gray-400">
                Enter the exact text spoken in your audio file, including punctuation
              </p>
            </div>
            <textarea
              id="refText"
              value={refText}
              onChange={(e) => setRefText(e.target.value)}
              className="w-full rounded border border-zinc-800 bg-zinc-900 px-4 py-2 text-white placeholder-gray-400 focus:border-white focus:outline-none focus:ring-1 focus:ring-white transition-colors h-32 resize-none"
              placeholder="e.g., Hello everyone, I'm John. Today is a beautiful day, perfect for taking a walk outside."
              required
            />
          </div>

          {/* Language Selection */}
          <div>
            <label htmlFor="language" className="block text-sm font-medium text-gray-300 mb-2">
              Language
            </label>
            <select
              id="language"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="w-full rounded border border-zinc-800 bg-zinc-900 px-4 py-2 text-white focus:border-white focus:outline-none focus:ring-1 focus:ring-white transition-colors"
            >
              <option value="zh-CN">Chinese</option>
              <option value="en-US">English</option>
            </select>
          </div>

          {/* File Upload Section */}
          <div>
            <h2 className="text-xl font-semibold mb-2">Upload voice samples</h2>
            <p className="text-gray-400 mb-4">
              Upload 3-30 seconds of high-quality audio for F5-TTS. Clear speech with minimal background noise works best. One good sample is sufficient.
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
                  <div key={index} className="flex items-center justify-between bg-zinc-900 p-3 rounded">
                    <span className="text-gray-300">{file.name}</span>
                    <span className="text-gray-400">{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                  </div>
                ))}
                <button
                  type="button"
                  onClick={handleFileUpload}
                  disabled={isUploading}
                  className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 text-white rounded transition-colors"
                >
                  {isUploading ? 'Uploading...' : 'Upload Samples'}
                </button>
              </div>
            )}

            {/* Uploaded Samples List */}
            {uploadedSamples.length > 0 && (
              <div className="mt-4">
                <h3 className="text-lg font-medium mb-2">Uploaded Samples</h3>
                <div className="space-y-2">
                  {uploadedSamples.map((sample, index) => (
                    <div key={index} className="flex items-center justify-between bg-green-900/20 border border-green-800 p-3 rounded">
                      <span className="text-gray-300">{sample.name}</span>
                      <span className="text-green-400">Ready</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Upload Progress */}
            {isUploading && (
              <div className="mt-6 space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Uploading</span>
                  <span>{uploadProgress}% complete</span>
                </div>
                <div className="h-2 bg-zinc-900 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-white transition-all duration-500"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Submit Button */}
          <div>
            <button
              type="submit"
              disabled={uploadedSamples.length === 0 || isCreating || !name.trim() || !refText.trim()}
              className={`w-full px-6 py-3 rounded text-center font-semibold border-2 border-white hover:bg-white hover:text-black transition-colors ${
                (uploadedSamples.length === 0 || isCreating || !name.trim() || !refText.trim()) ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              {isCreating ? 'Creating Clone...' : 'Create Voice Clone'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default VoiceClone; 