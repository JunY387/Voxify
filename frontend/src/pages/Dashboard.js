import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import authService from '../services/auth.service';
import jobService from '../services/job.service';
import voiceCloneService from '../services/voiceClone.service';

const Dashboard = () => {
  const [stats, setStats] = useState({
    voiceClones: 0,
    completedTasks: 0,
    processingTasks: 0
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  const user = authService.getCurrentUser();
  const firstName = user?.first_name || 'there';

  useEffect(() => {
    loadDashboardStats();
  }, []);

  const loadDashboardStats = async () => {
    try {
      setIsLoading(true);
      setError('');

      // Load voice clones
      const voiceClonesResponse = await voiceCloneService.listVoiceClones(1, 100);
      const voiceClones = voiceClonesResponse.data?.clones || [];

      // Load synthesis jobs
      const jobsResponse = await jobService.getSynthesisJobs();
      const jobs = jobsResponse || [];

      // Calculate statistics
      const completedJobs = jobs.filter(job => job.status === 'completed');
      const processingJobs = jobs.filter(job => job.status === 'processing' || job.status === 'pending');

      setStats({
        voiceClones: voiceClones.length,
        completedTasks: completedJobs.length,
        processingTasks: processingJobs.length
      });

    } catch (error) {
      console.error('Failed to load dashboard stats:', error);
      setError('Failed to load dashboard statistics');
      // Set default values on error
      setStats({
        voiceClones: 0,
        completedTasks: 0,
        processingTasks: 0
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold mb-4">
          Welcome back, {firstName}!
        </h1>
        <p className="text-gray-400 mb-8">
          Ready to start a new task? Choose an option below.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Link
            to="/voice-clone"
            className="group bg-zinc-900 rounded-lg p-6 hover:bg-zinc-800 transition-colors border border-zinc-800"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Clone Your Voice</h2>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
                className="w-6 h-6 text-white group-hover:translate-x-1 transition-transform"
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
              </svg>
            </div>
            <p className="text-gray-400">
              Create a digital copy of your voice that can speak any text naturally.
            </p>
          </Link>

          <Link
            to="/text-to-speech"
            className="group bg-zinc-900 rounded-lg p-6 hover:bg-zinc-800 transition-colors border border-zinc-800"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Text-to-Speech</h2>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
                className="w-6 h-6 text-white group-hover:translate-x-1 transition-transform"
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
              </svg>
            </div>
            <p className="text-gray-400">
              Convert any text into natural-sounding speech using your voice clones.
            </p>
          </Link>
        </div>

        {/* Quick Stats */}
        <div className="mt-12">
          <h2 className="text-2xl font-semibold mb-6">Your Statistics</h2>
          
          {error && (
            <div className="mb-6 text-red-400 text-center bg-red-500/10 py-3 rounded border border-red-500/20">
              {error}
            </div>
          )}
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-zinc-900 rounded-lg p-6 border border-zinc-800">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-gray-400 text-sm mb-2">Voice Clones</h3>
                  {isLoading ? (
                    <div className="animate-pulse h-8 w-8 bg-zinc-700 rounded"></div>
                  ) : (
                    <p className="text-2xl font-semibold">{stats.voiceClones}</p>
                  )}
                </div>
                <div className="p-3 bg-blue-500/10 rounded-lg">
                  <svg className="w-6 h-6 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                  </svg>
                </div>
              </div>
            </div>
            
            <div className="bg-zinc-900 rounded-lg p-6 border border-zinc-800">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-gray-400 text-sm mb-2">Completed Tasks</h3>
                  {isLoading ? (
                    <div className="animate-pulse h-8 w-12 bg-zinc-700 rounded"></div>
                  ) : (
                    <p className="text-2xl font-semibold">{stats.completedTasks}</p>
                  )}
                </div>
                <div className="p-3 bg-green-500/10 rounded-lg">
                  <svg className="w-6 h-6 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
              </div>
            </div>
            
            <div className="bg-zinc-900 rounded-lg p-6 border border-zinc-800">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-gray-400 text-sm mb-2">Processing</h3>
                  {isLoading ? (
                    <div className="animate-pulse h-8 w-8 bg-zinc-700 rounded"></div>
                  ) : (
                    <p className="text-2xl font-semibold">{stats.processingTasks}</p>
                  )}
                </div>
                <div className="p-3 bg-yellow-500/10 rounded-lg">
                  <svg className="w-6 h-6 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="mt-12">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-semibold">Quick Actions</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Link
              to="/tasks"
              className="flex items-center p-4 bg-zinc-900 rounded-lg border border-zinc-800 hover:border-white transition-colors group"
            >
              <div className="p-2 bg-zinc-800 rounded-lg mr-4">
                <svg className="w-5 h-5 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="font-medium">View All Tasks</h3>
                <p className="text-sm text-gray-400">Browse and manage your generated voices</p>
              </div>
              <svg className="w-5 h-5 text-gray-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </Link>
            
            <Link
              to="/settings"
              className="flex items-center p-4 bg-zinc-900 rounded-lg border border-zinc-800 hover:border-white transition-colors group"
            >
              <div className="p-2 bg-zinc-800 rounded-lg mr-4">
                <svg className="w-5 h-5 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="font-medium">Settings</h3>
                <p className="text-sm text-gray-400">Manage your account and preferences</p>
              </div>
              <svg className="w-5 h-5 text-gray-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 