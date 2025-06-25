import React, { useState, useEffect } from 'react';
import authService from '../services/auth.service';

const Settings = () => {
  const [profile, setProfile] = useState({
    email: '',
    first_name: '',
    last_name: '',
    email_verified: false,
    last_login_at: '',
  });
  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState('');

  useEffect(() => {
    loadUserProfile();
  }, []);

  const loadUserProfile = async () => {
    try {
      setLoading(true);
      const response = await authService.getUserProfile();
      if (response.data) {
        setProfile(response.data);
        setError(null);
      }
    } catch (err) {
      setError('Failed to load profile data');
      console.error('Error loading profile:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setProfile(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      await authService.updateUserProfile({
        first_name: profile.first_name,
        last_name: profile.last_name,
      });
      setSuccessMessage('Profile updated successfully');
      setIsEditing(false);
      setError(null);
    } catch (err) {
      setError('Failed to update profile');
      console.error('Error updating profile:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-black text-white p-8">
        <div className="flex justify-center items-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-white p-8">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Settings</h1>
        
        {error && (
          <div className="bg-red-900 text-white p-4 rounded-lg mb-6">
            {error}
          </div>
        )}
        
        {successMessage && (
          <div className="bg-zinc-800 text-white p-4 rounded-lg mb-6">
            {successMessage}
          </div>
        )}

        <div className="bg-zinc-900 rounded-lg p-6 border border-zinc-800">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold">Profile Information</h2>
            <button
              onClick={() => setIsEditing(!isEditing)}
              className="px-4 py-2 border border-white rounded-lg hover:bg-white hover:text-black transition-colors"
            >
              {isEditing ? 'Cancel' : 'Edit'}
            </button>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="space-y-6">
              <div>
                <label className="block text-gray-400 mb-2">Email</label>
                <input
                  type="email"
                  value={profile.email}
                  disabled
                  className="w-full bg-zinc-900 border border-zinc-800 rounded-lg p-3 text-white"
                />
              </div>

              <div>
                <label className="block text-gray-400 mb-2">First Name</label>
                <input
                  type="text"
                  name="first_name"
                  value={profile.first_name}
                  onChange={handleInputChange}
                  disabled={!isEditing}
                  className="w-full bg-zinc-900 border border-zinc-800 rounded-lg p-3 text-white"
                />
              </div>

              <div>
                <label className="block text-gray-400 mb-2">Last Name</label>
                <input
                  type="text"
                  name="last_name"
                  value={profile.last_name}
                  onChange={handleInputChange}
                  disabled={!isEditing}
                  className="w-full bg-zinc-900 border border-zinc-800 rounded-lg p-3 text-white"
                />
              </div>

              <div>
                <label className="block text-gray-400 mb-2">Email Verification Status</label>
                <div className="flex items-center space-x-2">
                  <span className={`inline-block w-2 h-2 rounded-full ${profile.email_verified ? 'bg-green-500' : 'bg-red-500'}`}></span>
                  <span>{profile.email_verified ? 'Verified' : 'Not Verified'}</span>
                </div>
              </div>

              <div>
                <label className="block text-gray-400 mb-2">Last Login</label>
                <div className="text-white">
                  {profile.last_login_at ? new Date(profile.last_login_at).toLocaleString() : 'Never'}
                </div>
              </div>

              {isEditing && (
                <div className="flex justify-end mt-6">
                  <button
                    type="submit"
                    disabled={loading}
                    className="px-6 py-2 border border-white rounded-lg hover:bg-white hover:text-black transition-colors"
                  >
                    Save Changes
                  </button>
                </div>
              )}
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Settings; 