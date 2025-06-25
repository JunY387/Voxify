import axios from 'axios';
import authService from './auth.service';
import apiConfig from '../config/api.config';

const api = axios.create({
  baseURL: apiConfig.apiBaseUrl,
  timeout: apiConfig.timeout,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor: Add token to request headers
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    // For FormData, remove Content-Type to let browser set it with boundary
    if (config.data instanceof FormData) {
      delete config.headers['Content-Type'];
    }
    
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor: Handle token expiration
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    // If error is 401 and not a refresh token request
    if (error.response?.status === 401 && !originalRequest._retry && !originalRequest.url.includes('/auth/refresh')) {
      originalRequest._retry = true;

      try {
        // Try to refresh token
        const refreshResult = await authService.refreshToken();
        if (!refreshResult.success) {
          throw new Error('Token refresh failed');
        }

        // Get new token and retry original request
        const newToken = localStorage.getItem('access_token');
        originalRequest.headers.Authorization = `Bearer ${newToken}`;
        return api(originalRequest);
      } catch (refreshError) {
        // If refresh fails, clear tokens and redirect to login
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        localStorage.removeItem('user');
        window.location.href = '/login';
        return Promise.reject(refreshError);
      }
    }

    return Promise.reject(error);
  }
);

// Helper function to create audio URL for playback
export const createAudioUrl = (jobId, isVoiceClone = false) => {
  const token = localStorage.getItem('access_token');
  const endpoint = isVoiceClone ? `/file/voice-clone/${jobId}` : `/file/synthesis/${jobId}`;
  return `${api.defaults.baseURL}${endpoint}?token=${token}`;
};

export default api; 