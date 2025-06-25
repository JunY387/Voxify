import api from './api';

class AuthService {
  async login(email, password) {
    try {
      const response = await api.post('/auth/login', {
        email,
        password
      });

      if (response.data.success) {
        const { access_token, refresh_token, user } = response.data.data;
        localStorage.setItem('access_token', access_token);
        localStorage.setItem('refresh_token', refresh_token);
        localStorage.setItem('user', JSON.stringify(user));
        return { success: true, user };
      }

      return { 
        success: false, 
        error: response.data.error?.message || 'Login failed' 
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.error?.message || 'Login failed'
      };
    }
  }

  async register(userData) {
    try {
      const response = await api.post('/auth/register', {
        email: userData.email,
        password: userData.password,
        first_name: userData.firstName,
        last_name: userData.lastName
      });

      if (response.data.success) {
        return { 
          success: true, 
          user: response.data.data.user 
        };
      }

      return {
        success: false,
        error: response.data.error?.message || 'Registration failed'
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.error?.message || 'Registration failed'
      };
    }
  }

  logout() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
    window.location.href = '/login';
  }

  getCurrentUser() {
    const userStr = localStorage.getItem('user');
    if (!userStr) return null;
    try {
      return JSON.parse(userStr);
    } catch {
      return null;
    }
  }

  isAuthenticated() {
    return !!localStorage.getItem('access_token');
  }

  async refreshToken() {
    try {
      const refreshToken = localStorage.getItem('refresh_token');
      if (!refreshToken) {
        throw new Error('No refresh token available');
      }

      const response = await api.post('/auth/refresh', {}, {
        headers: {
          'Authorization': `Bearer ${refreshToken}`
        }
      });

      if (response.data.success) {
        const { access_token, refresh_token } = response.data.data;
        localStorage.setItem('access_token', access_token);
        localStorage.setItem('refresh_token', refresh_token);
        return { success: true };
      }

      return {
        success: false,
        error: response.data.error?.message || 'Token refresh failed'
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.error?.message || 'Token refresh failed'
      };
    }
  }

  async getUserProfile() {
    try {
      const response = await api.get('/auth/profile');
      if (response.data.success && response.data.data && response.data.data.user) {
        return {
          data: response.data.data.user
        };
      }
      throw new Error('Invalid response format');
    } catch (error) {
      throw error;
    }
  }

  async updateUserProfile(userData) {
    try {
      const response = await api.put('/auth/profile', userData);
      if (response.data.success && response.data.data) {
        return {
          data: response.data.data.user || response.data.data,
          message: response.data.message
        };
      }
      throw new Error('Invalid response format');
    } catch (error) {
      throw error;
    }
  }
}

const authService = new AuthService();
export default authService; 