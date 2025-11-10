import api from './api';

class VoiceCloneService {
  async uploadVoiceSample(file, name, onProgress) {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        throw new Error('No authentication token found');
      }

      // Ensure we have a valid name
      const validName = (name || '').trim() || `Sample_${Date.now()}`;

      console.log(
        `[DEBUG] Uploading file: ${file.name}, with sample name: ${validName}`
      );

      const formData = new FormData();
      formData.append('file', file);
      formData.append('name', validName);

      // Debug FormData contents
      console.log('[DEBUG] FormData contents:');
      for (let [key, value] of formData.entries()) {
        console.log(`  ${key}:`, value);
      }

      const response = await api.post('/voice/samples', formData, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
        timeout: 300000, // 5-minute timeout for large audio files
        onUploadProgress: onProgress, // Upload progress callback
      });

      return response.data;
    } catch (error) {
      throw error.response?.data?.error || error;
    }
  }

  async createVoiceClone(cloneData) {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        throw new Error('No authentication token found');
      }

      const response = await api.post('/voice/clones', cloneData, {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      return response.data;
    } catch (error) {
      throw error.response?.data?.error || error;
    }
  }

  async listVoiceClones(page = 1, pageSize = 20) {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        throw new Error('No authentication token found');
      }

      const response = await api.get('/voice/clones', {
        params: { page, page_size: pageSize },
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      return response.data;
    } catch (error) {
      throw error.response?.data?.error || error;
    }
  }

  async getVoiceClone(cloneId) {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        throw new Error('No authentication token found');
      }

      const response = await api.get(`/voice/clones/${cloneId}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      return response.data;
    } catch (error) {
      throw error.response?.data?.error || error;
    }
  }

  async deleteVoiceClone(cloneId) {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        throw new Error('No authentication token found');
      }

      const response = await api.delete(`/voice/clones/${cloneId}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      return response.data;
    } catch (error) {
      throw error.response?.data?.error || error;
    }
  }

  async selectVoiceClone(cloneId) {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        throw new Error('No authentication token found');
      }

      const response = await api.post(
        `/voice/clones/${cloneId}/select`,
        {},
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      return response.data;
    } catch (error) {
      throw error.response?.data?.error || error;
    }
  }

  async synthesizeWithClone(cloneId, text, config = {}) {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        throw new Error('No authentication token found');
      }

      const response = await api.post(
        `/voice/clones/${cloneId}/synthesize`,
        {
          text: text,
          language: config.language || 'zh-CN',
          output_format: config.outputFormat || 'wav',
          sample_rate: config.sampleRate || 22050,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        }
      );

      return response.data;
    } catch (error) {
      throw error.response?.data?.error || error;
    }
  }

  async listVoiceSamples(page = 1, pageSize = 20) {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        throw new Error('No authentication token found');
      }

      const response = await api.get('/voice/samples', {
        params: { page, page_size: pageSize },
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      return response.data;
    } catch (error) {
      throw error.response?.data?.error || error;
    }
  }

  async deleteVoiceSample(sampleId) {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        throw new Error('No authentication token found');
      }

      const response = await api.delete(`/voice/samples/${sampleId}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      return response.data;
    } catch (error) {
      throw error.response?.data?.error || error;
    }
  }

}

const voiceCloneService = new VoiceCloneService();
export default voiceCloneService;
