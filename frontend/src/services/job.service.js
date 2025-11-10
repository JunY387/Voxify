import api from './api';

class JobService {
  async getSynthesisJobs() {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        throw new Error('No authentication token found');
      }

      const response = await api.get('/job', {
        params: {
          sort_by: 'created_at',
          sort_order: 'desc',
          limit: 100,
        },
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      return response.data.data;
    } catch (error) {
      if (error.response?.status === 404) {
        // If no jobs found, return empty array
        return [];
      }
      throw error.response?.data?.error || error;
    }
  }

  async getSynthesisJob(jobId) {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        throw new Error('No authentication token found');
      }

      const response = await api.get(`/job/${jobId}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      return response.data.data;
    } catch (error) {
      throw error.response?.data?.error || error;
    }
  }

  async createSynthesisJob(voiceModelId, text, config = {}) {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        throw new Error('No authentication token found');
      }

      const response = await api.post(
        '/job',
        {
          voice_model_id: voiceModelId,
          text_content: text,
          output_format: config.outputFormat || 'wav',
          sample_rate: config.sampleRate || 22050,
          speed: config.speed || 1.0,
          pitch: config.pitch || 1.0,
          volume: config.volume || 1.0,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      return response.data.data;
    } catch (error) {
      throw error.response?.data?.error || error;
    }
  }

  async cancelSynthesisJob(jobId) {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        throw new Error('No authentication token found');
      }

      const response = await api.post(
        `/job/${jobId}/cancel`,
        {},
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      return response.data.success;
    } catch (error) {
      throw error.response?.data?.error || error;
    }
  }

  async downloadSynthesisOutput(jobId) {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        throw new Error('No authentication token found');
      }

      const response = await api.get(`/job/${jobId}/download`, {
        responseType: 'blob',
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      return response.data;
    } catch (error) {
      throw error.response?.data?.error || error;
    }
  }

  async deleteJob(jobId) {
    try {
      const token = localStorage.getItem('access_token');
      if (!token) {
        throw new Error('No authentication token found');
      }

      await api.delete(`/job/${jobId}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      // DELETE endpoint returns 204 No Content on success
      return { success: true };
    } catch (error) {
      if (error.response?.status === 204) {
        return { success: true };
      }
      throw error.response?.data?.error || error;
    }
  }
}

const jobService = new JobService();
export default jobService;