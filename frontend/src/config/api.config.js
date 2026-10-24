// API Configuration File
const config = {
  // API base URL from environment variables
  apiBaseUrl:
    process.env.REACT_APP_API_BASE_URL ||
    (process.env.NODE_ENV === 'development'
      ? 'http://localhost:8000/api/v1' // Dev environment default
      : process.env.REACT_APP_PROD_API_URL), // Production environment

  // Development environment detection
  isDevelopment: process.env.NODE_ENV === 'development',

  // Timeout settings - 2 minutes for audio file uploads
  timeout: 120000,

  // Other API related configurations
  maxRetries: 3,
  retryDelay: 1000,
};

// Validate production URL
if (!config.isDevelopment && !config.apiBaseUrl) {
  console.error(
    'Production API URL not configured! Please set REACT_APP_API_BASE_URL or REACT_APP_PROD_API_URL environment variable.'
  );
}

export default config;
