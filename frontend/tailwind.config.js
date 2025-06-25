/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'primary': {
          DEFAULT: '#FFFFFF',
          dark: '#E5E5E5'
        },
        'dark': {
          DEFAULT: '#0A0A0A',
          paper: '#121212',
          lighter: '#1A1A1A'
        }
      },
      fontFamily: {
        'sans': ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
    },
  },
  plugins: [],
} 