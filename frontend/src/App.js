import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import VoiceClone from './pages/VoiceClone';
import TextToSpeech from './pages/TextToSpeech';
import Tasks from './pages/Tasks';
import Settings from './pages/Settings';
import authService from './services/auth.service';

// Protected Route Component
const ProtectedRoute = ({ children }) => {
  const isAuthenticated = authService.isAuthenticated();
  
  if (!isAuthenticated) {
    return <Navigate to="/login" />;
  }

  return children;
};

// Public Route Component (redirects to dashboard if already logged in)
const PublicRoute = ({ children }) => {
  const isAuthenticated = authService.isAuthenticated();
  
  if (isAuthenticated) {
    return <Navigate to="/dashboard" />;
  }

  return children;
};

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-[#0F1419]">
        <Navbar />
        <Routes>
          {/* Public Routes */}
          <Route path="/login" element={
            <PublicRoute>
              <Login />
            </PublicRoute>
          } />
          <Route path="/signup" element={
            <PublicRoute>
              <Register />
            </PublicRoute>
          } />
          <Route path="/register" element={
            <PublicRoute>
              <Register />
            </PublicRoute>
          } />

          {/* Protected Routes */}
          <Route path="/dashboard" element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          } />
          <Route path="/voice-clone" element={
            <ProtectedRoute>
              <VoiceClone />
            </ProtectedRoute>
          } />
          <Route path="/text-to-speech" element={
            <ProtectedRoute>
              <TextToSpeech />
            </ProtectedRoute>
          } />
          <Route path="/tasks" element={
            <ProtectedRoute>
              <Tasks />
            </ProtectedRoute>
          } />
          <Route path="/tasks/voice-clone" element={
            <ProtectedRoute>
              <VoiceClone />
            </ProtectedRoute>
          } />
          <Route path="/tasks/text-to-speech" element={
            <ProtectedRoute>
              <TextToSpeech />
            </ProtectedRoute>
          } />
          <Route path="/tasks/list" element={
            <ProtectedRoute>
              <Tasks />
            </ProtectedRoute>
          } />
          <Route path="/voices" element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          } />
          <Route path="/settings" element={
            <ProtectedRoute>
              <Settings />
            </ProtectedRoute>
          } />

          {/* Redirect root to dashboard or login */}
          <Route path="/" element={
            <Navigate to={authService.isAuthenticated() ? "/dashboard" : "/login"} replace />
          } />
        </Routes>
      </div>
    </Router>
  );
}

export default App; 