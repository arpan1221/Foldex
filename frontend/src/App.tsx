import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './components/auth/AuthContext';
import MainLayout from './components/layout/MainLayout';
import ErrorBoundary from './components/common/ErrorBoundary';
import ProtectedRoute from './components/common/ProtectedRoute';
import GoogleAuth from './components/auth/GoogleAuth';
import OAuthCallback from './components/auth/OAuthCallback';
import FolderInput from './components/folder/FolderInput';
import ChatInterface from './components/chat/ChatInterface';
import { systemService } from './services/api';

/**
 * App Component
 * 
 * Main application component with routing, authentication, and layout.
 * Includes error boundaries and protected routes.
 */
const App: React.FC = () => {
  // Check API health on mount
  React.useEffect(() => {
    const checkHealth = async () => {
      try {
        await systemService.healthCheck();
      } catch (error) {
        console.warn('API health check failed:', error);
        // Don't block app, just log warning
      }
    };
    checkHealth();
  }, []);

  return (
    <ErrorBoundary>
      <AuthProvider>
        <Router future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
          <Routes>
            {/* Public Routes */}
            <Route path="/" element={<GoogleAuth />} />
            <Route path="/auth/callback" element={<OAuthCallback />} />

            {/* Protected Routes */}
            <Route
              path="/folder"
              element={
                <ProtectedRoute>
                  <MainLayout>
                    <ErrorBoundary>
                      <FolderInput />
                    </ErrorBoundary>
                  </MainLayout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/chat/:folderId/:conversationId?"
              element={
                <ProtectedRoute>
                  <MainLayout>
                    <ErrorBoundary>
                      <ChatInterface />
                    </ErrorBoundary>
                  </MainLayout>
                </ProtectedRoute>
              }
            />

            {/* Catch-all route */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Router>
      </AuthProvider>
    </ErrorBoundary>
  );
};

export default App;

