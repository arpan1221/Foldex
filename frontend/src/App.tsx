import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './components/auth/AuthContext';
import MainLayout from './components/layout/MainLayout';
import ErrorBoundary from './components/common/ErrorBoundary';
import ProtectedRoute from './components/common/ProtectedRoute';
import GoogleAuth from './components/auth/GoogleAuth';
import FolderInput from './components/folder/FolderInput';
import ChatInterface from './components/chat/ChatInterface';

/**
 * App Component
 * 
 * Main application component with routing, authentication, and layout.
 * Includes error boundaries and protected routes.
 */
const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <AuthProvider>
        <Router>
          <Routes>
            {/* Public Routes */}
            <Route path="/" element={<GoogleAuth />} />

            {/* Protected Routes */}
            <Route
              path="/folder"
              element={
                <ProtectedRoute>
                  <MainLayout>
                    <FolderInput />
                  </MainLayout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/chat/:folderId"
              element={
                <ProtectedRoute>
                  <MainLayout>
                    <ChatInterface />
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

