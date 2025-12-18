import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './components/auth/AuthContext';
import GoogleAuth from './components/auth/GoogleAuth';
import FolderInput from './components/folder/FolderInput';
import ChatInterface from './components/chat/ChatInterface';

const App: React.FC = () => {
  return (
    <AuthProvider>
      <Router>
        <div className="min-h-screen bg-gray-50">
          <Routes>
            <Route path="/" element={<GoogleAuth />} />
            <Route path="/folder" element={<FolderInput />} />
            <Route path="/chat/:folderId" element={<ChatInterface />} />
          </Routes>
        </div>
      </Router>
    </AuthProvider>
  );
};

export default App;

