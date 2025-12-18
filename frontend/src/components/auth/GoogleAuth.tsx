import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from './AuthContext';

const GoogleAuth: React.FC = () => {
  const { isAuthenticated, login } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/folder');
    }
  }, [isAuthenticated, navigate]);

  const handleGoogleLogin = async () => {
    // TODO: Implement Google OAuth flow
    await login('mock_token');
    navigate('/folder');
  };

  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="bg-white p-8 rounded-lg shadow-md">
        <h1 className="text-2xl font-bold mb-4">Welcome to Foldex</h1>
        <p className="text-gray-600 mb-6">
          Connect your Google Drive to get started
        </p>
        <button
          onClick={handleGoogleLogin}
          className="w-full bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600"
        >
          Sign in with Google
        </button>
      </div>
    </div>
  );
};

export default GoogleAuth;

