import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';
import { authService } from '../../services/api';

/**
 * GoogleAuth Component
 * 
 * Landing page with Google OAuth authentication button.
 * Follows Figma wireframe design with dark gradient background
 * and modern card-based layout.
 */
const GoogleAuth: React.FC = () => {
  const { isAuthenticated, login } = useAuth();
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/folder');
    }
  }, [isAuthenticated, navigate]);

  const handleGoogleLogin = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Initialize Google OAuth
      // In production, this would use Google Identity Services
      const googleAuthUrl = getGoogleAuthUrl();
      
      // For development, we'll use a mock flow
      // In production, redirect to Google OAuth
      if (import.meta.env.DEV) {
        // Mock authentication for development
        const mockToken = 'mock_google_token_' + Date.now();
        const response = await authService.exchangeToken(mockToken);
        
        await login(response.access_token);
        navigate('/folder');
      } else {
        // Production: Redirect to Google OAuth
        window.location.href = googleAuthUrl;
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Authentication failed';
      setError(errorMessage);
      console.error('Google authentication error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const getGoogleAuthUrl = (): string => {
    // Build Google OAuth URL
    const clientId = import.meta.env.VITE_GOOGLE_CLIENT_ID || '';
    const redirectUri = encodeURIComponent(
      import.meta.env.VITE_GOOGLE_REDIRECT_URI || 
      `${window.location.origin}/auth/callback`
    );
    const scope = encodeURIComponent(
      'https://www.googleapis.com/auth/drive.readonly profile email'
    );
    
    return `https://accounts.google.com/o/oauth2/v2/auth?` +
      `client_id=${clientId}&` +
      `redirect_uri=${redirectUri}&` +
      `response_type=code&` +
      `scope=${scope}&` +
      `access_type=offline&` +
      `prompt=consent`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950 flex items-center justify-center p-4 sm:p-6">
      <div className="max-w-4xl w-full space-y-8 animate-fade-in">
        {/* Header Section */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center gap-3 mb-6">
            <div className="relative">
              <svg
                className="w-20 h-20 text-foldex-primary-500"
                fill="none"
                stroke="currentColor"
                strokeWidth={1.5}
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3.75 9.776c.112-.017.227-.026.344-.026h15.812c.117 0 .232.009.344.026m-16.5 0a2.25 2.25 0 00-1.883 2.542l.857 6a2.25 2.25 0 002.227 1.932H19.05a2.25 2.25 0 002.227-1.932l.857-6a2.25 2.25 0 00-1.883-2.542m-16.5 0V6A2.25 2.25 0 016 3.75h3.879a1.5 1.5 0 011.06.44l2.122 2.12a1.5 1.5 0 001.06.44H18A2.25 2.25 0 0120.25 9v.776"
                />
              </svg>
              <svg
                className="w-8 h-8 text-foldex-accent-400 absolute -top-2 -right-2"
                fill="none"
                stroke="currentColor"
                strokeWidth={2}
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z"
                />
              </svg>
            </div>
          </div>
          
          <h1 className="text-4xl sm:text-5xl font-bold bg-gradient-to-r from-foldex-primary-400 to-foldex-accent-400 bg-clip-text text-transparent">
            Foldex
          </h1>
          
          <p className="text-lg sm:text-xl text-gray-300 max-w-2xl mx-auto">
            Your AI assistant to talk to any Google Drive folder. Ask questions, find files, and get insights instantly.
          </p>
        </div>

        {/* Error Display */}
        {error && (
          <div className="max-w-md mx-auto bg-red-950/50 border border-red-800/50 rounded-lg p-4 text-red-200 text-sm">
            <div className="flex items-center gap-2">
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span>{error}</span>
            </div>
          </div>
        )}

        {/* Auth Button */}
        <div className="flex justify-center">
          <button
            onClick={handleGoogleLogin}
            disabled={isLoading}
            className={`
              group relative
              bg-white hover:bg-gray-50 
              text-gray-900 
              border-2 border-gray-600 
              shadow-lg hover:shadow-xl
              px-8 py-6 
              rounded-lg
              gap-3
              font-medium
              text-base sm:text-lg
              transition-all duration-200
              disabled:opacity-50 disabled:cursor-not-allowed
              disabled:hover:bg-white
              flex items-center justify-center
              min-w-[280px] sm:min-w-[320px]
            `}
          >
            {isLoading ? (
              <>
                <svg
                  className="animate-spin h-6 w-6 text-gray-900"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3 2.647z"
                  />
                </svg>
                <span>Connecting...</span>
              </>
            ) : (
              <>
                <svg
                  viewBox="0 0 24 24"
                  className="w-6 h-6 flex-shrink-0"
                  aria-hidden="true"
                >
                  <path
                    fill="#4285F4"
                    d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                  />
                  <path
                    fill="#34A853"
                    d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                  />
                  <path
                    fill="#FBBC05"
                    d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                  />
                  <path
                    fill="#EA4335"
                    d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                  />
                </svg>
                <span>Sign in with Google</span>
              </>
            )}
          </button>
        </div>

        {/* Features/Info Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-2xl mx-auto mt-12">
          <div className="bg-gray-800/50 backdrop-blur-sm border-2 border-gray-700 rounded-lg p-4">
            <h3 className="font-semibold text-gray-100 mb-2 flex items-center gap-2">
              <svg
                className="w-5 h-5 text-foldex-primary-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              Works with all files
            </h3>
            <p className="text-sm text-gray-400">
              PDFs, documents, spreadsheets, and more
            </p>
          </div>

          <div className="bg-gray-800/50 backdrop-blur-sm border-2 border-gray-700 rounded-lg p-4">
            <h3 className="font-semibold text-gray-100 mb-2 flex items-center gap-2">
              <svg
                className="w-5 h-5 text-foldex-accent-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 10V3L4 14h7v7l9-11h-7z"
                />
              </svg>
              Instant answers
            </h3>
            <p className="text-sm text-gray-400">
              Get insights from your files in seconds
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GoogleAuth;
