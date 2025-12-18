import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';
import AIAssistantIcon from '../common/AIAssistantIcon';

/**
 * AnimatedDemo Component
 * Shows an animated workflow of folder processing and chat
 */
const AnimatedDemo: React.FC = () => {
  const [step, setStep] = useState(0);
  const [streamedText, setStreamedText] = useState('');
  const [currentMessage, setCurrentMessage] = useState(0);
  
  const conversation = [
    {
      question: "What's the main topic of this folder?",
      answer: "This folder contains Q3 marketing materials including campaign briefs, analytics reports, and social media assets for the upcoming product launch.",
      citations: [
        { file: "Q3_Marketing_Strategy.pdf", page: "p. 2-3" },
        { file: "Campaign_Performance.pdf", page: "p. 1" }
      ]
    },
    {
      question: "What's our total budget for digital campaigns?",
      answer: "The total Q3 marketing budget is $450,000, with 35% ($157,500) allocated to digital campaigns and 25% ($112,500) to content creation.",
      citations: [
        { file: "Budget_Analysis.xlsx", page: "Sheet 1, Row 12" }
      ]
    }
  ];
  
  const currentConvo = conversation[currentMessage];
  
  useEffect(() => {
    // Don't auto-advance step 2, let the streaming logic handle it
    if (step === 2) return;
    
    const timer = setTimeout(() => {
      if (step < 4) {
        setStep(step + 1);
      }
    }, step === 0 ? 600 : step === 1 ? 1200 : step === 3 ? 800 : 0);
    
    return () => clearTimeout(timer);
  }, [step]);
  
  useEffect(() => {
    if (step === 2 && streamedText.length < currentConvo.answer.length) {
      const timer = setTimeout(() => {
        const nextLength = Math.min(streamedText.length + 1, currentConvo.answer.length);
        setStreamedText(currentConvo.answer.slice(0, nextLength));
      }, 25);
      return () => clearTimeout(timer);
    } else if (step === 2 && streamedText.length >= currentConvo.answer.length) {
      const timer = setTimeout(() => setStep(3), 600);
      return () => clearTimeout(timer);
    }
  }, [step, streamedText, currentConvo.answer]);
  
  useEffect(() => {
    if (step === 4) {
      const timer = setTimeout(() => {
        setStep(0);
        setStreamedText('');
        setCurrentMessage((currentMessage + 1) % conversation.length);
      }, 2500);
      return () => clearTimeout(timer);
    }
  }, [step, currentMessage, conversation.length]);

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg overflow-hidden h-[520px] flex shadow-xl">
      {/* Folder Sidebar */}
      <div 
        className="w-72 bg-gray-900/90 border-r border-gray-700 p-4 transition-all duration-500"
        style={{ 
          opacity: step >= 0 ? 1 : 0,
          transform: step >= 0 ? 'translateX(0)' : 'translateX(-20px)'
        }}
      >
        <div className="flex items-center gap-2 mb-4">
          <div className="w-8 h-8 bg-gray-700 rounded-lg flex items-center justify-center">
            <svg
              className="w-5 h-5 text-gray-300"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M3.75 9.776c.112-.017.227-.026.344-.026h15.812c.117 0 .232.009.344.026m-16.5 0a2.25 2.25 0 00-1.883 2.542l.857 6a2.25 2.25 0 002.227 1.932H19.05a2.25 2.25 0 002.227-1.932l.857-6a2.25 2.25 0 00-1.883-2.542m-16.5 0V6A2.25 2.25 0 016 3.75h3.879a1.5 1.5 0 011.06.44l2.122 2.12a1.5 1.5 0 001.06.44H18A2.25 2.25 0 0120.25 9v.776"
              />
            </svg>
          </div>
          <div>
            <div className="text-sm font-medium text-gray-100">Marketing Folder</div>
            <div className="text-xs text-gray-500">5 files</div>
          </div>
        </div>
        
        <div className="space-y-1.5">
          {[
            { name: 'Q3_Marketing_Strategy.pdf', size: '2.4 MB', type: 'PDF' },
            { name: 'Budget_Analysis.xlsx', size: '1.1 MB', type: 'Excel' },
            { name: 'Campaign_Performance.pdf', size: '3.2 MB', type: 'PDF' },
            { name: 'Meeting_Notes.docx', size: '245 KB', type: 'Word' },
            { name: 'Brand_Guidelines.pdf', size: '5.8 MB', type: 'PDF' },
          ].map((file, idx) => (
            <div
              key={idx}
              className="group flex items-start gap-2.5 px-3 py-2.5 rounded-lg hover:bg-gray-800/70 transition-all cursor-pointer"
              style={{
                opacity: step >= 0 ? 1 : 0,
                transform: step >= 0 ? 'translateX(0)' : 'translateX(-10px)',
                transitionDelay: `${idx * 80}ms`,
                backgroundColor: step >= 0 ? 'rgba(31, 41, 55, 0.4)' : 'transparent'
              }}
            >
              <div className="flex-shrink-0 w-8 h-8 bg-gray-700/70 rounded flex items-center justify-center mt-0.5">
                <svg
                  className="w-4 h-4 text-gray-400"
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
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-xs font-medium text-gray-200 truncate group-hover:text-white transition-colors">
                  {file.name}
                </div>
                <div className="text-xs text-gray-500 mt-0.5">
                  {file.size} • {file.type}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col bg-gray-900/60">
        {/* Messages */}
        <div className="flex-1 p-5 space-y-4 overflow-hidden">
          {/* User Message */}
          {step >= 1 && (
            <div 
              className="flex gap-3 animate-slide-up"
            >
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center text-xs font-medium text-gray-300 shadow-sm">
                You
              </div>
              <div className="flex-1 max-w-md">
                <div className="bg-gray-700/60 rounded-2xl px-4 py-3 border border-gray-600/50 shadow-sm">
                  <p className="text-sm text-gray-100">{currentConvo.question}</p>
                </div>
              </div>
            </div>
          )}

          {/* AI Response with Streaming */}
          {step >= 2 && (
            <div className="flex gap-3 animate-slide-up">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center shadow-sm">
                <AIAssistantIcon className="text-gray-200" size="sm" />
              </div>
              <div className="flex-1 max-w-lg">
                <div className="bg-gray-800/70 rounded-2xl px-4 py-3 border border-gray-600/40 shadow-sm">
                  <p className="text-sm text-gray-100 leading-relaxed">
                    {streamedText}
                    {streamedText.length < currentConvo.answer.length && (
                      <span className="inline-block w-1.5 h-4 bg-gray-400 ml-1 animate-pulse align-middle" />
                    )}
                  </p>
                  {step >= 3 && (
                    <div className="space-y-1.5 mt-3 pt-2.5 border-t border-gray-700/50 animate-fade-in">
                      <span className="text-xs text-gray-500 font-medium">Sources:</span>
                      <div className="flex flex-wrap gap-2">
                        {currentConvo.citations.map((citation, idx) => (
                          <div key={idx} className="flex items-center gap-1.5 px-2.5 py-1.5 bg-gray-700/60 rounded-md border border-gray-600/40 hover:bg-gray-700 transition-colors cursor-pointer">
                            <svg
                              className="w-3 h-3 text-gray-400 flex-shrink-0"
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
                            <div className="flex flex-col">
                              <span className="text-xs text-gray-300 leading-tight">{citation.file}</span>
                              <span className="text-[10px] text-gray-500 leading-tight">{citation.page}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-700 p-4 bg-gray-900/70">
          <div className="flex gap-3">
            <div className="flex-1 h-11 bg-gray-800/80 border border-gray-700/60 rounded-xl px-4 flex items-center shadow-sm">
              <span className="text-sm text-gray-500">Ask anything about your folder...</span>
            </div>
            <button className="h-11 w-11 bg-gray-700 hover:bg-gray-600 rounded-xl flex items-center justify-center transition-colors shadow-sm">
              <svg
                className="w-5 h-5 text-gray-300"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * GoogleAuth Component
 * 
 * Landing page with Google OAuth authentication button.
 * Follows Figma wireframe design with dark gradient background
 * and modern card-based layout.
 */
const GoogleAuth: React.FC = () => {
  const { isAuthenticated } = useAuth();
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
      const googleAuthUrl = getGoogleAuthUrl();
      
      // Redirect to Google OAuth for authentication
        window.location.href = googleAuthUrl;
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
    const env = (import.meta as any).env || {};
    const clientId = env.VITE_GOOGLE_CLIENT_ID || '';
    const redirectUri = encodeURIComponent(
      env.VITE_GOOGLE_REDIRECT_URI || 
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
                className="w-16 h-16 text-gray-300"
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
                className="w-7 h-7 text-gray-400 absolute -top-1 -right-1"
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
          
          <h1 className="text-4xl sm:text-5xl font-bold text-white">
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
              border border-gray-300 
              shadow-md hover:shadow-lg
              px-6 py-3 
              rounded-full
              gap-3
              font-medium
              text-base
              transition-all duration-200
              disabled:opacity-50 disabled:cursor-not-allowed
              disabled:hover:bg-white
              flex items-center justify-center
              w-auto
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

        {/* Animated Demo */}
        <div className="max-w-3xl mx-auto mt-12">
          <div className="text-center mb-4">
            <h3 className="font-semibold text-gray-100 flex items-center justify-center gap-2">
              <svg
                className="w-5 h-5 text-gray-300"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              See it in action
            </h3>
          </div>
          <AnimatedDemo />
        </div>

        {/* Key Features */}
        <div className="max-w-3xl mx-auto mt-6">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-5 text-center">
              <div className="w-12 h-12 bg-gray-700/50 rounded-lg flex items-center justify-center mx-auto mb-3">
                <svg
                  className="w-6 h-6 text-gray-300"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                  />
                </svg>
              </div>
              <h3 className="font-semibold text-gray-100 mb-1.5 text-sm">Privacy First</h3>
              <p className="text-xs text-gray-400 leading-relaxed">
                Your data stays in your Google Drive. We never store or share your files.
            </p>
          </div>

            <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-5 text-center">
              <div className="w-12 h-12 bg-gray-700/50 rounded-lg flex items-center justify-center mx-auto mb-3">
              <svg
                  className="w-6 h-6 text-gray-300"
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
              </div>
              <h3 className="font-semibold text-gray-100 mb-1.5 text-sm">Lightning Fast</h3>
              <p className="text-xs text-gray-400 leading-relaxed">
                Get instant answers with real-time streaming responses and smart caching.
              </p>
            </div>

            <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-5 text-center">
              <div className="w-12 h-12 bg-gray-700/50 rounded-lg flex items-center justify-center mx-auto mb-3">
                <svg
                  className="w-6 h-6 text-gray-300"
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
              </div>
              <h3 className="font-semibold text-gray-100 mb-1.5 text-sm">Source Citations</h3>
              <p className="text-xs text-gray-400 leading-relaxed">
                Every answer includes references to exact files and page numbers.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GoogleAuth;
