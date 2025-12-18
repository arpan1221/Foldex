import React, { useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { useChat } from '../../hooks/useChat';
import MessageBubble from './MessageBubble';
import CitationDisplay from './CitationDisplay';
import InputArea from './InputArea';
import Header from '../layout/Header';

/**
 * ChatInterface Component
 * 
 * Main chat container with message history, following Figma wireframe design.
 * Includes scrollable message area, typing indicators, and responsive layout.
 */
const ChatInterface: React.FC = () => {
  const { folderId: paramFolderId } = useParams<{ folderId: string }>();
  const folderId = paramFolderId || '';
  const { messages, sendMessage, isLoading, error } = useChat(folderId);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Focus input on '/' key
      if (e.key === '/' && e.target === document.body) {
        e.preventDefault();
        const input = document.querySelector('textarea[placeholder*="Ask anything"]') as HTMLTextAreaElement;
        input?.focus();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleSendMessage = async (content: string) => {
    if (!content.trim() || isLoading) return;
    await sendMessage(content);
  };

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      {/* Header */}
      <Header />

      {/* Chat Messages Area */}
      <div
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto p-4 sm:p-6"
        style={{
          scrollBehavior: 'smooth',
        }}
      >
        <div className="max-w-3xl mx-auto space-y-6">
          {/* Welcome Message (if no messages) */}
          {messages.length === 0 && (
            <div className="flex gap-4 animate-fade-in">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-foldex-primary-500 to-foldex-accent-500 flex items-center justify-center shadow-lg">
                <svg
                  className="w-5 h-5 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z"
                  />
                </svg>
              </div>
              <div className="flex-1 space-y-2">
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
                  <p className="text-gray-200">
                    👋 Hello! I've analyzed your folder and found files. Ask me anything about the contents!
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Messages */}
          {messages.map((message) => (
            <div key={message.message_id} className="flex gap-4 animate-fade-in">
              <MessageBubble message={message} />
            </div>
          ))}

          {/* Typing Indicator */}
          {isLoading && (
            <div className="flex gap-4 animate-fade-in">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-foldex-primary-500 to-foldex-accent-500 flex items-center justify-center shadow-lg">
                <svg
                  className="w-5 h-5 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z"
                  />
                </svg>
              </div>
              <div className="flex-1 space-y-2">
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="flex gap-4 animate-fade-in">
              <div className="flex-1">
                <div className="bg-red-950/30 border border-red-800/50 rounded-lg p-4">
                  <div className="flex items-center gap-2">
                    <svg
                      className="w-5 h-5 text-red-400"
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
                    <p className="text-sm text-red-300">
                      {error.message || 'Failed to send message. Please try again.'}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Scroll anchor */}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <InputArea
        onSendMessage={handleSendMessage}
        isLoading={isLoading}
        disabled={!folderId}
      />
    </div>
  );
};

export default ChatInterface;
