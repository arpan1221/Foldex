import React, { useEffect, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import { useChat } from '../../hooks/useChat';
import { useFolderProcessor } from '../../hooks/useFolderProcessor';
import { folderService } from '../../services/api';
import MessageBubble from './MessageBubble';
import InputArea from './InputArea';
import ErrorDisplay from '../common/ErrorDisplay';
import LoadingOverlay from '../common/LoadingOverlay';
import ProcessingStatus from '../folder/ProcessingStatus';
import AIAssistantIcon from '../common/AIAssistantIcon';

/**
 * ChatInterface Component
 * 
 * Main chat container with message history, following Figma wireframe design.
 * Includes scrollable message area, typing indicators, and responsive layout.
 */
const ChatInterface: React.FC = () => {
  const { folderId: paramFolderId, conversationId: paramConversationId } = useParams<{ folderId: string, conversationId: string }>();
  const folderId = paramFolderId || '';
  const conversationId = paramConversationId || null;
  const { messages, sendMessage, isLoading, error } = useChat(folderId, conversationId);
  const { status: processingStatus, isProcessing, error: processingError } = useFolderProcessor();
  const [folderExistsInSidebar, setFolderExistsInSidebar] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  
  // Check if folder exists in sidebar - poll until it appears
  useEffect(() => {
    if (!folderId) {
      setFolderExistsInSidebar(false);
      return;
    }
    
    const checkFolderExists = async () => {
      try {
        const folders = await folderService.getUserFolders();
        const exists = folders.some(f => f.folder_id === folderId);
        setFolderExistsInSidebar(exists);
      } catch (err) {
        console.error('Failed to check folder existence:', err);
        setFolderExistsInSidebar(false);
      }
    };
    
    // Check immediately
    checkFolderExists();
    
    // Poll every 2 seconds until folder appears (only if processing)
    const interval = setInterval(() => {
      if (isProcessing || processingStatus) {
        checkFolderExists();
      } else {
        // If not processing, check once more and stop
        checkFolderExists();
        clearInterval(interval);
      }
    }, 2000);
    
    return () => clearInterval(interval);
  }, [folderId, isProcessing, processingStatus]);

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
    <div className="h-full flex flex-col">
      {/* Chat Messages Area */}
      <div
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto p-4 sm:p-6"
        style={{
          scrollBehavior: 'smooth',
        }}
      >
        <div className="max-w-3xl mx-auto space-y-6">
          {/* Processing Status - Show until folder appears in sidebar */}
          {(isProcessing || (processingStatus && !folderExistsInSidebar)) && processingStatus && (
            <div className="animate-fade-in">
              <ProcessingStatus status={processingStatus} error={processingError} />
            </div>
          )}
          
          {/* Welcome Message (if no messages and not processing) */}
          {messages.length === 0 && !isProcessing && !processingStatus && (
            <div className="flex gap-4 animate-fade-in">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gray-600 flex items-center justify-center shadow-sm">
                <AIAssistantIcon className="text-gray-200" size="md" />
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
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gray-600 flex items-center justify-center shadow-sm">
                <AIAssistantIcon className="text-gray-200" size="md" />
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
            <div className="flex gap-4 animate-fade-in max-w-3xl mx-auto">
              <ErrorDisplay
                error={error}
                title="Message Error"
                onRetry={() => {
                  // Retry last message if needed
                  const lastUserMessage = messages
                    .filter((m) => m.role === 'user')
                    .pop();
                  if (lastUserMessage) {
                    // Could implement retry logic here
                  }
                }}
                onDismiss={() => {
                  // Error dismissal handled by hook
                }}
              />
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

      {/* Loading Overlay */}
      {isLoading && messages.length === 0 && (
        <LoadingOverlay
          isLoading={isLoading}
          message="Loading conversation..."
          fullScreen={false}
        />
      )}
    </div>
  );
};

export default ChatInterface;
