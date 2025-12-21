import React, { useEffect, useRef, useState } from 'react';
import { useParams, useSearchParams } from 'react-router-dom';
import { useChat } from '../../hooks/useChat';
import { useFolderProcessor } from '../../hooks/useFolderProcessor';
import { folderService, chatService } from '../../services/api';
import MessageBubble from './MessageBubble';
import InputArea from './InputArea';
import ErrorDisplay from '../common/ErrorDisplay';
import LoadingOverlay from '../common/LoadingOverlay';
import ProcessingStatus from '../folder/ProcessingStatus';
import FolderSummaryDisplay from '../folder/FolderSummaryDisplay';
import AIAssistantIcon from '../common/AIAssistantIcon';
import FileBadge from '../common/FileBadge';
import LearningNotification from '../common/LearningNotification';

/**
 * ChatInterface Component
 * 
 * Main chat container with message history, following Figma wireframe design.
 * Includes scrollable message area, typing indicators, and responsive layout.
 */
const ChatInterface: React.FC = () => {
  const { folderId: paramFolderId, conversationId: paramConversationId } = useParams<{ folderId: string, conversationId: string }>();
  const [searchParams] = useSearchParams();
  const folderId = paramFolderId || '';
  const conversationId = paramConversationId || null;
  const fileId = searchParams.get('file_id') || undefined;
  const fileName = searchParams.get('file_name') || undefined;
  const { messages, sendMessage, isLoading, error, statusMessage } = useChat(folderId, conversationId, fileId);
  const { status: processingStatus, isProcessing, error: processingError } = useFolderProcessor();
  const [folderExistsInSidebar, setFolderExistsInSidebar] = useState(false);
  const [folderMetadata, setFolderMetadata] = useState<{ file_count: number } | null>(null);
  const [isInitialChat, setIsInitialChat] = useState(false);
  const [isLoadingMetadata, setIsLoadingMetadata] = useState(true);
  const [isLearning, setIsLearning] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(true);
  const scrollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  
  // Check if user is near bottom of scroll (within 100px)
  const isNearBottom = (): boolean => {
    const container = chatContainerRef.current;
    if (!container) return true;
    const threshold = 100;
    const distanceFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
    return distanceFromBottom < threshold;
  };

  // Smart auto-scroll: only scroll if user is at bottom
  const scrollToBottom = (force: boolean = false) => {
    if (!messagesEndRef.current || !chatContainerRef.current) return;
    
    // Only auto-scroll if user is near bottom or forced
    if (force || shouldAutoScrollRef.current) {
      // Use requestAnimationFrame for smoother scrolling
      requestAnimationFrame(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
      });
    }
  };

  // Handle scroll events to detect user scrolling
  useEffect(() => {
    const container = chatContainerRef.current;
    if (!container) return;

    const handleScroll = () => {
      // Clear any pending scroll timeout
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }

      // Check if user is near bottom
      shouldAutoScrollRef.current = isNearBottom();
      
      // Set a timeout to re-enable auto-scroll if user scrolls back to bottom
      scrollTimeoutRef.current = setTimeout(() => {
        if (isNearBottom()) {
          shouldAutoScrollRef.current = true;
        }
      }, 150);
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    return () => {
      container.removeEventListener('scroll', handleScroll);
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
    };
  }, []);

  // Load folder metadata and check if this is the initial chat
  useEffect(() => {
    if (!folderId) {
      setIsLoadingMetadata(false);
      return;
    }

    const loadMetadata = async () => {
      setIsLoadingMetadata(true);
      try {
        // Load folder metadata to get file count
        const metadata = await folderService.getFolderMetadata(folderId);
        setFolderMetadata({ file_count: metadata.file_count });

        // Check if this is the initial chat (first conversation with title "Initial Chat")
        if (conversationId) {
          const conversations = await chatService.getFolderConversations(folderId);
          const currentConv = conversations.find(c => c.conversation_id === conversationId);
          setIsInitialChat(currentConv?.title === 'Initial Chat');
        } else {
          // If no conversation ID yet, check if there are any conversations
          const conversations = await chatService.getFolderConversations(folderId);
          setIsInitialChat(conversations.length === 0 || conversations[0]?.title === 'Initial Chat');
        }
      } catch (err) {
        console.error('Failed to load folder metadata:', err);
        setFolderMetadata(null);
        setIsInitialChat(false);
      } finally {
        setIsLoadingMetadata(false);
      }
    };

    loadMetadata();
  }, [folderId, conversationId]);
  
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

  // Track learning state from processing status
  useEffect(() => {
    if (processingStatus?.type === 'learning_started' || processingStatus?.type === 'summary_progress') {
      setIsLearning(true);
    } else if (processingStatus?.type === 'summary_complete') {
      setIsLearning(false);
    }
  }, [processingStatus?.type]);

  // Auto-scroll to bottom when new messages arrive (smart scroll)
  useEffect(() => {
    // Always scroll on initial load or when loading starts
    if (messages.length === 0 || isLoading) {
      shouldAutoScrollRef.current = true;
      scrollToBottom(true);
      return;
    }

    // For streaming updates, only scroll if user is at bottom
    const lastMessage = messages[messages.length - 1];
    if (lastMessage && lastMessage.role === 'assistant' && isLoading) {
      // During streaming, check if we should auto-scroll
      scrollToBottom();
    } else {
      // New message added, scroll if at bottom
      scrollToBottom();
    }
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
    await sendMessage(content, true, fileId);
  };

  // Determine if welcome message should be shown (only for initial chat with no messages)
  const shouldShowWelcome = messages.length === 0 && !isProcessing && !processingStatus && !isLoadingMetadata && isInitialChat;

  return (
    <div className="h-full flex flex-col">
      {/* Chat Messages Area */}
      <div
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto p-4 sm:p-6 scroll-smooth"
        style={{
          scrollBehavior: 'smooth',
          // Improve scroll performance
          willChange: 'scroll-position',
        }}
      >
        <div className="max-w-3xl mx-auto space-y-6">
          {/* Processing Status - Show until folder appears in sidebar */}
          {(isProcessing || (processingStatus && !folderExistsInSidebar)) && processingStatus && (
            <div className="animate-fade-in">
              <ProcessingStatus status={processingStatus} error={processingError} />
            </div>
          )}
          
          {/* Welcome Message (only for initial chat with no messages) */}
          {shouldShowWelcome && (
            <div className="flex gap-4 animate-fade-in">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gray-600 flex items-center justify-center shadow-sm">
                <AIAssistantIcon className="text-gray-200" size="md" />
              </div>
              <div className="flex-1 space-y-2">
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
                  <p className="text-gray-200">
                    👋 Hello! I've analyzed your folder and found {folderMetadata?.file_count || 0} {folderMetadata?.file_count === 1 ? 'file' : 'files'}. Ask me anything about the contents!
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Messages */}
          {messages.map((message, index) => {
            // Check if this is the last message and it's streaming
            const isLastMessage = index === messages.length - 1;
            const isStreaming = isLastMessage && isLoading && message.role === 'assistant';
            
            return (
              <div key={message.message_id} className="flex gap-4 animate-fade-in">
                <MessageBubble message={message} isStreaming={isStreaming} />
              </div>
            );
          })}

          {/* Status Message & Typing Indicator */}
          {isLoading && (
            <div className="flex gap-4 animate-fade-in">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gray-600 flex items-center justify-center shadow-sm">
                <AIAssistantIcon className="text-gray-200" size="md" />
              </div>
              <div className="flex-1 space-y-2">
                <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
                  <div className="flex items-center gap-2">
                    <div className="flex gap-1.5">
                      <div className="w-2 h-2 bg-foldex-primary-400 rounded-full animate-pulse"></div>
                      <div className="w-2 h-2 bg-foldex-primary-400 rounded-full animate-pulse" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 bg-foldex-primary-400 rounded-full animate-pulse" style={{ animationDelay: '300ms' }}></div>
                    </div>
                    {statusMessage && (
                      <span className="text-sm text-gray-400 italic">{statusMessage}</span>
                    )}
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

      {/* Folder Summary Display */}
      {folderId && (
        <div className="px-4 sm:px-6 pb-4">
          <div className="max-w-3xl mx-auto">
            {/* File Reference Badge */}
            {fileId && fileName && (
              <div className="mb-4 flex items-center gap-2">
                <span className="text-xs text-gray-500">Chatting with:</span>
                <FileBadge
                  fileName={fileName}
                  className="!px-2 !py-1 !text-xs"
                />
              </div>
            )}
            <FolderSummaryDisplay folderId={folderId} />
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="relative">
        <InputArea
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          disabled={!folderId}
        />
      </div>

      {/* Loading Overlay */}
      {isLoading && messages.length === 0 && (
        <LoadingOverlay
          isLoading={isLoading}
          message="Loading conversation..."
          fullScreen={false}
        />
      )}

      {/* Learning Notification - Shows during background summarization */}
      <LearningNotification isVisible={isLearning && !fileId} />
    </div>
  );
};

export default ChatInterface;
