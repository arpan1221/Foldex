import React, { useState, useEffect, useRef } from 'react';
import { ChatMessage } from '../../services/types';
import CitationDisplay from './CitationDisplay';
import { formatDate } from '../../utils/formatters';
import AIAssistantIcon from '../common/AIAssistantIcon';

interface MessageBubbleProps {
  message: ChatMessage;
  isStreaming?: boolean;
}

/**
 * MessageBubble Component
 * 
 * Individual message component with different styling for user and assistant messages.
 * Follows Figma wireframe design with proper spacing and visual hierarchy.
 * Includes smooth streaming text rendering with typing indicator.
 */
const MessageBubble: React.FC<MessageBubbleProps> = ({ message, isStreaming = false }) => {
  const isUser = message.role === 'user';
  const timestamp = typeof message.timestamp === 'string' 
    ? new Date(message.timestamp) 
    : message.timestamp;
  
  const [displayedContent, setDisplayedContent] = useState(message.content);
  const previousContentRef = useRef(message.content);
  const animationFrameRef = useRef<number | null>(null);

  // Check if content contains HTML (from inline citations)
  const containsHTML = (text: string): boolean => {
    return /<[a-z][\s\S]*>/i.test(text);
  };

  // Parse inline citations [1], [2], etc. and make them clickable (for plain text)
  const renderContentWithCitations = (content: string) => {
    // If content already contains HTML (from inline citations), render it directly
    if (containsHTML(content)) {
      return <div dangerouslySetInnerHTML={{ __html: content }} />;
    }

    if (!message.citations || message.citations.length === 0) {
      return content;
    }

    // Split content by citation pattern [1], [2], etc.
    const citationPattern = /\[(\d+)\]/g;
    const parts: (string | JSX.Element)[] = [];
    let lastIndex = 0;
    let match;

    while ((match = citationPattern.exec(content)) !== null) {
      // Add text before citation
      if (match.index > lastIndex) {
        parts.push(content.substring(lastIndex, match.index));
      }

      // Find the corresponding citation
      const citationNumber = parseInt(match[1], 10);
      const citation = message.citations.find(c => c.citation_number === citationNumber);

      // Add citation as clickable element
      if (citation) {
        parts.push(
          <button
            key={`cite-${match.index}`}
            onClick={() => {
              if (citation.google_drive_url) {
                window.open(citation.google_drive_url, '_blank', 'noopener,noreferrer');
              }
            }}
            title={`${citation.file_name}${citation.page_display ? `, ${citation.page_display}` : ''}`}
            className="inline-flex items-center text-foldex-primary-400 hover:text-foldex-primary-300 hover:underline cursor-pointer font-medium mx-0.5"
          >
            [{citationNumber}]
          </button>
        );
      } else {
        // Citation not found, render as plain text
        parts.push(match[0]);
      }

      lastIndex = match.index + match[0].length;
    }

    // Add remaining text
    if (lastIndex < content.length) {
      parts.push(content.substring(lastIndex));
    }

    return <>{parts}</>;
  };

  // Smooth text updates for streaming
  useEffect(() => {
    if (message.content === previousContentRef.current) {
      return;
    }

    // If content is shorter, it means it was reset (new message)
    if (message.content.length < previousContentRef.current.length) {
      setDisplayedContent(message.content);
      previousContentRef.current = message.content;
      return;
    }

    // For streaming updates, use requestAnimationFrame for smooth rendering
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    animationFrameRef.current = requestAnimationFrame(() => {
      setDisplayedContent(message.content);
      previousContentRef.current = message.content;
    });

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [message.content]);

  // Reset when message changes
  useEffect(() => {
    setDisplayedContent(message.content);
    previousContentRef.current = message.content;
  }, [message.message_id]);

  return (
    <>
      {/* Avatar */}
      {isUser ? (
        <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center">
          <span className="text-sm font-medium text-gray-300">You</span>
        </div>
      ) : (
        <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gray-600 flex items-center justify-center shadow-sm">
          <AIAssistantIcon className="text-gray-200" size="md" />
        </div>
      )}

      {/* Message Content */}
      <div className="flex-1 space-y-2 min-w-0">
        <div
          className={`
            rounded-lg p-4 border transition-all
            ${isUser
              ? 'bg-gray-800 border-gray-700'
              : 'bg-gray-800/50 border-gray-700'
            }
          `}
        >
          {/* Message Text */}
          <div className="text-gray-200 whitespace-pre-wrap break-words leading-relaxed">
            {!isUser && message.citations && message.citations.length > 0 ? (
              renderContentWithCitations(displayedContent)
            ) : (
              <span>{displayedContent}</span>
            )}
            {/* Typing cursor for streaming messages */}
            {isStreaming && !isUser && (
              <span className="inline-block w-0.5 h-4 bg-gray-400 ml-1 align-middle animate-pulse" />
            )}
          </div>

          {/* Timestamp - only show when not streaming */}
          {!isStreaming && (
            <div className="mt-2 pt-2 border-t border-gray-700">
              <span className="text-xs text-gray-500">
                {formatDate(timestamp)}
              </span>
            </div>
          )}
        </div>

        {/* Citations - show immediately when available, even during streaming */}
        {message.citations && message.citations.length > 0 && (
          <CitationDisplay citations={message.citations} />
        )}
      </div>
    </>
  );
};

export default MessageBubble;
