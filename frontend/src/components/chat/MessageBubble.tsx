import React from 'react';
import { ChatMessage } from '../../services/types';
import CitationDisplay from './CitationDisplay';
import { formatDate } from '../../utils/formatters';

interface MessageBubbleProps {
  message: ChatMessage;
}

/**
 * MessageBubble Component
 * 
 * Individual message component with different styling for user and assistant messages.
 * Follows Figma wireframe design with proper spacing and visual hierarchy.
 */
const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.role === 'user';
  const timestamp = typeof message.timestamp === 'string' 
    ? new Date(message.timestamp) 
    : message.timestamp;

  return (
    <>
      {/* Avatar */}
      {isUser ? (
        <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center">
          <span className="text-sm font-medium text-gray-300">You</span>
        </div>
      ) : (
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
          <p className="text-gray-200 whitespace-pre-wrap break-words">
            {message.content}
          </p>

          {/* Timestamp */}
          <div className="mt-2 pt-2 border-t border-gray-700">
            <span className="text-xs text-gray-500">
              {formatDate(timestamp)}
            </span>
          </div>
        </div>

        {/* Citations */}
        {message.citations && message.citations.length > 0 && (
          <CitationDisplay citations={message.citations} />
        )}
      </div>
    </>
  );
};

export default MessageBubble;
