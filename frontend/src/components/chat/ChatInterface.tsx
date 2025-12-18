import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useChat } from '../../hooks/useChat';
import MessageBubble from './MessageBubble';
import CitationDisplay from './CitationDisplay';

interface ChatInterfaceProps {
  folderId?: string;
}

const ChatInterface: React.FC<ChatInterfaceProps> = () => {
  const { folderId: paramFolderId } = useParams<{ folderId: string }>();
  const folderId = paramFolderId || '';
  const { messages, sendMessage, isLoading } = useChat(folderId);
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    await sendMessage(inputValue);
    setInputValue('');
  };

  return (
    <div className="flex flex-col h-screen">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div key={message.message_id}>
            <MessageBubble message={message} />
            {message.citations && message.citations.length > 0 && (
              <CitationDisplay citations={message.citations} />
            )}
          </div>
        ))}
        {isLoading && (
          <div className="text-gray-500">Thinking...</div>
        )}
      </div>
      <form onSubmit={handleSubmit} className="p-4 border-t">
        <div className="flex gap-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 px-4 py-2 border rounded-lg"
          />
          <button
            type="submit"
            disabled={isLoading}
            className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInterface;

