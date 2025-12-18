import React, { useState, useRef, useEffect } from 'react';

interface InputAreaProps {
  onSendMessage: (content: string) => Promise<void>;
  isLoading: boolean;
  disabled?: boolean;
}

/**
 * InputArea Component
 * 
 * Message input area with send button, file attachment support, and keyboard shortcuts.
 * Follows Figma wireframe design with gradient button and responsive layout.
 */
const InputArea: React.FC<InputAreaProps> = ({
  onSendMessage,
  isLoading,
  disabled = false,
}) => {
  const [inputValue, setInputValue] = useState('');
  const [isComposing, setIsComposing] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [inputValue]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading || disabled) return;

    const message = inputValue.trim();
    setInputValue('');
    
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    await onSendMessage(message);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Send on Enter (but not Shift+Enter for new line)
    if (e.key === 'Enter' && !e.shiftKey && !isComposing) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handlePaste = async (e: React.ClipboardEvent) => {
    // Handle pasted text
    const pastedText = e.clipboardData.getData('text');
    if (pastedText.length > 1000) {
      e.preventDefault();
      // Could show a warning for very long pastes
    }
  };

  return (
    <div className="border-t border-gray-800 bg-gray-900/80 backdrop-blur-sm p-4 sm:p-6">
      <div className="max-w-3xl mx-auto">
        <form onSubmit={handleSubmit} className="space-y-3">
          {/* Input Container */}
          <div className="flex gap-3 items-end">
            {/* Attachment Button (Future) */}
            <button
              type="button"
              disabled={isLoading || disabled}
              className="flex-shrink-0 w-10 h-10 rounded-lg bg-gray-800 border border-gray-700 flex items-center justify-center text-gray-400 hover:text-gray-200 hover:bg-gray-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Attach file (coming soon)"
            >
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
                  d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"
                />
              </svg>
            </button>

            {/* Text Input */}
            <div className="flex-1 relative">
              <textarea
                ref={textareaRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                onPaste={handlePaste}
                onCompositionStart={() => setIsComposing(true)}
                onCompositionEnd={() => setIsComposing(false)}
                placeholder="Ask anything about your folder..."
                disabled={isLoading || disabled}
                rows={1}
                className={`
                  w-full
                  px-4 py-3
                  bg-gray-800
                  border-2 border-gray-700
                  rounded-lg
                  text-gray-100
                  placeholder:text-gray-500
                  resize-none
                  focus:outline-none focus:ring-2 focus:ring-gray-600 focus:border-transparent
                  transition-all
                  disabled:opacity-50 disabled:cursor-not-allowed
                  max-h-[200px] overflow-y-auto
                `}
                style={{
                  minHeight: '48px',
                }}
              />

              {/* Character count (optional, for very long messages) */}
              {inputValue.length > 500 && (
                <div className="absolute bottom-2 right-2 text-xs text-gray-500">
                  {inputValue.length}/2000
                </div>
              )}
            </div>

            {/* Send Button */}
            <button
              type="submit"
              disabled={!inputValue.trim() || isLoading || disabled}
              className={`
                flex-shrink-0
                h-12 w-12
                rounded-lg
                bg-gray-700 hover:bg-gray-600
                text-white
                flex items-center justify-center
                transition-all duration-200
                disabled:opacity-50 disabled:cursor-not-allowed
                disabled:hover:bg-gray-700
                shadow-lg hover:shadow-xl
                focus:outline-none focus:ring-2 focus:ring-gray-600 focus:ring-offset-2 focus:ring-offset-gray-900
              `}
              title="Send message (Enter)"
            >
              {isLoading ? (
                <svg
                  className="animate-spin h-5 w-5"
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
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
              ) : (
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
                    d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                  />
                </svg>
              )}
            </button>
          </div>

          {/* Helper Text */}
          <div className="flex items-center justify-between text-xs text-gray-500 px-1">
            <div className="flex items-center gap-4">
              <span>Press Enter to send, Shift+Enter for new line</span>
            </div>
            <p className="text-gray-500">
              Foldex can make mistakes. Verify important information.
            </p>
          </div>
        </form>
      </div>
    </div>
  );
};

export default InputArea;

