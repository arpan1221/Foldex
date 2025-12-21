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
          <div 
            className="flex-1 relative rounded-2xl bg-gray-800 border-2 border-gray-700 focus-within:border-gray-500 transition-colors"
            style={{ outline: 'none' }}
            onFocus={(e) => e.currentTarget.style.outline = 'none'}
            onBlur={(e) => e.currentTarget.style.outline = 'none'}
          >
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
                px-4 pr-14
                bg-transparent
                border-0
                rounded-2xl
                text-gray-100
                placeholder:text-gray-500
                resize-none
                focus:outline-none
                focus:ring-0
                focus:border-0
                transition-all
                disabled:opacity-50 disabled:cursor-not-allowed
                max-h-[200px] overflow-y-auto
              `}
              style={{
                minHeight: '48px',
                paddingTop: '12px',
                paddingBottom: '12px',
                lineHeight: '24px',
                outline: 'none',
              }}
              onFocus={(e) => {
                e.currentTarget.style.outline = 'none';
                e.currentTarget.style.boxShadow = 'none';
              }}
            />

            {/* Send Button - Inside input, vertically centered */}
            <button
              type="submit"
              disabled={!inputValue.trim() || isLoading || disabled}
              className={`
                absolute
                right-2
                top-1/2
                -translate-y-1/2
                h-8 w-8
                rounded-xl
                bg-gray-700 hover:bg-gray-600
                text-white
                flex items-center justify-center
                transition-all duration-200
                disabled:opacity-50 disabled:cursor-not-allowed
                disabled:hover:bg-gray-700
                shadow-md hover:shadow-lg
                focus:outline-none
              `}
              title="Send message (Enter)"
            >
              {isLoading ? (
                <svg
                  className="animate-spin h-4 w-4"
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
                  className="w-4 h-4"
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

            {/* Character count (optional, for very long messages) */}
            {inputValue.length > 500 && (
              <div className="absolute bottom-12 right-2 text-xs text-gray-500">
                {inputValue.length}/2000
              </div>
            )}
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

