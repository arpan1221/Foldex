import React, { useState } from 'react';

interface DebugData {
  query: string;
  query_embedding_preview?: number[];
  retrieval_metrics?: {
    time_seconds?: number;
    chunk_count?: number;
    average_similarity_score?: number;
  };
  retrieved_chunks?: Array<{
    content: string;
    content_length: number;
    score?: number;
    file_name: string;
    file_id?: string;
    page_number?: number;
    chunk_id?: string;
  }>;
  context?: {
    text?: string;
    token_count?: number;
    length?: number;
  };
  llm_metrics?: {
    time_seconds?: number;
    raw_response?: string;
    cleaned_response?: string;
    raw_length?: number;
    cleaned_length?: number;
  };
  error?: string;
}

interface DebugPanelProps {
  debugData?: DebugData | null;
  isOpen: boolean;
  onClose: () => void;
}

const DebugPanel: React.FC<DebugPanelProps> = ({ debugData, isOpen, onClose }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'chunks' | 'context' | 'llm'>('overview');

  if (!isOpen || !debugData) {
    return null;
  }

  const formatTime = (seconds?: number) => {
    if (!seconds) return 'N/A';
    return `${(seconds * 1000).toFixed(0)}ms`;
  };

  const formatScore = (score?: number) => {
    if (score === undefined || score === null) return 'N/A';
    return score.toFixed(4);
  };

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-lg w-full max-w-6xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className="text-xl font-semibold text-gray-200">Debug Information</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200 transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-700">
          {(['overview', 'chunks', 'context', 'llm'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                activeTab === tab
                  ? 'text-foldex-primary-400 border-b-2 border-foldex-primary-400'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {/* Overview Tab */}
          {activeTab === 'overview' && (
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold text-gray-300 mb-2">Query</h3>
                <div className="bg-gray-800 rounded p-3 text-gray-200 font-mono text-sm">
                  {debugData.query}
                </div>
              </div>

              {debugData.query_embedding_preview && (
                <div>
                  <h3 className="text-sm font-semibold text-gray-300 mb-2">Query Embedding (First 5 dims)</h3>
                  <div className="bg-gray-800 rounded p-3 text-gray-200 font-mono text-xs">
                    [{debugData.query_embedding_preview.map(d => d.toFixed(4)).join(', ')}]
                  </div>
                </div>
              )}

              {debugData.retrieval_metrics && (
                <div>
                  <h3 className="text-sm font-semibold text-gray-300 mb-2">Retrieval Metrics</h3>
                  <div className="bg-gray-800 rounded p-3 space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Retrieval Time:</span>
                      <span className="text-gray-200">{formatTime(debugData.retrieval_metrics.time_seconds)}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Chunks Retrieved:</span>
                      <span className="text-gray-200">{debugData.retrieval_metrics.chunk_count || 0}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Average Similarity Score:</span>
                      <span className="text-gray-200">{formatScore(debugData.retrieval_metrics.average_similarity_score)}</span>
                    </div>
                  </div>
                </div>
              )}

              {debugData.llm_metrics && (
                <div>
                  <h3 className="text-sm font-semibold text-gray-300 mb-2">LLM Metrics</h3>
                  <div className="bg-gray-800 rounded p-3 space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Inference Time:</span>
                      <span className="text-gray-200">{formatTime(debugData.llm_metrics.time_seconds)}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Raw Response Length:</span>
                      <span className="text-gray-200">{debugData.llm_metrics.raw_length || 0} chars</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Cleaned Response Length:</span>
                      <span className="text-gray-200">{debugData.llm_metrics.cleaned_length || 0} chars</span>
                    </div>
                  </div>
                </div>
              )}

              {debugData.context && (
                <div>
                  <h3 className="text-sm font-semibold text-gray-300 mb-2">Context Summary</h3>
                  <div className="bg-gray-800 rounded p-3 space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Context Length:</span>
                      <span className="text-gray-200">{debugData.context.length || 0} chars</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Estimated Tokens:</span>
                      <span className="text-gray-200">{debugData.context.token_count || 'N/A'}</span>
                    </div>
                  </div>
                </div>
              )}

              {debugData.error && (
                <div>
                  <h3 className="text-sm font-semibold text-red-400 mb-2">Error</h3>
                  <div className="bg-red-900/20 border border-red-700 rounded p-3 text-red-300 text-sm">
                    {debugData.error}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Chunks Tab */}
          {activeTab === 'chunks' && (
            <div className="space-y-3">
              <div className="text-sm text-gray-400 mb-4">
                {debugData.retrieved_chunks?.length || 0} chunks retrieved
              </div>
              {debugData.retrieved_chunks?.map((chunk, idx) => (
                <div key={idx} className="bg-gray-800 rounded p-4 border border-gray-700">
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <div className="text-sm font-semibold text-gray-200">
                        {chunk.file_name}
                        {chunk.page_number && <span className="text-gray-400 ml-2">(p.{chunk.page_number})</span>}
                      </div>
                      {chunk.chunk_id && (
                        <div className="text-xs text-gray-500 mt-1">ID: {chunk.chunk_id}</div>
                      )}
                    </div>
                    {chunk.score !== undefined && (
                      <div className="text-sm font-mono text-foldex-primary-400">
                        Score: {formatScore(chunk.score)}
                      </div>
                    )}
                  </div>
                  <div className="text-xs text-gray-400 mb-2">
                    Length: {chunk.content_length} chars
                  </div>
                  <div className="bg-gray-900 rounded p-2 text-xs text-gray-300 font-mono max-h-32 overflow-y-auto">
                    {chunk.content}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Context Tab */}
          {activeTab === 'context' && (
            <div>
              <div className="text-sm text-gray-400 mb-2">
                Context sent to LLM ({debugData.context?.length || 0} chars, ~{debugData.context?.token_count || 'N/A'} tokens)
              </div>
              <div className="bg-gray-800 rounded p-4 border border-gray-700">
                <pre className="text-xs text-gray-300 font-mono whitespace-pre-wrap overflow-x-auto">
                  {debugData.context?.text || 'No context available'}
                </pre>
              </div>
            </div>
          )}

          {/* LLM Tab */}
          {activeTab === 'llm' && (
            <div className="space-y-4">
              {debugData.llm_metrics?.raw_response && (
                <div>
                  <h3 className="text-sm font-semibold text-gray-300 mb-2">Raw LLM Response</h3>
                  <div className="bg-gray-800 rounded p-4 border border-gray-700">
                    <pre className="text-xs text-gray-300 font-mono whitespace-pre-wrap overflow-x-auto max-h-64 overflow-y-auto">
                      {debugData.llm_metrics.raw_response}
                    </pre>
                  </div>
                </div>
              )}
              {debugData.llm_metrics?.cleaned_response && (
                <div>
                  <h3 className="text-sm font-semibold text-gray-300 mb-2">Cleaned Response</h3>
                  <div className="bg-gray-800 rounded p-4 border border-gray-700">
                    <pre className="text-xs text-gray-300 font-mono whitespace-pre-wrap overflow-x-auto max-h-64 overflow-y-auto">
                      {debugData.llm_metrics.cleaned_response}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DebugPanel;

