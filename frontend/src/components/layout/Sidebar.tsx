import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation, useParams } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';
import { folderService, chatService, APIException } from '../../services/api';
import { FolderMetadata, TreeNode, Conversation } from '../../services/types';
import LoadingSpinner from '../common/LoadingSpinner';
import ErrorDisplay from '../common/ErrorDisplay';
import TreeNodeComponent from './TreeNode';

/**
 * Sidebar Component
 * 
 * Navigation sidebar with folder list, search, and quick actions.
 * Follows Figma wireframe design with dark theme and responsive behavior.
 */
const Sidebar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { folderId: currentFolderId, conversationId: currentConversationId } = useParams<{ folderId: string, conversationId: string }>();
  const { isAuthenticated } = useAuth();
  const [folders, setFolders] = useState<FolderMetadata[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<APIException | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sidebarWidth, setSidebarWidth] = useState<number>(() => {
    const saved = localStorage.getItem('sidebarWidth');
    return saved ? parseInt(saved, 10) : 384; // Default 384px (w-96)
  });
  const [isResizing, setIsResizing] = useState(false);
  const [folderTrees, setFolderTrees] = useState<Record<string, TreeNode>>({});
  const [folderConversations, setFolderConversations] = useState<Record<string, Conversation[]>>({});
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [loadingTrees, setLoadingTrees] = useState<Set<string>>(new Set());
  const [loadingConversations, setLoadingConversations] = useState<Set<string>>(new Set());
  const [deletingFolderId, setDeletingFolderId] = useState<string | null>(null);
  const [deletingConversationId, setDeletingConversationId] = useState<string | null>(null);
  const [creatingConversationId, setCreatingConversationId] = useState<string | null>(null);
  const [folderSummaries, setFolderSummaries] = useState<Record<string, any>>({});
  const [summarizingFolders, setSummarizingFolders] = useState<Set<string>>(new Set());
  const [buildingGraphFolders, setBuildingGraphFolders] = useState<Set<string>>(new Set());
  const pollIntervalsRef = React.useRef<Record<string, NodeJS.Timeout>>({});

  // Load folders on mount and when location changes (folder processed)
  useEffect(() => {
    if (isAuthenticated) {
      loadFolders();
    }
  }, [isAuthenticated, location.pathname]);

  // Listen for graph_complete custom events to update sidebar dynamically
  useEffect(() => {
    if (!isAuthenticated) return;

    // Handler to update folder summary when graph completes
    const handleGraphCompleteEvent = async (event: Event) => {
      const customEvent = event as CustomEvent;
      const folderId = customEvent.detail?.folder_id;
      if (!folderId) return;

      // Update folder summary to reflect graph completion
      try {
        const summary = await folderService.getFolderSummary(folderId);
        setFolderSummaries(prev => ({
          ...prev,
          [folderId]: summary  // Store full summary object for consistency
        }));
        console.log('Graph completed, updated sidebar for folder:', folderId, summary.graph_statistics);
        // Clear polling interval if it exists
        if (pollIntervalsRef.current[`graph_${folderId}`]) {
          clearInterval(pollIntervalsRef.current[`graph_${folderId}`]);
          delete pollIntervalsRef.current[`graph_${folderId}`];
        }
        setBuildingGraphFolders(prev => {
          const newSet = new Set(prev);
          newSet.delete(folderId);
          return newSet;
        });
      } catch (err) {
        console.debug('Failed to update folder summary after graph completion:', err);
        setBuildingGraphFolders(prev => {
          const newSet = new Set(prev);
          newSet.delete(folderId);
          return newSet;
        });
      }
    };

    // Listen for custom graph_complete and summary_complete events
    const handleSummaryCompleteEvent = async (event: Event) => {
      const customEvent = event as CustomEvent;
      const folderId = customEvent.detail?.folder_id;
      if (!folderId) return;

      // Update folder summary and clear summarizing state
      try {
        const summary = await folderService.getFolderSummary(folderId);
        setFolderSummaries(prev => ({
          ...prev,
          [folderId]: summary  // Store full summary object, not just graph_statistics
        }));
        console.log('Summary completed, updated sidebar for folder:', folderId);
        // Clear polling interval if it exists
        if (pollIntervalsRef.current[`summary_${folderId}`]) {
          clearInterval(pollIntervalsRef.current[`summary_${folderId}`]);
          delete pollIntervalsRef.current[`summary_${folderId}`];
        }
        // Clear summarizing state
        setSummarizingFolders(prev => {
          const newSet = new Set(prev);
          newSet.delete(folderId);
          return newSet;
        });
      } catch (err) {
        console.debug('Failed to update folder summary after completion:', err);
        setSummarizingFolders(prev => {
          const newSet = new Set(prev);
          newSet.delete(folderId);
          return newSet;
        });
      }
    };

    window.addEventListener('graph_complete', handleGraphCompleteEvent);
    window.addEventListener('summary_complete', handleSummaryCompleteEvent);

    return () => {
      window.removeEventListener('graph_complete', handleGraphCompleteEvent);
      window.removeEventListener('summary_complete', handleSummaryCompleteEvent);
      // Cleanup any polling intervals
      Object.values(pollIntervalsRef.current).forEach(interval => clearInterval(interval));
      pollIntervalsRef.current = {};
    };
  }, [isAuthenticated]);

  // Save sidebar width to localStorage
  useEffect(() => {
    localStorage.setItem('sidebarWidth', sidebarWidth.toString());
  }, [sidebarWidth]);

  // Handle resizing
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;
      const newWidth = e.clientX;
      const minWidth = 240; // Minimum sidebar width
      const maxWidth = 600; // Maximum sidebar width
      setSidebarWidth(Math.max(minWidth, Math.min(maxWidth, newWidth)));
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isResizing]);

  const loadFolders = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const foldersData = await folderService.getUserFolders();
      setFolders(foldersData);
    } catch (err) {
      const apiError = err instanceof APIException 
        ? err 
        : new APIException(err instanceof Error ? err.message : 'Failed to load folders');
      setError(apiError);
      console.error('Failed to load folders:', apiError);
      setFolders([]);
    } finally {
      setIsLoading(false);
    }
  };

  const filteredFolders = folders.filter((folder) =>
    folder.folder_name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const loadFolderConversations = async (folderId: string) => {
    setLoadingConversations(prev => new Set(prev).add(folderId));
    try {
      const conversations = await chatService.getFolderConversations(folderId);
      setFolderConversations(prev => ({ ...prev, [folderId]: conversations }));
    } catch (err) {
      console.error(`Failed to load conversations for folder ${folderId}:`, err);
    } finally {
      setLoadingConversations(prev => {
        const newSet = new Set(prev);
        newSet.delete(folderId);
        return newSet;
      });
    }
  };

  const handleFolderClick = async (folderId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    
    // Check if we already have conversations for this folder
    const conversations = folderConversations[folderId] || [];
    
    if (conversations.length > 0) {
      // Navigate to the most recent conversation
      navigate(`/chat/${folderId}/${conversations[0].conversation_id}`);
    } else {
      // Start a new conversation
      handleNewConversation(folderId, e);
    }
  };

  const handleConversationClick = (folderId: string, conversationId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    navigate(`/chat/${folderId}/${conversationId}`);
  };

  const handleNewConversation = async (folderId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setCreatingConversationId(folderId);
    try {
      const newConv = await chatService.createConversation(folderId, 'New Chat');
      setFolderConversations(prev => ({
        ...prev,
        [folderId]: [newConv, ...(prev[folderId] || [])]
      }));
      navigate(`/chat/${folderId}/${newConv.conversation_id}`);
    } catch (err) {
      console.error('Failed to create new conversation:', err);
      alert('Failed to start a new chat. Please try again.');
    } finally {
      setCreatingConversationId(null);
    }
  };

  const handleFileClick = async (fileId: string, fileName: string, folderId: string) => {
    try {
      // Create a new conversation for file-specific chat
      const newConv = await chatService.createConversation(folderId, `Chat: ${fileName}`);
      // Navigate to chat with file_id in URL params (we'll handle this in ChatInterface)
      navigate(`/chat/${folderId}/${newConv.conversation_id}?file_id=${fileId}&file_name=${encodeURIComponent(fileName)}`);
    } catch (err) {
      console.error('Failed to start file chat:', err);
      alert('Failed to start chat with file. Please try again.');
    }
  };

  const handleFolderExpand = async (folderId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    
    const isExpanding = !expandedNodes.has(folderId);
    
    // Toggle expansion
    setExpandedNodes(prev => {
      const newSet = new Set(prev);
      if (newSet.has(folderId)) {
        newSet.delete(folderId);
      } else {
        newSet.add(folderId);
      }
      return newSet;
    });
    
    if (isExpanding) {
      // Load tree if not already loaded
      if (!folderTrees[folderId]) {
        setLoadingTrees(prev => new Set(prev).add(folderId));
        try {
          const tree = await folderService.getFolderTree(folderId);
          setFolderTrees(prev => ({ ...prev, [folderId]: tree }));
        } catch (err) {
          console.error('Failed to load folder tree:', err);
        } finally {
          setLoadingTrees(prev => {
            const newSet = new Set(prev);
            newSet.delete(folderId);
            return newSet;
          });
        }
      }
      
      // Always reload conversations when expanding
      loadFolderConversations(folderId);
      
      // Load folder summary to check if graph is available
      if (!folderSummaries[folderId]) {
        try {
          const summary = await folderService.getFolderSummary(folderId);
          setFolderSummaries(prev => ({
            ...prev,
            [folderId]: summary  // Store full summary object for consistency
          }));
        } catch (err) {
          // Silently fail - graph might not be ready yet
          console.debug('Could not load folder summary for graph check:', err);
        }
      }
    }
  };
  
  const handleGraphVisualization = (folderId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    navigate(`/graph/${folderId}`);
  };

  const handleSummarizeFolder = async (folderId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    
    setSummarizingFolders(prev => new Set(prev).add(folderId));
    try {
      await folderService.generateFolderSummary(folderId);
      // Success - WebSocket will notify us when complete, but also poll as fallback
      // Clear any existing polling interval for this folder
      if (pollIntervalsRef.current[`summary_${folderId}`]) {
        clearInterval(pollIntervalsRef.current[`summary_${folderId}`]);
      }
      // Poll for completion every 2 seconds for up to 60 seconds
      let attempts = 0;
      const maxAttempts = 30;
      pollIntervalsRef.current[`summary_${folderId}`] = setInterval(async () => {
        attempts++;
        try {
          const summary = await folderService.getFolderSummary(folderId);
          // Check if summary is complete (has summary text or learning_status is complete)
          if (summary.summary || summary.learning_status === 'learning_complete') {
            setFolderSummaries(prev => ({
              ...prev,
              [folderId]: summary
            }));
            setSummarizingFolders(prev => {
              const newSet = new Set(prev);
              newSet.delete(folderId);
              return newSet;
            });
            if (pollIntervalsRef.current[`summary_${folderId}`]) {
              clearInterval(pollIntervalsRef.current[`summary_${folderId}`]);
              delete pollIntervalsRef.current[`summary_${folderId}`];
            }
          }
        } catch (err) {
          console.debug('Polling for summary completion:', err);
        }
        
        if (attempts >= maxAttempts) {
          // Stop polling after max attempts
          if (pollIntervalsRef.current[`summary_${folderId}`]) {
            clearInterval(pollIntervalsRef.current[`summary_${folderId}`]);
            delete pollIntervalsRef.current[`summary_${folderId}`];
          }
          setSummarizingFolders(prev => {
            const newSet = new Set(prev);
            newSet.delete(folderId);
            return newSet;
          });
        }
      }, 2000);
    } catch (err) {
      console.error('Failed to start folder summarization:', err);
      alert('Failed to start folder summarization. Please try again.');
      setSummarizingFolders(prev => {
        const newSet = new Set(prev);
        newSet.delete(folderId);
        return newSet;
      });
    }
  };


  const handleBuildKnowledgeGraph = async (folderId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    
    setBuildingGraphFolders(prev => new Set(prev).add(folderId));
    try {
      await folderService.buildKnowledgeGraph(folderId);
      // Success - WebSocket will notify us when complete, but also poll as fallback
      // Clear any existing polling interval for this folder
      if (pollIntervalsRef.current[`graph_${folderId}`]) {
        clearInterval(pollIntervalsRef.current[`graph_${folderId}`]);
      }
      // Poll for completion every 2 seconds for up to 120 seconds (graph building can take longer)
      let attempts = 0;
      const maxAttempts = 60;
      pollIntervalsRef.current[`graph_${folderId}`] = setInterval(async () => {
        attempts++;
        try {
          const summary = await folderService.getFolderSummary(folderId);
          // Check if graph is complete (has graph_statistics with node_count > 0)
          if (summary.graph_statistics && summary.graph_statistics.node_count > 0) {
            setFolderSummaries(prev => ({
              ...prev,
              [folderId]: summary
            }));
            setBuildingGraphFolders(prev => {
              const newSet = new Set(prev);
              newSet.delete(folderId);
              return newSet;
            });
            if (pollIntervalsRef.current[`graph_${folderId}`]) {
              clearInterval(pollIntervalsRef.current[`graph_${folderId}`]);
              delete pollIntervalsRef.current[`graph_${folderId}`];
            }
          }
        } catch (err) {
          console.debug('Polling for graph completion:', err);
        }
        
        if (attempts >= maxAttempts) {
          // Stop polling after max attempts
          if (pollIntervalsRef.current[`graph_${folderId}`]) {
            clearInterval(pollIntervalsRef.current[`graph_${folderId}`]);
            delete pollIntervalsRef.current[`graph_${folderId}`];
          }
          setBuildingGraphFolders(prev => {
            const newSet = new Set(prev);
            newSet.delete(folderId);
            return newSet;
          });
        }
      }, 2000);
    } catch (err) {
      console.error('Failed to start knowledge graph build:', err);
      alert('Failed to start knowledge graph build. Please try again.');
      setBuildingGraphFolders(prev => {
        const newSet = new Set(prev);
        newSet.delete(folderId);
        return newSet;
      });
    }
  };

  const handleToggleNode = (nodeId: string) => {
    setExpandedNodes(prev => {
      const newSet = new Set(prev);
      if (newSet.has(nodeId)) {
        newSet.delete(nodeId);
      } else {
        newSet.add(nodeId);
      }
      return newSet;
    });
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const formatDate = (dateStr?: string | Date): string => {
    if (!dateStr) return 'Unknown';
    const date = typeof dateStr === 'string' ? new Date(dateStr) : dateStr;
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: date.getFullYear() !== new Date().getFullYear() ? 'numeric' : undefined,
    }).format(date);
  };

  const handleNewFolder = () => {
    navigate('/folder');
  };

  const handleDeleteFolder = async (folderId: string, folderName: string, e: React.MouseEvent) => {
    e.stopPropagation();
    
    if (!window.confirm(`Are you sure you want to delete "${folderName}"? This will remove all indexed data for this folder.`)) {
      return;
    }

    setDeletingFolderId(folderId);
    try {
      await folderService.deleteFolder(folderId);
      
      // Remove from local state
      setFolders(prev => prev.filter(f => f.folder_id !== folderId));
      setExpandedNodes(prev => {
        const newSet = new Set(prev);
        newSet.delete(folderId);
        return newSet;
      });
      setFolderTrees(prev => {
        const newTrees = { ...prev };
        delete newTrees[folderId];
        return newTrees;
      });
      
      // If we're currently viewing this folder, navigate away
      if (location.pathname === `/chat/${folderId}`) {
        navigate('/folder');
      }
    } catch (err) {
      console.error('Failed to delete folder:', err);
      alert('Failed to delete folder. Please try again.');
    } finally {
      setDeletingFolderId(null);
    }
  };

  const handleDeleteConversation = async (folderId: string, conversationId: string, conversationTitle: string, e: React.MouseEvent) => {
    e.stopPropagation();
    
    if (!window.confirm(`Are you sure you want to delete "${conversationTitle}"? This action cannot be undone.`)) {
      return;
    }

    setDeletingConversationId(conversationId);
    try {
      await chatService.deleteConversation(conversationId);
      
      // Remove from local state
      setFolderConversations(prev => {
        const conversations = prev[folderId] || [];
        return {
          ...prev,
          [folderId]: conversations.filter(c => c.conversation_id !== conversationId)
        };
      });
      
      // If we're currently viewing this conversation, navigate to folder
      if (currentConversationId === conversationId && currentFolderId === folderId) {
        navigate(`/chat/${folderId}`);
      }
    } catch (err) {
      console.error('Failed to delete conversation:', err);
      alert('Failed to delete conversation. Please try again.');
    } finally {
      setDeletingConversationId(null);
    }
  };

  if (!isAuthenticated) {
    return null;
  }

  return (
    <>
      <aside
        className="bg-gray-900 border-r border-gray-800 flex flex-col relative"
        style={{ 
          width: `${sidebarWidth}px`,
          transition: isResizing ? 'none' : 'width 0.2s ease'
        }}
      >
        {/* Sidebar Header */}
        <div className="p-4 border-b border-gray-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
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
                d="M3.75 9.776c.112-.017.227-.026.344-.026h15.812c.117 0 .232.009.344.026m-16.5 0a2.25 2.25 0 00-1.883 2.542l.857 6a2.25 2.25 0 002.227 1.932H19.05a2.25 2.25 0 002.227-1.932l.857-6a2.25 2.25 0 00-1.883-2.542m-16.5 0V6A2.25 2.25 0 016 3.75h3.879a1.5 1.5 0 011.06.44l2.122 2.12a1.5 1.5 0 001.06.44H18A2.25 2.25 0 0120.25 9v.776"
              />
            </svg>
            <h2 className="text-xl font-semibold text-gray-100">Folders</h2>
          </div>
        </div>

      {/* Search */}
      <div className="p-4 border-b border-gray-800">
          <div className="relative">
            <svg
              className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
            <input
              type="text"
              placeholder="Search folders..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 bg-gray-800 border border-gray-700 rounded-lg text-base text-gray-100 placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-gray-600 focus:border-transparent"
            />
          </div>
        </div>

      {/* New Folder Button */}
      <div className="p-4 border-b border-gray-800">
          <button
            onClick={handleNewFolder}
            className="w-full px-4 py-2.5 bg-gray-700 hover:bg-gray-600 text-base text-white rounded-lg font-medium transition-all flex items-center justify-center gap-2"
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
                d="M12 4v16m8-8H4"
              />
            </svg>
            <span>New Folder</span>
          </button>
        </div>

      {/* Folders List */}
      <div className="flex-1 overflow-y-auto">
        {error && (
          <div className="p-4">
            <ErrorDisplay
              error={error}
              title="Failed to load folders"
              onRetry={loadFolders}
              onDismiss={() => setError(null)}
            />
          </div>
        )}
        {isLoading ? (
          <div className="p-4 flex items-center justify-center">
            <LoadingSpinner size="sm" />
          </div>
        ) : filteredFolders.length === 0 && !error ? (
          <div className="p-4 text-center">
            <svg
              className="w-12 h-12 text-gray-600 mx-auto mb-3"
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
            <p className="text-gray-400 text-base mb-2">No folders yet</p>
            <button
              onClick={handleNewFolder}
              className="text-gray-300 hover:text-gray-200 text-base font-medium"
            >
              Add your first folder
            </button>
          </div>
        ) : (
          <div className="p-2">
            <div className="text-sm font-medium text-gray-500 px-2 py-1 mb-1">
              {filteredFolders.length} {filteredFolders.length === 1 ? 'folder' : 'folders'}
            </div>
            {filteredFolders.map((folder) => {
              const isActive = currentFolderId === folder.folder_id;
              const isExpanded = expandedNodes.has(folder.folder_id);
              const tree = folderTrees[folder.folder_id];
              const conversations = folderConversations[folder.folder_id] || [];
              const isLoadingTree = loadingTrees.has(folder.folder_id);
              const isLoadingConvs = loadingConversations.has(folder.folder_id);
              const isCreatingConv = creatingConversationId === folder.folder_id;
              
              return (
                <div key={folder.folder_id} className="mb-2">
                  <div className="flex items-center group">
                    <button
                      onClick={(e) => handleFolderExpand(folder.folder_id, e)}
                      className="p-1.5 rounded-lg hover:bg-gray-800 text-gray-500 hover:text-gray-300 transition-colors mr-1"
                    >
                      <svg
                        className={`w-4 h-4 transition-transform duration-200 ${isExpanded ? 'rotate-90' : ''}`}
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    </button>
                    <button
                      onClick={(e) => handleFolderClick(folder.folder_id, e)}
                      className={`
                        flex-1 flex items-center gap-3 px-3 py-2 rounded-lg
                        transition-all duration-200
                        ${isActive && !currentConversationId 
                          ? 'bg-blue-600/10 text-blue-400 ring-1 ring-blue-500/50' 
                          : 'text-gray-300 hover:bg-gray-800/50 hover:text-white'}
                      `}
                    >
                      <svg
                        className={`w-5 h-5 flex-shrink-0 ${isActive && !currentConversationId ? 'text-blue-400' : 'text-gray-400 group-hover:text-gray-300'}`}
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                      </svg>
                      <span className="text-base font-medium truncate">{folder.folder_name}</span>
                    </button>
                    <div className="flex items-center opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={(e) => handleNewConversation(folder.folder_id, e)}
                        disabled={isCreatingConv}
                        className="p-1.5 text-gray-500 hover:text-green-400 transition-colors"
                        title="New Chat"
                      >
                        {isCreatingConv ? (
                          <div className="w-4 h-4 border-2 border-green-400 border-t-transparent rounded-full animate-spin"></div>
                        ) : (
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                          </svg>
                        )}
                      </button>
                      <button
                        onClick={(e) => handleDeleteFolder(folder.folder_id, folder.folder_name, e)}
                        className="p-1.5 text-gray-500 hover:text-red-400 transition-colors"
                        title="Delete Folder"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  </div>

                  {isExpanded && (
                    <div className="ml-4 mt-1 space-y-1 border-l border-gray-800 pl-2">
                      {/* Conversations Section */}
                      <div className="py-1">
                        <div className="flex items-center justify-between px-2 mb-1">
                          <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Chats</span>
                          {isLoadingConvs && <div className="w-3 h-3 border border-gray-500 border-t-transparent rounded-full animate-spin"></div>}
                        </div>
                        
                        {conversations.length > 0 ? (
                          <div className="space-y-0.5">
                            {conversations.map((conv) => (
                              <div
                                key={conv.conversation_id}
                                className="group/chat-item flex items-center gap-1"
                              >
                                {/* Active indicator dot */}
                                {currentConversationId === conv.conversation_id && (
                                  <div className="w-1.5 h-1.5 bg-blue-400 rounded-full flex-shrink-0 ml-1"></div>
                                )}
                                {currentConversationId !== conv.conversation_id && (
                                  <div className="w-1.5 h-1.5 flex-shrink-0 ml-1"></div>
                                )}
                                
                                <button
                                  onClick={(e) => handleConversationClick(folder.folder_id, conv.conversation_id, e)}
                                  className={`
                                    flex-1 flex items-center gap-2 px-2 py-1.5 rounded-md text-left group/chat
                                    transition-all duration-200
                                    ${currentConversationId === conv.conversation_id
                                      ? 'text-blue-300'
                                    : 'text-gray-400 hover:bg-gray-800/50 hover:text-gray-200'}
                                `}
                              >
                                <svg className={`w-3.5 h-3.5 flex-shrink-0 ${currentConversationId === conv.conversation_id ? 'text-blue-400' : 'text-gray-500 group-hover/chat:text-gray-400'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                                </svg>
                                <span className="text-sm font-medium truncate flex-1">{conv.title}</span>
                              </button>
                                <button
                                  onClick={(e) => handleDeleteConversation(folder.folder_id, conv.conversation_id, conv.title, e)}
                                  disabled={deletingConversationId === conv.conversation_id}
                                  className={`
                                    px-1.5 py-1.5 rounded-md opacity-0 group-hover/chat-item:opacity-100
                                    transition-all duration-200
                                    ${deletingConversationId === conv.conversation_id
                                      ? 'opacity-100 cursor-wait'
                                      : 'hover:bg-red-600/20 hover:text-red-400'}
                                    text-gray-500 hover:text-red-400
                                  `}
                                  title="Delete conversation"
                                >
                                  {deletingConversationId === conv.conversation_id ? (
                                    <svg className="w-3.5 h-3.5 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                    </svg>
                                  ) : (
                                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                    </svg>
                                  )}
                                </button>
                              </div>
                            ))}
                          </div>
                        ) : !isLoadingConvs && (
                          <div className="px-2 py-1 text-xs text-gray-600 italic">No chats yet</div>
                        )}
                      </div>

                      {/* Folder Actions Section */}
                      <div className="py-2 border-t border-gray-800 mt-2 space-y-1">
                        <div className="px-2 mb-1">
                          <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Actions</span>
                        </div>
                        
                        {/* Summarize Folder Button - shows loading state during summarization */}
                        {summarizingFolders.has(folder.folder_id) ? (
                          <button
                            disabled={true}
                            className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-left text-xs font-medium text-blue-400 opacity-50 cursor-not-allowed"
                            title="Summarizing folder contents..."
                          >
                            <svg className="w-3.5 h-3.5 flex-shrink-0 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                            </svg>
                            <span>Summarizing...</span>
                          </button>
                        ) : (
                          <button
                            onClick={(e) => handleSummarizeFolder(folder.folder_id, e)}
                            className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-left text-xs font-medium text-blue-400 hover:bg-blue-600/20 hover:text-blue-300 transition-all duration-200"
                            title={folderSummaries[folder.folder_id]?.summary ? "Regenerate folder summary" : "Summarize folder contents for general queries"}
                          >
                            <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            <span>Summarize folder contents</span>
                          </button>
                        )}

                        {/* Build Knowledge Graph Button - transforms to Knowledge Graph button after build */}
                        {folderSummaries[folder.folder_id]?.graph_statistics && folderSummaries[folder.folder_id].graph_statistics!.node_count > 0 ? (
                          // Show "Knowledge Graph" button if graph exists
                          <button
                            onClick={(e) => handleGraphVisualization(folder.folder_id, e)}
                            className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-left text-xs font-medium text-purple-400 hover:bg-purple-600/20 hover:text-purple-300 transition-all duration-200"
                            title="Visualize Knowledge Graph"
                          >
                            <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                            </svg>
                            <span>Knowledge Graph</span>
                          </button>
                        ) : (
                          // Show "Build Knowledge Graph" button if graph doesn't exist
                          <button
                            onClick={(e) => handleBuildKnowledgeGraph(folder.folder_id, e)}
                            disabled={buildingGraphFolders.has(folder.folder_id)}
                            className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-left text-xs font-medium text-purple-400 hover:bg-purple-600/20 hover:text-purple-300 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                            title="Build knowledge graph for relations between documents"
                          >
                            {buildingGraphFolders.has(folder.folder_id) ? (
                              <>
                                <svg className="w-3.5 h-3.5 flex-shrink-0 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                </svg>
                                <span>Building graph...</span>
                              </>
                            ) : (
                              <>
                                <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                                </svg>
                                <span>Build knowledge graph</span>
                              </>
                            )}
                          </button>
                        )}
                      </div>
                      
                      {/* Files Section (Optional Toggle) */}
                      <div className="pt-2">
                        <div className="px-2 mb-1">
                          <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Files</span>
                        </div>
                        {isLoadingTree ? (
                          <div className="py-2 flex justify-center"><div className="w-4 h-4 border-2 border-gray-700 border-t-transparent rounded-full animate-spin"></div></div>
                        ) : tree?.children?.length ? (
                          tree.children.map((child) => (
                            <TreeNodeComponent
                              key={child.id}
                              node={child}
                              level={0}
                              expandedNodes={expandedNodes}
                              onToggleExpand={handleToggleNode}
                              formatFileSize={formatFileSize}
                              formatDate={formatDate}
                              onFileClick={handleFileClick}
                              folderId={folder.folder_id}
                            />
                          ))
                        ) : (
                          <div className="px-2 py-1 text-xs text-gray-600 italic">Empty</div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

        {/* Resize Handle - wider invisible area with visible border on hover */}
        <div
          onMouseDown={(e) => {
            e.preventDefault();
            setIsResizing(true);
          }}
          className={`
            absolute top-0 right-0 w-1 h-full cursor-col-resize z-10
            hover:bg-gray-600 transition-colors
            ${isResizing ? 'bg-gray-600' : 'bg-transparent'}
          `}
          title="Drag to resize sidebar"
        />
      </aside>
    </>
  );
};

export default Sidebar;

