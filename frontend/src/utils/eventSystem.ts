/**
 * Robust Event System for Foldex
 * 
 * Provides a reliable event-based communication system that works even if
 * WebSocket messages are delayed or missed. Uses multiple mechanisms:
 * 1. Window custom events (primary)
 * 2. Polling fallback (secondary)
 * 3. LocalStorage synchronization (tertiary)
 */

export type EventType = 
  | 'processing_complete'
  | 'summary_complete'
  | 'graph_complete'
  | 'summary_error'
  | 'graph_error';

export interface EventDetail {
  folder_id: string;
  [key: string]: any;
}

type EventHandler = (detail: EventDetail) => void;

class RobustEventSystem {
  private handlers: Map<EventType, Set<EventHandler>> = new Map();
  private pollIntervals: Map<string, ReturnType<typeof setInterval>> = new Map();
  private eventCache: Map<string, { type: EventType; detail: EventDetail; timestamp: number }> = new Map();

  /**
   * Subscribe to an event type
   */
  on(eventType: EventType, handler: EventHandler): () => void {
    if (!this.handlers.has(eventType)) {
      this.handlers.set(eventType, new Set());
    }
    this.handlers.get(eventType)!.add(handler);

    // Also listen to window events for this type
    const windowHandler = (e: Event) => {
      const customEvent = e as CustomEvent;
      if (customEvent.detail) {
        handler(customEvent.detail);
      }
    };
    window.addEventListener(eventType, windowHandler);

    // Return unsubscribe function
    return () => {
      this.handlers.get(eventType)?.delete(handler);
      window.removeEventListener(eventType, windowHandler);
    };
  }

  /**
   * Emit an event (dispatches to all handlers and window events)
   */
  emit(eventType: EventType, detail: EventDetail): void {
    // Store in cache for polling fallback
    const cacheKey = `${eventType}_${detail.folder_id}`;
    this.eventCache.set(cacheKey, {
      type: eventType,
      detail,
      timestamp: Date.now(),
    });

    // Dispatch to window (for cross-component communication)
    window.dispatchEvent(new CustomEvent(eventType, { detail }));

    // Call all registered handlers
    const handlers = this.handlers.get(eventType);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(detail);
        } catch (error) {
          console.error(`Error in event handler for ${eventType}:`, error);
        }
      });
    }

    // Also store in localStorage as backup (for cross-tab communication)
    try {
      localStorage.setItem(`foldex_event_${cacheKey}`, JSON.stringify({
        type: eventType,
        detail,
        timestamp: Date.now(),
      }));
    } catch (error) {
      // localStorage might be unavailable, ignore
      console.debug('Could not store event in localStorage:', error);
    }
  }

  /**
   * Start polling for events (useful when WebSocket might miss events)
   */
  startPolling(
    folderId: string,
    eventType: EventType,
    checkFn: () => Promise<boolean>,
    options: {
      interval?: number;
      maxAttempts?: number;
      onComplete?: (detail: EventDetail) => void;
    } = {}
  ): () => void {
    const { interval = 2000, maxAttempts = 30, onComplete } = options;
    const pollKey = `${eventType}_${folderId}`;

    // Clear existing polling if any
    this.stopPolling(pollKey);

    let attempts = 0;
    const pollInterval = setInterval(async () => {
      attempts++;

      // Check cache first
      const cacheKey = `${eventType}_${folderId}`;
      const cached = this.eventCache.get(cacheKey);
      if (cached && Date.now() - cached.timestamp < 60000) { // Cache valid for 1 minute
        this.emit(eventType, cached.detail);
        if (onComplete) {
          onComplete(cached.detail);
        }
        this.stopPolling(pollKey);
        return;
      }

      // Check localStorage
      try {
        const stored = localStorage.getItem(`foldex_event_${cacheKey}`);
        if (stored) {
          const eventData = JSON.parse(stored);
          if (Date.now() - eventData.timestamp < 60000) {
            this.emit(eventType, eventData.detail);
            if (onComplete) {
              onComplete(eventData.detail);
            }
            this.stopPolling(pollKey);
            return;
          }
        }
      } catch (error) {
        // Ignore localStorage errors
      }

      // Poll using check function
      try {
        const isComplete = await checkFn();
        if (isComplete) {
          const detail: EventDetail = { folder_id: folderId };
          this.emit(eventType, detail);
          if (onComplete) {
            onComplete(detail);
          }
          this.stopPolling(pollKey);
        }
      } catch (error) {
        console.debug(`Polling error for ${eventType} (${folderId}):`, error);
      }

      if (attempts >= maxAttempts) {
        console.debug(`Polling stopped after ${maxAttempts} attempts for ${eventType} (${folderId})`);
        this.stopPolling(pollKey);
      }
    }, interval);

    this.pollIntervals.set(pollKey, pollInterval);
    
    // Return stop function
    return () => this.stopPolling(pollKey);
  }

  /**
   * Stop polling for a specific event
   */
  stopPolling(pollKey: string): void {
    const interval = this.pollIntervals.get(pollKey);
    if (interval) {
      clearInterval(interval);
      this.pollIntervals.delete(pollKey);
    }
  }

  /**
   * Stop all polling
   */
  stopAllPolling(): void {
    this.pollIntervals.forEach((interval) => clearInterval(interval));
    this.pollIntervals.clear();
  }

  /**
   * Clear event cache
   */
  clearCache(folderId?: string): void {
    if (folderId) {
      const keysToDelete: string[] = [];
      this.eventCache.forEach((_, key) => {
        if (key.endsWith(`_${folderId}`)) {
          keysToDelete.push(key);
        }
      });
      keysToDelete.forEach(key => this.eventCache.delete(key));
    } else {
      this.eventCache.clear();
    }
  }
}

// Singleton instance
export const eventSystem = new RobustEventSystem();

