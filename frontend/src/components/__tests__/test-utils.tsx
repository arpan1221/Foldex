/**
 * Test utilities and helpers for React component testing.
 */

import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { vi } from 'vitest';
import { AuthProvider } from '../auth/AuthContext';

/**
 * Custom render function with providers.
 */
const AllTheProviders = ({ children }: { children: React.ReactNode }) => {
  return (
    <BrowserRouter>
      <AuthProvider>
        {children}
      </AuthProvider>
    </BrowserRouter>
  );
};

const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>,
) => render(ui, { wrapper: AllTheProviders, ...options });

export * from '@testing-library/react';
export { customRender as render };

/**
 * Mock localStorage
 */
export const mockLocalStorage = () => {
  const store: Record<string, string> = {};
  
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString();
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      Object.keys(store).forEach(key => delete store[key]);
    },
  };
};

/**
 * Mock WebSocket
 */
export const mockWebSocket = () => {
  const listeners: Record<string, Function[]> = {};
  let readyState = WebSocket.CONNECTING;
  
  const ws = {
    send: vi.fn(),
    close: vi.fn(),
    addEventListener: vi.fn((event: string, callback: Function) => {
      if (!listeners[event]) {
        listeners[event] = [];
      }
      listeners[event].push(callback);
    }),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn((event: Event) => {
      const eventType = event.type;
      if (listeners[eventType]) {
        listeners[eventType].forEach(cb => cb(event));
      }
    }),
    get readyState() {
      return readyState;
    },
    set readyState(value) {
      readyState = value;
    },
    trigger: (event: string, data?: any) => {
      if (listeners[event]) {
        listeners[event].forEach(cb => {
          if (event === 'message') {
            cb({ data: JSON.stringify(data) });
          } else {
            cb(data);
          }
        });
      }
    },
  };
  
  return ws;
};

/**
 * Wait for async operations
 */
export const waitForAsync = () => new Promise(resolve => setTimeout(resolve, 0));

