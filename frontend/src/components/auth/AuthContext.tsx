import React, { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react';
import { authService } from '../../services/api';

/**
 * User information interface
 */
export interface User {
  user_id: string;
  email: string;
  name?: string;
  picture?: string;
  google_id?: string;
}

/**
 * Authentication context type
 */
interface AuthContextType {
  isAuthenticated: boolean;
  isLoading: boolean;
  user: User | null;
  error: string | null;
  login: (token: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
}

/**
 * AuthContext
 * 
 * Provides authentication state and methods throughout the application.
 * Handles token storage, user information, and API authentication.
 */
const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [user, setUser] = useState<User | null>(null);
  const [error, setError] = useState<string | null>(null);

  /**
   * Validate stored token and fetch user information
   */
  const validateToken = useCallback(async (token: string): Promise<boolean> => {
    try {
      const userInfo = await authService.getCurrentUser();
      setUser({
        user_id: userInfo.user_id || userInfo.sub || '',
        email: userInfo.email || '',
        name: userInfo.name || userInfo.email?.split('@')[0] || 'User',
        picture: userInfo.picture,
        google_id: userInfo.google_id,
      });
      setIsAuthenticated(true);
      setError(null);
      return true;
    } catch (err) {
      console.error('Token validation failed:', err);
      // Token is invalid, clear it
      localStorage.removeItem('auth_token');
      setIsAuthenticated(false);
      setUser(null);
      return false;
    }
  }, []);

  /**
   * Initialize authentication state on mount
   */
  useEffect(() => {
    const initializeAuth = async () => {
      setIsLoading(true);
      try {
        const token = localStorage.getItem('auth_token');
        if (token) {
          const isValid = await validateToken(token);
          if (!isValid) {
            setError('Session expired. Please sign in again.');
          }
        } else {
          setIsAuthenticated(false);
        }
      } catch (err) {
        console.error('Auth initialization error:', err);
        setError('Failed to initialize authentication');
        setIsAuthenticated(false);
      } finally {
        setIsLoading(false);
      }
    };

    initializeAuth();
  }, [validateToken]);

  /**
   * Login with token
   */
  const login = useCallback(async (token: string): Promise<void> => {
    setIsLoading(true);
    setError(null);

    try {
      // Store token
      localStorage.setItem('auth_token', token);

      // Validate token and fetch user info
      const isValid = await validateToken(token);
      if (!isValid) {
        throw new Error('Invalid authentication token');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Login failed';
      setError(errorMessage);
      localStorage.removeItem('auth_token');
      setIsAuthenticated(false);
      setUser(null);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [validateToken]);

  /**
   * Logout and clear authentication state
   */
  const logout = useCallback(async (): Promise<void> => {
    setIsLoading(true);
    try {
      // Clear token
      localStorage.removeItem('auth_token');
      
      // Reset state
      setIsAuthenticated(false);
      setUser(null);
      setError(null);
    } catch (err) {
      console.error('Logout error:', err);
      // Still clear local state even if API call fails
      localStorage.removeItem('auth_token');
      setIsAuthenticated(false);
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Refresh user information
   */
  const refreshUser = useCallback(async (): Promise<void> => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      await validateToken(token);
    }
  }, [validateToken]);

  const value: AuthContextType = {
    isAuthenticated,
    isLoading,
    user,
    error,
    login,
    logout,
    refreshUser,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

/**
 * useAuth Hook
 * 
 * Custom hook to access authentication context.
 * 
 * @throws {Error} If used outside AuthProvider
 * 
 * @example
 * ```tsx
 * const { isAuthenticated, user, login, logout } = useAuth();
 * ```
 */
export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
