import React, { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react';
import { authService } from '../../services/api';
import { UserResponse } from '../../services/types';

/**
 * Authentication context type
 */
interface AuthContextType {
  isAuthenticated: boolean;
  isLoading: boolean;
  user: UserResponse | null;
  error: string | null;
  login: (googleToken: string) => Promise<void>;
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
  const [user, setUser] = useState<UserResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  /**
   * Validate stored token and fetch user information
   */
  const validateToken = useCallback(async (_token?: string): Promise<boolean> => {
    try {
      const userInfo = await authService.getCurrentUser();
      setUser(userInfo);
      setIsAuthenticated(true);
      setError(null);
      return true;
    } catch (err) {
      console.error('Token validation failed:', err);
      // Token is invalid, clear it
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
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
        const token = localStorage.getItem('access_token');
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
  const login = useCallback(async (googleToken: string): Promise<void> => {
    setIsLoading(true);
    setError(null);

    try {
      // Exchange Google token for JWT
      const response = await authService.exchangeToken(googleToken);
      
      // Store tokens
      localStorage.setItem('access_token', response.access_token);
      if (response.refresh_token) {
        localStorage.setItem('refresh_token', response.refresh_token);
      }

      // Validate token and fetch user info
      const isValid = await validateToken(response.access_token);
      if (!isValid) {
        throw new Error('Invalid authentication token');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Login failed';
      setError(errorMessage);
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
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
      // Call logout API (best-effort)
      await authService.logout();
    } catch (err) {
      console.warn('Logout API error:', err);
      // Still clear local state even if API call fails
    } finally {
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      setIsAuthenticated(false);
      setUser(null);
      setError(null);
      setIsLoading(false);
    }
  }, []);

  /**
   * Refresh user information
   */
  const refreshUser = useCallback(async (): Promise<void> => {
    const token = localStorage.getItem('access_token');
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
