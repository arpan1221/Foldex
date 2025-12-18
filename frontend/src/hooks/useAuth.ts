import { useAuth as useAuthContext } from '../components/auth/AuthContext';

/**
 * useAuth Hook
 * 
 * Re-exports the useAuth hook from AuthContext for convenience.
 * Provides access to authentication state and methods.
 * 
 * @returns {AuthContextType} Authentication context with:
 *   - isAuthenticated: boolean
 *   - isLoading: boolean
 *   - user: User | null
 *   - error: string | null
 *   - login: (token: string) => Promise<void>
 *   - logout: () => Promise<void>
 *   - refreshUser: () => Promise<void>
 * 
 * @example
 * ```tsx
 * import { useAuth } from '../hooks/useAuth';
 * 
 * function MyComponent() {
 *   const { isAuthenticated, user, login, logout } = useAuth();
 *   
 *   if (!isAuthenticated) {
 *     return <LoginButton onLogin={login} />;
 *   }
 *   
 *   return <div>Welcome, {user?.name}!</div>;
 * }
 * ```
 */
export const useAuth = useAuthContext;
