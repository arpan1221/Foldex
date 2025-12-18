/**
 * Tests for GoogleAuth component.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import GoogleAuth from '../../auth/GoogleAuth';
import { useAuth } from '../../../hooks/useAuth';

// Mock useAuth hook
vi.mock('../../../hooks/useAuth', () => ({
  useAuth: vi.fn(),
}));

// Mock useNavigate
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

describe('GoogleAuth', () => {
  const mockLogin = vi.fn();
  const mockUseAuth = useAuth as ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAuth.mockReturnValue({
      isAuthenticated: false,
      login: mockLogin,
      isLoading: false,
      error: null,
    });
  });

  it('renders login button', () => {
    render(
      <BrowserRouter>
        <GoogleAuth />
      </BrowserRouter>
    );

    expect(screen.getByText(/sign in with google/i)).toBeInTheDocument();
  });

  it('displays loading state when authenticating', async () => {
    mockUseAuth.mockReturnValue({
      isAuthenticated: false,
      login: mockLogin,
      isLoading: true,
      error: null,
    });

    render(
      <BrowserRouter>
        <GoogleAuth />
      </BrowserRouter>
    );

    // Should show loading indicator
    expect(screen.getByRole('button')).toBeDisabled();
  });

  it('navigates to folder page when authenticated', () => {
    mockUseAuth.mockReturnValue({
      isAuthenticated: true,
      login: mockLogin,
      isLoading: false,
      error: null,
    });

    render(
      <BrowserRouter>
        <GoogleAuth />
      </BrowserRouter>
    );

    expect(mockNavigate).toHaveBeenCalledWith('/folder');
  });

  it('displays error message when authentication fails', () => {
    mockUseAuth.mockReturnValue({
      isAuthenticated: false,
      login: mockLogin,
      isLoading: false,
      error: 'Authentication failed',
    });

    render(
      <BrowserRouter>
        <GoogleAuth />
      </BrowserRouter>
    );

    expect(screen.getByText(/authentication failed/i)).toBeInTheDocument();
  });

  it('calls login when button is clicked', async () => {
    const userEvent = (await import('@testing-library/user-event')).default;

    render(
      <BrowserRouter>
        <GoogleAuth />
      </BrowserRouter>
    );

    const button = screen.getByText(/sign in with google/i);
    await userEvent.click(button);

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalled();
    });
  });
});

