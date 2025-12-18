/**
 * Tests for ErrorDisplay component.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ErrorDisplay from '../../common/ErrorDisplay';
import { APIException } from '../../../services/api';

describe('ErrorDisplay', () => {
  it('renders error message', () => {
    const error = new Error('Test error message');
    render(<ErrorDisplay error={error} />);
    
    expect(screen.getByText('Test error message')).toBeInTheDocument();
  });

  it('renders custom title', () => {
    const error = new Error('Test error');
    render(<ErrorDisplay error={error} title="Custom Error Title" />);
    
    expect(screen.getByText('Custom Error Title')).toBeInTheDocument();
  });

  it('displays APIException with status code', () => {
    const error = new APIException('API Error', 404, 'NotFound');
    render(<ErrorDisplay error={error} />);
    
    expect(screen.getByText('API Error')).toBeInTheDocument();
    expect(screen.getByText(/status: 404/i)).toBeInTheDocument();
  });

  it('calls onRetry when retry button is clicked', async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    const error = new Error('Test error');
    
    render(<ErrorDisplay error={error} onRetry={onRetry} />);
    
    const retryButton = screen.getByText(/retry/i);
    await user.click(retryButton);
    
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it('calls onDismiss when dismiss button is clicked', async () => {
    const user = userEvent.setup();
    const onDismiss = vi.fn();
    const error = new Error('Test error');
    
    render(<ErrorDisplay error={error} onDismiss={onDismiss} />);
    
    const dismissButton = screen.getByLabelText(/dismiss error/i);
    await user.click(dismissButton);
    
    expect(onDismiss).toHaveBeenCalledTimes(1);
  });

  it('does not render when error is null', () => {
    const { container } = render(<ErrorDisplay error={null} />);
    expect(container.firstChild).toBeNull();
  });

  it('hides retry button when onRetry is not provided', () => {
    const error = new Error('Test error');
    render(<ErrorDisplay error={error} />);
    
    expect(screen.queryByText(/retry/i)).not.toBeInTheDocument();
  });
});

