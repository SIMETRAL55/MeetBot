import { render, screen } from '@testing-library/react';
import { JobCard } from '@/components/JobCard';

jest.mock('@/lib/api', () => ({
  api: {
    cancelJob: jest.fn(),
    restartJob: jest.fn(),
    deleteJob: jest.fn()
  }
}));

describe('JobCard', () => {
  const mockJob = {
    id: '123',
    original_filename: 'test-audio.mp3',
    status: 'pending' as const,
    created_at: new Date().toISOString(),
    progress: 0,
    stage_progress: 0
  };

  it('renders job filename and status', () => {
    render(<JobCard job={mockJob} />);
    
    // Check filename
    expect(screen.getByText('test-audio.mp3')).toBeInTheDocument();
    
    // Check status text
    expect(screen.getByText(/Pending/i)).toBeInTheDocument();
  });

  it('shows cancel button when pending', () => {
    render(<JobCard job={mockJob} />);
    
    const stopButton = screen.getByRole('button', { name: /Stop/i });
    expect(stopButton).toBeInTheDocument();
  });
});
