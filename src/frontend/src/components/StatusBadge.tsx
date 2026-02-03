import { Badge } from '@/components/ui/badge';
import { Loader2, Clock } from 'lucide-react';
import type { TaskStatus, WorkerStatus, WorkflowStatus } from '@/types/api';

interface StatusBadgeProps {
  status: TaskStatus | WorkerStatus | WorkflowStatus | string;
  showIcon?: boolean;
  className?: string;
}

export function StatusBadge({ status, showIcon = true, className }: StatusBadgeProps) {
  const normalizedStatus = status.toLowerCase();
  
  const getVariant = () => {
    switch (normalizedStatus) {
      case 'pending':
        return 'pending';
      case 'running':
        return 'running';
      case 'completed':
        return 'completed';
      case 'failed':
        return 'failed';
      case 'idle':
        return 'idle';
      case 'busy':
        return 'busy';
      case 'offline':
        return 'offline';
      default:
        return 'secondary';
    }
  };

  const getIcon = () => {
    if (!showIcon) return null;
    switch (normalizedStatus) {
      case 'running':
        return <Loader2 className="mr-1 h-3 w-3 animate-spin" />;
      case 'pending':
        return <Clock className="mr-1 h-3 w-3" />;
      default:
        return null;
    }
  };

  return (
    <Badge variant={getVariant()} className={className}>
      {getIcon()}
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </Badge>
  );
}
