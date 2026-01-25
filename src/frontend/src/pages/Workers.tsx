import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { StatusBadge } from '@/components/StatusBadge';
import { getWorkers, registerWorker } from '@/lib/api';
import { formatRelativeTime, truncateId } from '@/lib/utils';
import type { WorkerInfo } from '@/types/api';
import {
  Users,
  Plus,
  Cpu,
  Activity,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  Server,
  Zap,
} from 'lucide-react';

export function Workers() {
  const [registerDialogOpen, setRegisterDialogOpen] = useState(false);
  const [hostname, setHostname] = useState('');
  const queryClient = useQueryClient();

  const { data: workersData, isLoading, isFetching } = useQuery({
    queryKey: ['workers'],
    queryFn: getWorkers,
  });

  const registerMutation = useMutation({
    mutationFn: (hostname: string) => registerWorker(hostname),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['workers'] });
      setRegisterDialogOpen(false);
      setHostname('');
    },
  });

  const workers = workersData?.workers || [];
  const isRefreshing = isFetching && !isLoading;

  const stats = {
    total: workers.length,
    active: workers.filter((w) => w.status !== 'offline').length,
    busy: workers.filter((w) => w.status === 'busy').length,
    idle: workers.filter((w) => w.status === 'idle').length,
  };

  const totalCompleted = workers.reduce((acc, w) => acc + w.tasks_completed, 0);
  const totalFailed = workers.reduce((acc, w) => acc + w.tasks_failed, 0);

  // Only show full loading on initial load
  if (isLoading && !workersData) {
    return (
      <div className="flex h-[60vh] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-3xl font-bold tracking-tight">Workers</h1>
            {isRefreshing && (
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            )}
          </div>
          <p className="text-muted-foreground">
            Monitor and manage worker nodes
          </p>
        </div>
        <Dialog open={registerDialogOpen} onOpenChange={setRegisterDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="mr-2 h-4 w-4" />
              Register Worker
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Register New Worker</DialogTitle>
              <DialogDescription>
                Add a new worker node to the scheduler pool.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="hostname">Hostname</Label>
                <Input
                  id="hostname"
                  placeholder="worker-1.local"
                  value={hostname}
                  onChange={(e) => setHostname(e.target.value)}
                />
              </div>
            </div>
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => setRegisterDialogOpen(false)}
              >
                Cancel
              </Button>
              <Button
                onClick={() => registerMutation.mutate(hostname)}
                disabled={!hostname || registerMutation.isPending}
              >
                {registerMutation.isPending && (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                )}
                Register
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardContent className="flex items-center gap-4 p-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-blue-100 dark:bg-blue-900/30">
              <Users className="h-6 w-6 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <p className="text-2xl font-bold">{stats.total}</p>
              <p className="text-sm text-muted-foreground">Total Workers</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex items-center gap-4 p-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-green-100 dark:bg-green-900/30">
              <Activity className="h-6 w-6 text-green-600 dark:text-green-400" />
            </div>
            <div>
              <p className="text-2xl font-bold">{stats.active}</p>
              <p className="text-sm text-muted-foreground">Active</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex items-center gap-4 p-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-purple-100 dark:bg-purple-900/30">
              <Cpu className="h-6 w-6 text-purple-600 dark:text-purple-400" />
            </div>
            <div>
              <p className="text-2xl font-bold">{stats.busy}</p>
              <p className="text-sm text-muted-foreground">Busy</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex items-center gap-4 p-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-orange-100 dark:bg-orange-900/30">
              <Zap className="h-6 w-6 text-orange-600 dark:text-orange-400" />
            </div>
            <div>
              <p className="text-2xl font-bold">{totalCompleted}</p>
              <p className="text-sm text-muted-foreground">Tasks Completed</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Worker List */}
      {workers.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {workers.map((worker) => (
            <WorkerCard key={worker.worker_id} worker={worker} />
          ))}
        </div>
      ) : (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted">
              <Users className="h-8 w-8 text-muted-foreground" />
            </div>
            <h3 className="mt-4 text-lg font-semibold">No Workers Registered</h3>
            <p className="mt-1 text-sm text-muted-foreground">
              Register a worker to start executing tasks.
            </p>
            <Button className="mt-4" onClick={() => setRegisterDialogOpen(true)}>
              <Plus className="mr-2 h-4 w-4" />
              Register Worker
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function WorkerCard({ worker }: { worker: WorkerInfo }) {
  const getStatusIcon = () => {
    switch (worker.status) {
      case 'busy':
        return <Loader2 className="h-4 w-4 animate-spin text-purple-500" />;
      case 'idle':
        return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case 'offline':
        return <XCircle className="h-4 w-4 text-gray-400" />;
      default:
        return <Activity className="h-4 w-4 text-blue-500" />;
    }
  };

  return (
    <Card className="transition-shadow hover:shadow-md">
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
              <Server className="h-5 w-5" />
            </div>
            <div>
              <CardTitle className="text-base">{worker.worker_id}</CardTitle>
              <CardDescription className="text-xs">
                {worker.hostname}
              </CardDescription>
            </div>
          </div>
          <StatusBadge status={worker.status} />
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="flex items-center gap-2 text-muted-foreground">
              <Clock className="h-3 w-3" />
              Last heartbeat
            </span>
            <span className="font-medium">
              {formatRelativeTime(worker.last_heartbeat)}
            </span>
          </div>

          <div className="flex items-center justify-between text-sm">
            <span className="flex items-center gap-2 text-muted-foreground">
              <CheckCircle2 className="h-3 w-3 text-green-500" />
              Completed
            </span>
            <span className="font-medium">{worker.tasks_completed}</span>
          </div>

          <div className="flex items-center justify-between text-sm">
            <span className="flex items-center gap-2 text-muted-foreground">
              <XCircle className="h-3 w-3 text-red-500" />
              Failed
            </span>
            <span className="font-medium">{worker.tasks_failed}</span>
          </div>

          {worker.current_task_id && (
            <div className="rounded-lg bg-muted/50 p-2">
              <p className="text-xs text-muted-foreground">Current Task</p>
              <p className="text-sm font-medium">
                {truncateId(worker.current_task_id)}
              </p>
            </div>
          )}

          {worker.capabilities && Object.keys(worker.capabilities).length > 0 && (
            <div className="flex flex-wrap gap-1">
              {Object.entries(worker.capabilities).map(([key, value]) => (
                <Badge key={key} variant="outline" className="text-xs">
                  {key}: {String(value)}
                </Badge>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
