import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { getTools, addTool } from '@/lib/api';
import type { ToolInfo, AddToolRequest } from '@/types/api';
import {
  Wrench,
  Plus,
  Box,
  Terminal,
  Container,
  Search,
  Loader2,
  Star,
  Code,
} from 'lucide-react';

export function Tools() {
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [newTool, setNewTool] = useState<AddToolRequest>({
    name: '',
    image: '',
    command: '',
    runtime: '',
    is_baseline: false,
  });
  const queryClient = useQueryClient();

  const { data: toolsData, isLoading, isFetching } = useQuery({
    queryKey: ['tools'],
    queryFn: getTools,
  });

  const addMutation = useMutation({
    mutationFn: addTool,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tools'] });
      setAddDialogOpen(false);
      setNewTool({
        name: '',
        image: '',
        command: '',
        runtime: '',
        is_baseline: false,
      });
    },
  });

  const tools = toolsData?.tools || [];
  const isRefreshing = isFetching && !isLoading;

  const filteredTools = tools.filter((tool) => {
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      return (
        tool.name.toLowerCase().includes(query) ||
        tool.container.image.toLowerCase().includes(query)
      );
    }
    return true;
  });

  const baselineTools = filteredTools.filter((t) => t.is_baseline);
  const attackTools = filteredTools.filter((t) => !t.is_baseline);

  // Only show full loading on initial load
  if (isLoading && !toolsData) {
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
            <h1 className="text-3xl font-bold tracking-tight">Tools</h1>
            {isRefreshing && (
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            )}
          </div>
          <p className="text-muted-foreground">
            Manage pipeline tools and containers
          </p>
        </div>
        <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="mr-2 h-4 w-4" />
              Add Tool
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle>Add New Tool</DialogTitle>
              <DialogDescription>
                Add a new tool to the pipeline registry.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="name">Tool Name</Label>
                <Input
                  id="name"
                  placeholder="my-tool"
                  value={newTool.name}
                  onChange={(e) =>
                    setNewTool({ ...newTool, name: e.target.value })
                  }
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="image">Container Image</Label>
                <Input
                  id="image"
                  placeholder="docker.io/myorg/myimage:latest"
                  value={newTool.image}
                  onChange={(e) =>
                    setNewTool({ ...newTool, image: e.target.value })
                  }
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="command">Command</Label>
                <Input
                  id="command"
                  placeholder="python main.py"
                  value={newTool.command}
                  onChange={(e) =>
                    setNewTool({ ...newTool, command: e.target.value })
                  }
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="runtime">Runtime (optional)</Label>
                <Input
                  id="runtime"
                  placeholder="nvidia"
                  value={newTool.runtime || ''}
                  onChange={(e) =>
                    setNewTool({ ...newTool, runtime: e.target.value })
                  }
                />
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="is_baseline"
                  checked={newTool.is_baseline}
                  onChange={(e) =>
                    setNewTool({ ...newTool, is_baseline: e.target.checked })
                  }
                  className="h-4 w-4 rounded border-gray-300"
                />
                <Label htmlFor="is_baseline">Baseline tool</Label>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setAddDialogOpen(false)}>
                Cancel
              </Button>
              <Button
                onClick={() => addMutation.mutate(newTool)}
                disabled={
                  !newTool.name ||
                  !newTool.image ||
                  !newTool.command ||
                  addMutation.isPending
                }
              >
                {addMutation.isPending && (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                )}
                Add Tool
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Stats */}
      <div className="grid gap-4 sm:grid-cols-3">
        <Card>
          <CardContent className="flex items-center gap-4 p-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-blue-100 dark:bg-blue-900/30">
              <Wrench className="h-6 w-6 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <p className="text-2xl font-bold">{tools.length}</p>
              <p className="text-sm text-muted-foreground">Total Tools</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex items-center gap-4 p-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-yellow-100 dark:bg-yellow-900/30">
              <Star className="h-6 w-6 text-yellow-600 dark:text-yellow-400" />
            </div>
            <div>
              <p className="text-2xl font-bold">{baselineTools.length}</p>
              <p className="text-sm text-muted-foreground">Baseline Tools</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex items-center gap-4 p-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-red-100 dark:bg-red-900/30">
              <Code className="h-6 w-6 text-red-600 dark:text-red-400" />
            </div>
            <div>
              <p className="text-2xl font-bold">{attackTools.length}</p>
              <p className="text-sm text-muted-foreground">Attack Tools</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Search */}
      <Card>
        <CardContent className="p-4">
          <div className="relative max-w-sm">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search tools..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>
        </CardContent>
      </Card>

      {/* Baseline Tools */}
      {baselineTools.length > 0 && (
        <div className="space-y-4">
          <h2 className="flex items-center gap-2 text-lg font-semibold">
            <Star className="h-5 w-5 text-yellow-500" />
            Baseline Tools
          </h2>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {baselineTools.map((tool) => (
              <ToolCard key={tool.name} tool={tool} />
            ))}
          </div>
        </div>
      )}

      {/* Attack Tools */}
      {attackTools.length > 0 && (
        <div className="space-y-4">
          <h2 className="flex items-center gap-2 text-lg font-semibold">
            <Code className="h-5 w-5 text-red-500" />
            Attack Tools
          </h2>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {attackTools.map((tool) => (
              <ToolCard key={tool.name} tool={tool} />
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {tools.length === 0 && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted">
              <Wrench className="h-8 w-8 text-muted-foreground" />
            </div>
            <h3 className="mt-4 text-lg font-semibold">No Tools Available</h3>
            <p className="mt-1 text-sm text-muted-foreground">
              Add tools to start building your pipeline.
            </p>
            <Button className="mt-4" onClick={() => setAddDialogOpen(true)}>
              <Plus className="mr-2 h-4 w-4" />
              Add Tool
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function ToolCard({ tool }: { tool: ToolInfo }) {
  return (
    <Card className="transition-shadow hover:shadow-md">
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
              <Box className="h-5 w-5" />
            </div>
            <div>
              <CardTitle className="text-base">{tool.name}</CardTitle>
              {tool.is_baseline && (
                <Badge variant="outline" className="mt-1 text-xs">
                  <Star className="mr-1 h-3 w-3 text-yellow-500" />
                  Baseline
                </Badge>
              )}
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <div className="space-y-1">
            <p className="flex items-center gap-1 text-xs text-muted-foreground">
              <Container className="h-3 w-3" />
              Image
            </p>
            <p className="truncate text-sm font-mono">{tool.container.image}</p>
          </div>

          <div className="space-y-1">
            <p className="flex items-center gap-1 text-xs text-muted-foreground">
              <Terminal className="h-3 w-3" />
              Command
            </p>
            <p className="truncate text-sm font-mono">{tool.container.command}</p>
          </div>

          {tool.container.runtime && (
            <Badge variant="secondary" className="text-xs">
              Runtime: {tool.container.runtime}
            </Badge>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
