import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from '@/components/ui/dialog';
import { getRegistryTools, getRegistryEvaluators, addRegistryTool, addRegistryEvaluator, ToolInfo, EvaluatorInfo, AddToolRequest, AddEvaluatorRequest } from '@/lib/api';

export function Registry() {
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [evaluators, setEvaluators] = useState<EvaluatorInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [toolDialogOpen, setToolDialogOpen] = useState(false);
  const [evalDialogOpen, setEvalDialogOpen] = useState(false);
  
  // Form state
  const [newTool, setNewTool] = useState<AddToolRequest>({
    name: '',
    image: '',
    command: '',
    runtime: undefined,
    is_baseline: false,
  });
  
  const [newEvaluator, setNewEvaluator] = useState<AddEvaluatorRequest>({
    name: '',
    image: '',
    command: '',
    required_artifacts: [],
    metrics: [],
    defense_types: [],
  });

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    try {
      setLoading(true);
      const [toolsRes, evalsRes] = await Promise.all([
        getRegistryTools(),
        getRegistryEvaluators(),
      ]);
      setTools(toolsRes.tools);
      setEvaluators(evalsRes.evaluators);
      setError(null);
    } catch (err) {
      setError('Failed to load registry data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function handleAddTool() {
    try {
      await addRegistryTool(newTool);
      setToolDialogOpen(false);
      setNewTool({ name: '', image: '', command: '', runtime: undefined, is_baseline: false });
      loadData();
    } catch (err) {
      console.error('Failed to add tool:', err);
    }
  }

  async function handleAddEvaluator() {
    try {
      await addRegistryEvaluator(newEvaluator);
      setEvalDialogOpen(false);
      setNewEvaluator({ name: '', image: '', command: '', required_artifacts: [], metrics: [], defense_types: [] });
      loadData();
    } catch (err) {
      console.error('Failed to add evaluator:', err);
    }
  }

  // Group tools by category
  const toolsByCategory = tools.reduce((acc, tool) => {
    const category = tool.is_baseline ? 'Baseline' : 
      tool.name.startsWith('pre') ? 'Pre-Training' :
      tool.name.startsWith('in') ? 'In-Training' :
      tool.name.startsWith('post') ? 'Post-Training' :
      tool.name.startsWith('deploy') ? 'Deployment' : 'Other';
    
    if (!acc[category]) acc[category] = [];
    acc[category].push(tool);
    return acc;
  }, {} as Record<string, ToolInfo[]>);

  if (loading) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 w-48 bg-gray-200 rounded" />
          <div className="h-64 bg-gray-200 rounded" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <Card>
          <CardContent className="pt-6">
            <p className="text-red-500">{error}</p>
            <Button onClick={loadData} className="mt-4">Retry</Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Registry</h1>
          <p className="text-gray-600">Manage tools and evaluators</p>
        </div>
        <Button onClick={loadData} variant="outline">Refresh</Button>
      </div>

      <Tabs defaultValue="tools" className="space-y-4">
        <TabsList>
          <TabsTrigger value="tools">Tools ({tools.length})</TabsTrigger>
          <TabsTrigger value="evaluators">Evaluators ({evaluators.length})</TabsTrigger>
        </TabsList>

        <TabsContent value="tools" className="space-y-4">
          <div className="flex justify-end">
            <Dialog open={toolDialogOpen} onOpenChange={setToolDialogOpen}>
              <DialogTrigger asChild>
                <Button>Add Tool</Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Add New Tool</DialogTitle>
                  <DialogDescription>
                    Add a new tool to the registry. For persistence, also update configs/tools.yaml.
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4 py-4">
                  <div className="space-y-2">
                    <Label htmlFor="tool-name">Name</Label>
                    <Input
                      id="tool-name"
                      value={newTool.name}
                      onChange={(e) => setNewTool({ ...newTool, name: e.target.value })}
                      placeholder="my-tool"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="tool-image">Container Image</Label>
                    <Input
                      id="tool-image"
                      value={newTool.image}
                      onChange={(e) => setNewTool({ ...newTool, image: e.target.value })}
                      placeholder="ghcr.io/landseer-project/tools/pre/my-tool:v1"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="tool-command">Command</Label>
                    <Input
                      id="tool-command"
                      value={newTool.command}
                      onChange={(e) => setNewTool({ ...newTool, command: e.target.value })}
                      placeholder="python main.py"
                    />
                  </div>
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="tool-baseline"
                      checked={newTool.is_baseline}
                      onChange={(e) => setNewTool({ ...newTool, is_baseline: e.target.checked })}
                    />
                    <Label htmlFor="tool-baseline">Is Baseline (noop)</Label>
                  </div>
                </div>
                <DialogFooter>
                  <Button variant="outline" onClick={() => setToolDialogOpen(false)}>Cancel</Button>
                  <Button onClick={handleAddTool}>Add Tool</Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>

          {Object.entries(toolsByCategory).map(([category, categoryTools]) => (
            <Card key={category}>
              <CardHeader>
                <CardTitle className="text-lg">{category}</CardTitle>
                <CardDescription>{categoryTools.length} tools</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                  {categoryTools.map((tool) => (
                    <div key={tool.name} className="border rounded-lg p-4 space-y-2">
                      <div className="flex items-center justify-between">
                        <h3 className="font-medium">{tool.name}</h3>
                        {tool.is_baseline && <Badge variant="secondary">baseline</Badge>}
                      </div>
                      <p className="text-sm text-gray-500 truncate" title={tool.container.image}>
                        {tool.container.image}
                      </p>
                      <p className="text-xs text-gray-400 font-mono truncate" title={tool.container.command}>
                        {tool.container.command}
                      </p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </TabsContent>

        <TabsContent value="evaluators" className="space-y-4">
          <div className="flex justify-end">
            <Dialog open={evalDialogOpen} onOpenChange={setEvalDialogOpen}>
              <DialogTrigger asChild>
                <Button>Add Evaluator</Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Add New Evaluator</DialogTitle>
                  <DialogDescription>
                    Add a new evaluator to the registry. For persistence, also update configs/evaluators.yaml.
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4 py-4">
                  <div className="space-y-2">
                    <Label htmlFor="eval-name">Name</Label>
                    <Input
                      id="eval-name"
                      value={newEvaluator.name}
                      onChange={(e) => setNewEvaluator({ ...newEvaluator, name: e.target.value })}
                      placeholder="my-evaluator"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="eval-image">Container Image</Label>
                    <Input
                      id="eval-image"
                      value={newEvaluator.image}
                      onChange={(e) => setNewEvaluator({ ...newEvaluator, image: e.target.value })}
                      placeholder="ghcr.io/landseer-project/evals/my-eval:v1"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="eval-command">Command</Label>
                    <Input
                      id="eval-command"
                      value={newEvaluator.command}
                      onChange={(e) => setNewEvaluator({ ...newEvaluator, command: e.target.value })}
                      placeholder="python evaluate.py"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="eval-metrics">Metrics (comma-separated)</Label>
                    <Input
                      id="eval-metrics"
                      value={newEvaluator.metrics?.join(', ') || ''}
                      onChange={(e) => setNewEvaluator({ ...newEvaluator, metrics: e.target.value.split(',').map(s => s.trim()).filter(Boolean) })}
                      placeholder="accuracy, precision, recall"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="eval-artifacts">Required Artifacts (comma-separated)</Label>
                    <Input
                      id="eval-artifacts"
                      value={newEvaluator.required_artifacts?.join(', ') || ''}
                      onChange={(e) => setNewEvaluator({ ...newEvaluator, required_artifacts: e.target.value.split(',').map(s => s.trim()).filter(Boolean) })}
                      placeholder="watermark_key.json"
                    />
                  </div>
                </div>
                <DialogFooter>
                  <Button variant="outline" onClick={() => setEvalDialogOpen(false)}>Cancel</Button>
                  <Button onClick={handleAddEvaluator}>Add Evaluator</Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            {evaluators.map((evaluator) => (
              <Card key={evaluator.name}>
                <CardHeader>
                  <CardTitle className="text-lg">{evaluator.name}</CardTitle>
                  <CardDescription className="truncate" title={evaluator.container.image}>
                    {evaluator.container.image}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">Command</p>
                    <p className="text-sm font-mono">{evaluator.container.command}</p>
                  </div>
                  
                  {evaluator.metrics.length > 0 && (
                    <div>
                      <p className="text-sm font-medium text-gray-500 mb-1">Metrics</p>
                      <div className="flex flex-wrap gap-1">
                        {evaluator.metrics.map((metric) => (
                          <Badge key={metric} variant="outline">{metric}</Badge>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {evaluator.required_artifacts.length > 0 && (
                    <div>
                      <p className="text-sm font-medium text-gray-500 mb-1">Required Artifacts</p>
                      <div className="flex flex-wrap gap-1">
                        {evaluator.required_artifacts.map((artifact) => (
                          <Badge key={artifact} variant="secondary">{artifact}</Badge>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {evaluator.defense_types.length > 0 && (
                    <div>
                      <p className="text-sm font-medium text-gray-500 mb-1">Defense Types</p>
                      <div className="flex flex-wrap gap-1">
                        {evaluator.defense_types.map((type) => (
                          <Badge key={type}>{type}</Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
