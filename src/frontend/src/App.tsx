import { Routes, Route } from 'react-router-dom';
import { Layout } from '@/components/Layout';
import { Dashboard } from '@/pages/Dashboard';
import { Tasks } from '@/pages/Tasks';
import { Workflows } from '@/pages/Workflows';
import { WorkflowDetail } from '@/pages/WorkflowDetail';
import { Workers } from '@/pages/Workers';
import { Tools } from '@/pages/Tools';
import { Registry } from '@/pages/Registry';
import { Metrics } from '@/pages/Metrics';

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/tasks" element={<Tasks />} />
        <Route path="/workflows" element={<Workflows />} />
        <Route path="/workflows/:id" element={<WorkflowDetail />} />
        <Route path="/workers" element={<Workers />} />
        <Route path="/tools" element={<Tools />} />
        <Route path="/registry" element={<Registry />} />
        <Route path="/metrics" element={<Metrics />} />
        <Route path="/pipelines/:id/metrics" element={<Metrics />} />
      </Routes>
    </Layout>
  );
}

export default App;
