import unittest, tempfile, os, time, json, sys
from types import SimpleNamespace
from pathlib import Path

# Ensure src is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from landseer_pipeline.pipeline.artifact_cache import ArtifactCache

class DummyDocker:
    def __init__(self, image: str, command: str):
        self.image = image
        self.command = command

class DummyAux:
    def __init__(self, local_path: str, container_path: str):
        self.local_path = local_path
        self.container_path = container_path
        self.required = False
        self.description = None

class ArtifactCacheBasicTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.cache = ArtifactCache(self.root / 'store')
        # dataset meta
        self.meta = self.root / 'dataset_meta.json'
        self.meta.write_text(json.dumps({"name":"cifar10","version":"1"}))
        # model script
        self.model_script = self.root / 'model.py'
        self.model_script.write_text('print("model")')
        # auxiliary file
        self.aux_file = self.root / 'aux.txt'
        self.aux_file.write_text('data-v1')

    def tearDown(self):
        self.tmpdir.cleanup()

    def _make_tool(self):
        docker = DummyDocker('python:3.11-slim','echo test')
        aux = [DummyAux(str(self.aux_file), '/aux/aux.txt')]
        tool = SimpleNamespace(name='dummy', docker=docker, params={'alpha':1}, auxiliary_files=aux, required_inputs=None)
        return tool

    def test_identity_and_node_hash_stability_and_reuse(self):
        tool = self._make_tool()
        tool_hash_1 = self.cache.tool_identity_hash(tool)
        tool_hash_2 = self.cache.tool_identity_hash(tool)
        self.assertEqual(tool_hash_1, tool_hash_2, 'Tool identity hash must be stable')

        ds_hash = self.cache.dataset_hash(self.meta, 'clean')
        model_hash = self.cache.model_hash(str(self.model_script), {'lr':0.1})
        parents = [ds_hash, model_hash]
        node_hash = self.cache.node_hash(parents, tool_hash_1)

        node_dir = self.cache.path_for(node_hash)
        out_dir = node_dir / 'output'
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'model.pt').write_text('weights')
        manifest = {"node_hash":node_hash, "tool_identity":tool_hash_1, "parents":parents, "files":["output/model.pt"]}
        self.cache.write_success(node_hash, manifest)
        self.assertTrue(self.cache.exists(node_hash))
        manifest_path = node_dir / 'manifest.json'
        mtime_before = manifest_path.stat().st_mtime
        # Simulate a cache hit: exists() must not alter files
        self.assertTrue(self.cache.exists(node_hash))
        time.sleep(0.05)  # ensure detectable time delta if rewritten
        mtime_after = manifest_path.stat().st_mtime
        self.assertEqual(mtime_before, mtime_after, 'Manifest should not be rewritten on cache hit')

    def test_auxiliary_change_influences_hash(self):
        tool = self._make_tool()
        h1 = self.cache.tool_identity_hash(tool)
        # Modify auxiliary file content
        self.aux_file.write_text('data-v2')
        h2 = self.cache.tool_identity_hash(tool)
        self.assertNotEqual(h1, h2, 'Auxiliary file content change should update tool identity hash')

if __name__ == '__main__':
    unittest.main()
