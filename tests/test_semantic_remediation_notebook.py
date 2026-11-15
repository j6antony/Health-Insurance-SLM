import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from self_instruct_pipeline.config import ROOT


class RemediationNotebookTests(unittest.TestCase):
    def test_verified_bundle_replaces_stale_project_and_installs_profile(self):
        notebook = ROOT / 'distillation_colab_semantic_remediation_round1.ipynb'
        cells = json.loads(notebook.read_text())['cells']
        trees = [ast.parse(''.join(cell['source'])) for cell in cells if cell['cell_type'] == 'code']
        nodes = [node for tree in trees for node in tree.body]
        helper = next(node for node in nodes if isinstance(node, ast.FunctionDef)
                      and node.name == 'verified_project')
        checksum = next(ast.literal_eval(node.value) for node in nodes if isinstance(node, ast.Assign)
                        and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id == 'EXPECTED_BUNDLE_SHA256')
        bundle = ROOT / 'colab_distillation_bundle.zip'
        self.assertEqual(hashlib.sha256(bundle.read_bytes()).hexdigest(), checksum)
        namespace = {}
        exec('import hashlib, zipfile\nfrom pathlib import Path\nfrom io import BytesIO\n'
             'from tempfile import TemporaryDirectory, mkdtemp', namespace)
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(notebook), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder) / 'Health-Insurance-SLM-Colab'
            project.mkdir()
            stale = project / 'stale.py'
            stale.write_text('obsolete')
            namespace['verified_project'](project, bundle, checksum)
            self.assertFalse(stale.exists())
            profile = json.loads((project / 'config/experiments/semantic-remediation-round1-v1.json').read_text())
            self.assertEqual(profile['overrides']['max_rounds'], 1)
            self.assertEqual(profile['overrides']['scenarios_per_round'], 200)
            self.assertEqual(profile['overrides']['adaptive_pairs_per_round'], 80)
            self.assertFalse(profile['overrides']['preference_training'])
            self.assertEqual(profile['overrides']['quality']['minimum_new_pair_fraction'], 1.0)
        call, = [node for tree in trees for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == 'run_loop']
        arguments = {keyword.arg: keyword.value for keyword in call.keywords}
        self.assertIs(ast.literal_eval(arguments['evaluate_release']), False)
        self.assertEqual(arguments['continuation_authorized'].id, 'CONTINUATION_AUTHORIZED')
        self.assertEqual(arguments['semantic_result'].id, 'SEMANTIC_RESULT')
