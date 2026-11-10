"""Read-only readiness report. No model downloads, teacher calls, or GPU allocation."""
import argparse
import importlib.util
import json
from importlib.metadata import version, PackageNotFoundError

from self_instruct_pipeline.config import load_config, resolve_path
from .core import load_loop_config
from .benchmarks import load_benchmarks
from .quality import validate_quality_settings


def readiness(parent_config, config, check_runtime=True):
    schemas = json.loads(resolve_path(parent_config['paths']['schemas']).read_text())['tasks']
    issues = []
    result = {'evaluation_mode': config['evaluation']['mode'], 'task_count': len(schemas),
              'quality': validate_quality_settings(config, schemas), 'issues': issues}
    for split in ('validation', 'test'):
        if config['evaluation']['mode'] == 'reviewed':
            path = resolve_path(config['evaluation'][f'{split}_path'])
            if not path.is_file(): issues.append(f'Missing reviewed {split}: {path}')
    if not issues:
        try:
            validation, test = load_benchmarks(config, schemas)
            result['validation_examples'] = len(validation) if validation is not None else None
            result['test_examples'] = len(test)
        except (ValueError, KeyError, TypeError) as exc:
            issues.append(f'Benchmark contract: {exc}')
    if config['evaluation']['mode'] == 'synthetic_smoke':
        result['performance_claims_allowed'] = False
    if check_runtime:
        expected = {'transformers': '4.57.3', 'trl': '0.25.1', 'peft': '0.18.0', 'lighteval': '0.11.0'}
        for package, required in expected.items():
            try:
                installed = version(package)
                if installed != required: issues.append(f'{package}=={required} required; found {installed}')
            except PackageNotFoundError:
                issues.append(f'{package}=={required} not installed')
        if importlib.util.find_spec('torch') is None:
            issues.append('CUDA-enabled PyTorch is not installed')
        else:
            import torch
            if not torch.cuda.is_available(): issues.append('CUDA is unavailable')
            result['visible_cuda_devices'] = torch.cuda.device_count()
    result['ready'] = not issues
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/distillation_loop.json')
    parser.add_argument('--parent-config', default='config/self_instruct.json')
    parser.add_argument('--skip-runtime', action='store_true', help='Check data/configuration only')
    args = parser.parse_args()
    report = readiness(load_config(args.parent_config), load_loop_config(args.config), not args.skip_runtime)
    print(json.dumps(report, indent=2))
    return 0 if report['ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
