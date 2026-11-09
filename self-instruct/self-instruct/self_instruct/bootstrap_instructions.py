"""Compatibility entry point for the active structured, balanced Self-Instruct CLI.

The numbered GPT-3 bootstrapper has been superseded. All implementation lives
in self_instruct_pipeline at the workspace root; this file does not call davinci.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_self_instruct import main


def translate_arguments(argv):
    aliases = {'--batch_dir': '--output-dir', '--num_instructions_to_generate': '--target',
               '--seed_tasks_path': '--seed-file', '--request_batch_size': '--batch-size',
               '--num_prompt_instructions': '--num-seed-examples'}
    command = 'dry-run' if '--dry_run' in argv else 'generate'
    args = [aliases.get(arg, arg) for arg in argv if arg != '--dry_run']
    return [command, *args]


if __name__ == '__main__':
    raise SystemExit(main(translate_arguments(sys.argv[1:])))
