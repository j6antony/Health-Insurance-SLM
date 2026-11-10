#!/usr/bin/env python3
"""Unified Self-Instruct CLI. No student weights or legacy GPT-3 dependencies."""
import argparse
import json
import sys

from self_instruct_pipeline.config import load_config
from self_instruct_pipeline.parent_backend import ParentError
from self_instruct_pipeline.pipeline import Pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['dry-run', 'generate', 'answer', 'export', 'all', 'resume'])
    parser.add_argument('--config', help='JSON config; relative data paths resolve from the workspace root')
    parser.add_argument('--output-dir', help='Override the output directory, useful for isolated smoke runs')
    parser.add_argument('--target', type=int, help='Verified-pair target for all/resume; scenario target for generate')
    parser.add_argument('--batch-size', type=int, help='Override batch size for this isolated run')
    parser.add_argument('--seed-file', help='Override the explicit-task seed JSONL')
    parser.add_argument('--num-seed-examples', type=int, help='6-8 diverse demonstrations per request')
    parser.add_argument('--strict-answers', action='store_true', help='Return an error if answers/verified-pair target remain incomplete; good pairs are still exported')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.output_dir:
            config['paths']['output_dir'] = args.output_dir
        if args.seed_file:
            config['paths']['seeds'] = args.seed_file
        if args.num_seed_examples is not None:
            if not 6 <= args.num_seed_examples <= 8:
                raise ValueError('Use 6-8 seed demonstrations')
            config['sampling']['num_seed_examples'] = args.num_seed_examples
        for argument, key in ((args.target, 'target_accepted_scenarios'), (args.batch_size, 'batch_size')):
            if argument is not None:
                if argument < 1:
                    raise ValueError('Target and batch size must be positive')
                config['generation'][key] = argument
        pipeline = Pipeline(config)
        if args.command == 'dry-run':
            print(json.dumps(pipeline.plan(), indent=2, ensure_ascii=False))
            return 0
        with pipeline.store.lock():
            # Reconstruct under lock, so concurrent invocations cannot race reads.
            pipeline = Pipeline(config)
            pipeline.initialize_manifest()
            if args.command in ('all', 'resume'):
                result = pipeline.build_pairs()
                if args.strict_answers and result['status'] != 'target_reached':
                    raise ValueError('Verified-pair target not reached; valid partial dataset was exported')
            elif args.command == 'generate':
                pipeline.generate()
            elif args.command == 'answer':
                try:
                    pipeline.answer(allow_partial=not args.strict_answers)
                finally:
                    pipeline.export()
            elif args.command == 'export':
                pipeline.export()
            pipeline.checkpoint()
        return 0
    except (ValueError, ParentError, OSError, KeyError) as exc:
        print(f'Self-Instruct stopped: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
