#!/usr/bin/env python3
"""Export persistent Supabase API history to Excel-readable CSV. No model loads."""
import argparse
import json
from self_instruct_pipeline.usage_cloud import export_cloud_usage, sync_usage_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/api_usage_all_runs.csv')
    parser.add_argument('--run-id', help='Optional run UUID; omit to export all saved runs')
    parser.add_argument('--sync-dir', help='First upload existing local journals under this folder')
    args = parser.parse_args()
    if args.sync_dir:
        print(json.dumps(sync_usage_history(args.sync_dir)))
    print(json.dumps(export_cloud_usage(args.output, args.run_id)))


if __name__ == '__main__':
    main()
