"""
Collect and summarise ablation study results into a table.

Usage:
    python tools/collect_ablation_results.py [--work-dir WORK_DIR] [--seed SEED]

Scans work_dirs/ablation_ssl/ for evaluation logs and produces:
  1. A console table of mIoU / per-class IoU
  2. A CSV file at work_dirs/ablation_ssl/ablation_results.csv

This script parses the JSON evaluation logs produced by mmseg.
"""

import argparse
import json
import os
import glob
import csv
from collections import OrderedDict


def parse_eval_log(log_path):
    """Extract the best mIoU entry from a JSON evaluation log."""
    best_miou = -1.0
    best_entry = None

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if 'mIoU' in entry:
                miou = entry['mIoU']
                if miou > best_miou:
                    best_miou = miou
                    best_entry = entry

    return best_entry


def find_eval_log(exp_dir):
    """Find the evaluation JSON log in an experiment directory."""
    # mmseg typically writes to <work_dir>/<timestamp>.log.json
    candidates = sorted(glob.glob(os.path.join(exp_dir, '*.log.json')))
    if candidates:
        return candidates[-1]  # latest log
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Collect ablation study results')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/ablation_ssl',
        help='Root directory containing ablation experiment folders')
    parser.add_argument(
        '--seed', default=0, type=int,
        help='Seed suffix used in folder names (default: 0)')
    parser.add_argument(
        '--output',
        default=None,
        help='Output CSV path (default: <work-dir>/ablation_results.csv)')
    args = parser.parse_args()

    work_dir = args.work_dir
    seed = args.seed
    output_csv = args.output or os.path.join(work_dir, 'ablation_results.csv')

    if not os.path.isdir(work_dir):
        print(f"Work directory not found: {work_dir}")
        return

    # Discover experiment directories
    exp_dirs = sorted([
        d for d in os.listdir(work_dir)
        if os.path.isdir(os.path.join(work_dir, d))
    ])

    if not exp_dirs:
        print(f"No experiment directories found in {work_dir}")
        return

    # Ablation group labels for display
    GROUP_LABELS = {
        'satellite_ssl': 'Baseline (Full Model)',
        'A1': 'w/o Boundary Loss',
        'A2': 'w/o Prototype Loss',
        'A3': 'w/o Pseudo-Label Warmup',
        'A4': 'w/o ClassMix',
        'A5': 'w/o Rare Class Sampling',
        'A6': 'Baseline Self-Training Only',
        'B1_boundary_lambda_01': 'boundary_λ = 0.1',
        'B1_boundary_lambda_03': 'boundary_λ = 0.3',
        'B1_boundary_lambda_07': 'boundary_λ = 0.7',
        'B1_boundary_lambda_10': 'boundary_λ = 1.0',
        'B2_proto_lambda_005': 'proto_λ = 0.05',
        'B2_proto_lambda_02': 'proto_λ = 0.2',
        'B2_proto_lambda_05': 'proto_λ = 0.5',
        'B3_max_groups_32': 'max_groups = 32',
        'B3_max_groups_64': 'max_groups = 64',
        'B3_max_groups_128': 'max_groups = 128',
        'B4_warmup_500': 'warmup = 500',
        'B4_warmup_2000': 'warmup = 2000',
        'B4_warmup_5000': 'warmup = 5000',
        'C1': 'Boundary: Sobel',
        'C2': 'Boundary: Laplacian',
        'C3': 'Boundary: Hybrid',
    }

    results = []
    class_names = None

    for exp_name in exp_dirs:
        exp_path = os.path.join(work_dir, exp_name)
        log_path = find_eval_log(exp_path)

        if log_path is None:
            print(f"  [SKIP] {exp_name}: no evaluation log found")
            continue

        best = parse_eval_log(log_path)
        if best is None:
            print(f"  [SKIP] {exp_name}: no mIoU entry in log")
            continue

        # Determine display label
        label = exp_name
        for key, display in GROUP_LABELS.items():
            if key in exp_name:
                label = display
                break

        row = OrderedDict()
        row['Experiment'] = label
        row['Config'] = exp_name
        row['mIoU'] = best.get('mIoU', -1)
        row['mAcc'] = best.get('mAcc', -1)

        # Per-class IoU (if available)
        if 'IoU' in best and isinstance(best['IoU'], dict):
            if class_names is None:
                class_names = list(best['IoU'].keys())
            for cls in class_names:
                row[f'IoU_{cls}'] = best['IoU'].get(cls, -1)

        results.append(row)

    if not results:
        print("No results collected. Have the experiments finished training?")
        return

    # Sort: baseline first, then A, B, C groups
    def sort_key(r):
        cfg = r['Config']
        if 'satellite_ssl' in cfg and 'ablation' not in cfg:
            return (0, cfg)
        return (1, cfg)

    results.sort(key=sort_key)

    # Print console table
    print("\n" + "=" * 80)
    print(" DAPCN-SSL Ablation Study Results")
    print("=" * 80)

    # Header
    header = f"{'Experiment':<35} {'mIoU':>6} {'mAcc':>6}"
    if class_names:
        for cls in class_names:
            header += f" {cls[:8]:>8}"
    print(header)
    print("-" * len(header))

    baseline_miou = None
    for r in results:
        if baseline_miou is None:
            baseline_miou = r['mIoU']

        delta = r['mIoU'] - baseline_miou if baseline_miou else 0
        delta_str = f"({delta:+.2f})" if r['Config'] != results[0]['Config'] else ""

        line = f"{r['Experiment']:<35} {r['mIoU']:>5.2f}% {r['mAcc']:>5.2f}%"
        if class_names:
            for cls in class_names:
                val = r.get(f'IoU_{cls}', -1)
                line += f" {val:>7.2f}%"
        if delta_str:
            line += f"  {delta_str}"
        print(line)

    print("=" * 80)

    # Write CSV
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_csv}")


if __name__ == '__main__':
    main()
