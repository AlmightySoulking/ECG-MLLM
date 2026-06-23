#!/usr/bin/env python3
"""Generate a summary table image from a judged predictions JSON.

Example:
  python scripts/generate_judged_table.py \
    --input outputs/tinygptv_stage3_mimic_multi_predictions_judged.json \
    --output outputs/tinygptv_stage3_summary_table.png

The script computes mean judge scores grouped by the number of ECGs
(length of the `ecg_paths` field) for counts 2..6 by default, plus an
overall mean and the count of highest scores (default max score = 5).
"""
from __future__ import annotations

import argparse
import json
import re
import math
from collections import defaultdict
from pathlib import Path
from typing import List

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise RuntimeError("matplotlib is required. Install with: pip install matplotlib") from e


def extract_score(s):
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    try:
        return float(s)
    except Exception:
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(s))
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None


def compute_metrics(samples: List[dict], counts: List[int], max_score: float, total_samples: int | None = None):
    groups = defaultdict(list)
    all_scores = []
    for samp in samples:
        ecg_paths = samp.get('ecg_paths') or []
        n = len(ecg_paths)
        score = samp.get('judge_score', None)
        if score is None:
            score = extract_score(samp.get('judge_response'))
        else:
            score = extract_score(score)
        if score is None:
            continue
        groups[n].append(score)
        all_scores.append(score)

    row_means = []
    for c in counts:
        vals = groups.get(c, [])
        if vals:
            row_means.append(sum(vals) / len(vals))
        else:
            row_means.append(float('nan'))

    all_mean = sum(all_scores) / len(all_scores) if all_scores else float('nan')
    highest_count = sum(1 for s in all_scores if math.isclose(s, max_score))
    total = total_samples if total_samples is not None else len(samples)

    return {
        'counts': counts,
        'means': row_means,
        'all_mean': all_mean,
        'highest_count': highest_count,
        'total_samples': total,
    }


def render_table(out_path: Path, metrics: dict, title: str = None, dpi: int = 200):
    counts = metrics['counts']
    means = metrics['means']
    all_mean = metrics['all_mean']
    highest = f"{metrics['highest_count']}/{metrics['total_samples']}"

    fmt = [f"{x:.2f}" if not (isinstance(x, float) and math.isnan(x)) else "-" for x in means]
    fmt_all = f"{all_mean:.2f}" if not math.isnan(all_mean) else "-"

    labels = [str(c) for c in counts] + ['All', 'Highest (#)']
    row = fmt + [fmt_all, highest]

    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.axis('off')

    the_table = ax.table(cellText=[row], colLabels=labels, rowLabels=[title or out_path.stem], loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1, 2)

    # simple cell styling
    for (r, c), cell in the_table.get_celld().items():
        cell.set_edgecolor('black')
        if r == 0:
            cell.set_text_props(weight='bold')
        if c == -1:
            cell.set_text_props(weight='bold')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def parse_counts(s: str) -> List[int]:
    if not s:
        return [2, 3, 4, 5, 6]
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return [int(x) for x in parts]


def main():
    parser = argparse.ArgumentParser(description='Generate summary table image from judged JSON')
    parser.add_argument('--input', '-i', required=True, help='Path to judged JSON file')
    parser.add_argument('--output', '-o', default='outputs/judged_summary_table.png', help='Output PNG path')
    parser.add_argument('--counts', '-c', default='2,3,4,5,6', help='Comma-separated ECG counts to include (default: 2,3,4,5,6)')
    parser.add_argument('--max-score', type=float, default=5.0, help='Value considered the "highest" score (default: 5.0)')
    parser.add_argument('--dpi', type=int, default=200, help='Image DPI')
    parser.add_argument('--json', dest='json', action='store_true', help='Write metrics as JSON to --output instead of PNG')
    args = parser.parse_args()

    p = Path(args.input)
    if not p.exists():
        raise SystemExit(f'Input file not found: {p}')

    obj = json.loads(p.read_text())
    samples = obj.get('samples', [])
    total_samples = obj.get('summary', {}).get('num_samples', None)

    counts = parse_counts(args.counts)
    metrics = compute_metrics(samples, counts, args.max_score, total_samples)

    out = Path(args.output)
    if args.json:
        # prepare JSON-serializable metrics (replace NaN with null)
        def none_if_nan(x):
            return None if (isinstance(x, float) and math.isnan(x)) else x

        metrics_serializable = {
            'counts': metrics['counts'],
            'means': [none_if_nan(m) for m in metrics['means']],
            'all_mean': none_if_nan(metrics['all_mean']),
            'highest_count': metrics['highest_count'],
            'total_samples': metrics['total_samples'],
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics_serializable, indent=2))
        print('WROTE', str(out))
        print(json.dumps(metrics_serializable, indent=2))
        return
    else:
        render_table(out, metrics, title=p.stem, dpi=args.dpi)

    # Print a short summary
    print('WROTE', str(out))
    for c, m in zip(metrics['counts'], metrics['means']):
        if math.isnan(m):
            print(f'{c}: -')
        else:
            print(f'{c}: {m:.4f}')
    if not math.isnan(metrics['all_mean']):
        print('All mean:', f"{metrics['all_mean']:.4f}")
    print('Highest:', f"{metrics['highest_count']}/{metrics['total_samples']}")


if __name__ == '__main__':
    main()
