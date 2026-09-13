"""Prepare a verified nested corpus or compare two agentic evaluation reports."""
import argparse
import hashlib
import io
import json
from pathlib import Path


def prepare_experiment(raw, output, baseline):
    import pandas as pd
    from data.recipes.prepare_recipes import clean_recipes
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    # Match the existing CSV's LF text across operating systems.
    with raw.open(encoding="utf-8", newline=None) as source:
        cleaned = clean_recipes(source)
    current = pd.read_csv(baseline)
    sampled = cleaned.sample(n=5000, random_state=42)
    reproduced = pd.read_csv(io.StringIO(sampled.to_csv(index=False)))
    pd.testing.assert_frame_equal(current, reproduced)
    larger = cleaned.sample(n=20000, random_state=42)
    assert set(current['id']).issubset(set(larger['id']))
    larger.to_csv(output, index=False)
    manifest = {
        'seed': 42, 'newline_mode': 'universal LF (matches existing 5k descriptions)', 'raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
        'baseline_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest(),
        'larger_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'baseline_reproduced': True, 'baseline_is_subset': True,
        'cleaned_count': len(cleaned), 'larger_count': len(larger),
    }
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def compare(small, large):
    if small['benchmarks_sha256'] != large['benchmarks_sha256']:
        raise ValueError('Benchmark definitions differ')
    rows = [r['cases']['agentic'] for r in [small, large]]
    if {r['id'] for r in rows[0]} != {r['id'] for r in rows[1]}:
        raise ValueError('Benchmark case sets differ')
    fields = ['average_raw_precision_at_5', 'average_filtered_precision_at_5',
              'queries_with_enough_relevant_candidates', 'low_dataset_coverage_queries',
              'average_usable_candidate_rate', 'hard_time_violations',
              'forbidden_ingredient_violations', 'invalid_source_ids']
    summary = {field: [report['aggregate']['agentic'][field] for report in [small, large]] for field in fields}
    summary['retrieval_failure_count'] = [sum(bool(r['retrieval_failure']) for r in group) for group in rows]
    comparison = []
    lookup = {r['id']: r for r in rows[1]}
    for first in rows[0]:
        second = lookup[first['id']]
        delta = second['filtered_precision_at_5'] - first['filtered_precision_at_5']
        comparison.append({
            'id': first['id'], 'change': 'improved' if delta > 0 else 'regressed' if delta < 0 else 'same',
            'filtered_precision_at_5': [first['filtered_precision_at_5'], second['filtered_precision_at_5']],
            'corpus_proxy_relevant_count': [first['corpus_proxy_relevant_count'], second['corpus_proxy_relevant_count']],
            'relevant_filtered_count': [first['relevant_filtered_count'], second['relevant_filtered_count']],
            'retrieval_failure': [first['retrieval_failure'], second['retrieval_failure']],
        })
    return {'columns': ['5k', '20k'], 'summary': summary, 'cases': comparison}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--raw', type=Path, required=True)
    prep.add_argument('--baseline', type=Path, default=Path('data/recipes/recipes_rag.csv'))
    prep.add_argument('--output', type=Path, default=Path('data/recipes/recipes_rag_20k.csv'))
    comp = sub.add_parser('compare')
    comp.add_argument('small', type=Path)
    comp.add_argument('large', type=Path)
    comp.add_argument('--output', type=Path, default=Path('/tmp/corpus-comparison.json'))
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare_experiment(args.raw, args.output, args.baseline)
    else:
        result = compare(json.loads(args.small.read_text()), json.loads(args.large.read_text()))
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
