"""Deterministic metadata-proxy evaluation; see RAG_EVALUATION.md."""
import argparse
import ast
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean

from workflow import (
    deterministic_retrieval_filter_node,
    recipe_retriever_node,
    retrieval_query_node,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'data/recipes/recipes_rag.csv'
BENCHMARKS = Path(__file__).with_name('rag_benchmarks.json')
CUISINE_TAGS = {
    'korean', 'chinese', 'italian', 'ethiopian', 'japanese', 'thai', 'indian',
    'mexican', 'greek', 'french', 'spanish', 'vietnamese', 'lebanese', 'moroccan',
    'german', 'irish', 'british', 'american', 'canadian', 'cajun', 'caribbean',
}


def load_metadata():
    with DATASET.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['minutes'] = int(row['minutes'])
        for field in ('tags', 'ingredients'):
            row[field] = ast.literal_eval(row[field])
    return {row['id']: row for row in rows}

# just transform the text from "boneless chicken breats" to ["boneless", "chicken", "breasts"]"
def tokens(text):
    return re.findall(r'[a-z0-9]+', text.lower())

# checks if the text matches with the term
def matches(text, term):
    # Independent audit of literal phrases with simple singular/plural variants.
    def singular(word):
        return word[:-1] if len(word) > 3 and word.endswith('s') else word
    haystack = [singular(t) for t in tokens(text)]
    needle = [singular(t) for t in tokens(term)]
    return bool(needle) and any(haystack[i:i + len(needle)] == needle
                               for i in range(len(haystack) - len(needle) + 1))

# checks if a forbidden ingredient appear
def forbidden(row, case):
    return any(matches(ingredient, term) for ingredient in row['ingredients']
               for term in case.get('forbidden_terms', []))


def relevant(row, case):
    """AND between groups; OR within a group. Hard constraints also apply."""
    if not row or not 0 < row['minutes'] <= case.get('max_minutes', 180):
        return False
    if forbidden(row, case):
        return False
    if any(not set(group).intersection(row['tags'])
           for group in case.get('expected_tag_groups', [])):
        return False
    if any(not any(matches(ingredient, term) for ingredient in row['ingredients']
                   for term in group)
           for group in case.get('expected_ingredient_groups', [])):
        return False
    text = ' '.join([row['name'], row['description'], *row['tags'], *row['ingredients']])
    return all(matches(text, term) for term in case.get('expected_terms', []))


def precision_at_5(recipes, metadata, case):
    if not any(case.get(key) for key in
               ('expected_terms', 'expected_tag_groups', 'expected_ingredient_groups')):
        return None
    return sum(relevant(metadata.get(str(r.get('recipe_id'))), case)
               for r in recipes[:5]) / 5


def name_diversity(recipes):
    names = [' '.join(tokens(r['name'])) for r in recipes]
    duplicates = len(names) - len(set(names))
    # Fraction of results near an earlier name, not fraction of all name pairs.
    near = sum(any(name != prior and SequenceMatcher(None, name, prior).ratio() >= .9
                   for prior in names[:i]) for i, name in enumerate(names))
    return {'duplicate_name_rate': duplicates / len(names) if names else 0,
            'near_duplicate_name_rate': near / len(names) if names else 0}


def cuisine_diversity(recipes, metadata):
    counts = Counter()
    tagged = 0
    for recipe in recipes:
        row = metadata.get(str(recipe.get('recipe_id')), {})
        tags = CUISINE_TAGS.intersection(row.get('tags', []))
        tagged += bool(tags)
        counts.update(tags)
    return {'distinct_cuisine_tags': len(counts), 'cuisine_tag_counts': dict(counts),
            'cuisine_tag_coverage': tagged / len(recipes) if recipes else 0}


def evaluate_case(case, metadata):
    minimum = case.get('minimum_valid_candidates', 7)
    constraints = dict(case.get('planning_constraints', {}))
    constraints.update(max_cooking_time=case.get('max_minutes', 180),
                       avoid=case.get('forbidden_terms', []), meals_needed=minimum)
    state = {'planning_constraints': constraints}
    query = (retrieval_query_node(state)['retrieval_query']
             if 'planning_constraints' in case else case['query'])
    state['retrieval_query'] = query
    state.update(recipe_retriever_node(state))
    retrieved = state['retrieved_recipes']
    filter_error = None
    try:
        filtered = deterministic_retrieval_filter_node(state)['filtered_recipes']
    except ValueError as exc:
        filter_error = str(exc)
        # Observe survivors of the same production filter without its pool-size gate.
        diagnostic = {**state, 'planning_constraints': {**constraints, 'meals_needed': 0}}
        filtered = deterministic_retrieval_filter_node(diagnostic)['filtered_recipes']
    missing = sum(not r.get('recipe_id') for r in retrieved)
    invalid = sum(bool(r.get('recipe_id')) and str(r['recipe_id']) not in metadata
                  for r in retrieved)
    mismatches = sum(r['name'] != metadata[str(r['recipe_id'])]['name'] or
                     r['minutes'] != metadata[str(r['recipe_id'])]['minutes'] or
                     r['ingredients'] != metadata[str(r['recipe_id'])]['ingredients']
                     for r in retrieved if str(r.get('recipe_id')) in metadata)
    corpus_count = sum(relevant(row, case) for row in metadata.values())
    result = {
        'id': case['id'], 'category': case['category'], 'scenario': case['query'],
        'executed_query': query,
        'query_mode': 'production_query_builder' if 'planning_constraints' in case else 'literal',
        'retrieved_count': len(retrieved), 'filtered_count': len(filtered),
        'usable_candidate_rate': len(filtered) / len(retrieved) if retrieved else 0,
        'max_minutes': case.get('max_minutes'),
        'time_violations': sum(not 0 < r['minutes'] <= constraints['max_cooking_time'] for r in filtered),
        'forbidden_ingredient_violations': sum(forbidden(r, case) for r in filtered),
        'minimum_valid_candidates': minimum, 'enough_candidates': len(filtered) >= minimum,
        'preferred_adapter_pool_size': max(10, minimum),
        'enough_for_preferred_pool': len(filtered) >= max(10, minimum),
        'raw_precision_at_5': precision_at_5(retrieved, metadata, case),
        'filtered_precision_at_5': precision_at_5(filtered, metadata, case),
        'relevant_filtered_count': sum(relevant(metadata.get(str(r['recipe_id'])), case) for r in filtered),
        'corpus_proxy_relevant_count': corpus_count,
        'low_dataset_coverage': corpus_count < minimum,
        'missing_source_ids': missing, 'invalid_source_ids': invalid,
        'source_metadata_mismatches': mismatches,
        'adapter_source_recipe_id_check': 'not run; retrieval recipe_id checked against CSV',
        'filter_error': filter_error,
        'raw_name_diversity': name_diversity(retrieved),
        'filtered_name_diversity': name_diversity(filtered),
        'filtered_cuisine_diversity': cuisine_diversity(filtered, metadata),
    }
    for label, recipes in [('raw_top_5', retrieved[:5]), ('filtered_top_5', filtered[:5])]:
        result[label] = [
            {'recipe_id': r['recipe_id'], 'name': r['name'], 'minutes': r['minutes'],
             'relevant': relevant(metadata.get(str(r['recipe_id'])), case),
             'tags': metadata.get(str(r['recipe_id']), {}).get('tags', [])}
            for r in recipes
        ]
    return result


def aggregate(results):
    scored = [r for r in results if r['filtered_precision_at_5'] is not None]
    return {
        'queries_tested': len(results), 'queries_scored': len(scored),
        'average_raw_precision_at_5': mean(r['raw_precision_at_5'] for r in scored) if scored else None,
        'average_filtered_precision_at_5': mean(r['filtered_precision_at_5'] for r in scored) if scored else None,
        'average_usable_candidate_rate': mean(r['usable_candidate_rate'] for r in results),
        'queries_with_enough_candidates': sum(r['enough_candidates'] for r in results),
        'queries_with_preferred_pool': sum(r['enough_for_preferred_pool'] for r in results),
        'queries_with_enough_relevant_candidates': sum(r['relevant_filtered_count'] >= r['minimum_valid_candidates'] for r in results),
        'hard_time_violations': sum(r['time_violations'] for r in results),
        'forbidden_ingredient_violations': sum(r['forbidden_ingredient_violations'] for r in results),
        'missing_source_ids': sum(r['missing_source_ids'] for r in results),
        'invalid_source_ids': sum(r['invalid_source_ids'] for r in results),
        'source_metadata_mismatches': sum(r['source_metadata_mismatches'] for r in results),
        'low_dataset_coverage_queries': sum(r['low_dataset_coverage'] for r in results),
        'filter_pool_errors': sum(r['filter_error'] is not None for r in results),
        'average_raw_duplicate_name_rate': mean(r['raw_name_diversity']['duplicate_name_rate'] for r in results),
        'average_raw_near_duplicate_name_rate': mean(r['raw_name_diversity']['near_duplicate_name_rate'] for r in results),
        'average_filtered_duplicate_name_rate': mean(r['filtered_name_diversity']['duplicate_name_rate'] for r in results),
        'average_filtered_near_duplicate_name_rate': mean(r['filtered_name_diversity']['near_duplicate_name_rate'] for r in results),
        'average_filtered_distinct_cuisine_tags': mean(r['filtered_cuisine_diversity']['distinct_cuisine_tags'] for r in results),
        'average_filtered_cuisine_tag_coverage': mean(r['filtered_cuisine_diversity']['cuisine_tag_coverage'] for r in results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmarks', type=Path, default=BENCHMARKS)
    parser.add_argument('--output', type=Path, default=Path('/tmp/rag-evaluation.json'))
    args = parser.parse_args()
    cases = json.loads(args.benchmarks.read_text())
    metadata = load_metadata()
    results = []
    for case in cases:
        result = evaluate_case(case, metadata)
        results.append(result)
        print(f"\n{result['id']}: {result['executed_query']}", flush=True)
        print(f"  Retrieved {result['retrieved_count']} -> filtered {result['filtered_count']}; "
              f"usable {result['usable_candidate_rate']:.1%}; "
              f"P@5 raw/filtered {result['raw_precision_at_5']}/{result['filtered_precision_at_5']}; "
              f"enough={result['enough_candidates']}; corpus matches={result['corpus_proxy_relevant_count']}")
        print(f"  Time/exclusion violations={result['time_violations']}/{result['forbidden_ingredient_violations']}; "
              f"missing/invalid IDs={result['missing_source_ids']}/{result['invalid_source_ids']}; "
              f"duplicates={result['filtered_name_diversity']}; cuisine diversity={result['filtered_cuisine_diversity']}")
        print('  Top filtered:', '; '.join(f"{r['name']} [{'match' if r['relevant'] else 'no match'}]"
                                         for r in result['filtered_top_5']))
        if result['filter_error']:
            print('  Production pool gate:', result['filter_error'])
    summary = aggregate(results)
    worst = sorted((r for r in results if r['filtered_precision_at_5'] is not None),
                   key=lambda r: (r['filtered_precision_at_5'], r['relevant_filtered_count'], r['id']))[:5]
    report = {'created_at': datetime.now(timezone.utc).isoformat(),
              'dataset_sha256': hashlib.sha256(DATASET.read_bytes()).hexdigest(),
              'benchmarks_sha256': hashlib.sha256(args.benchmarks.read_bytes()).hexdigest(),
              'metric': 'strict metadata relevance proxy; not human labels',
              'benchmarks': cases, 'aggregate': summary, 'cases': results,
              'worst_queries': [r['id'] for r in worst]}
    args.output.write_text(json.dumps(report, indent=2))
    print('\nRAG Evaluation\n-------------------------')
    for key, value in summary.items():
        print(f'{key}: {value}')
    print('Worst queries:', ', '.join(report['worst_queries']))
    print('JSON report:', args.output)


if __name__ == '__main__':
    main()
