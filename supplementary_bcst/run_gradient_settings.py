"""Evaluate the saved BCST gradient settings without resampling angles."""
from pathlib import Path
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seeds', type=int, nargs='+')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / 'bcst_unique_primary_campaign_20260809'
    sys.path.insert(0, str(root / 'source'))
    from bcst_unique_campaign import supplements as numerical

    inputs = json.loads((root / 'gradient_inputs' / 'FROZEN_ANGLE_INPUT_MANIFEST.json').read_text())
    saved = json.loads((root.parent / 'gradient_results' / 'UNIQUE_FIXED_ANGLE_PANEL.json').read_text())
    sample = saved['settings'][0]
    module, core, objective, paper_root = numerical._load_historical_modules(root)
    rows = []
    for seed in sorted({int(row['problem_seed']) for row in inputs['settings']}):
        if args.seeds and seed not in args.seeds:
            continue
        context = numerical._build_gradient_context(
            module=module, core=core, objective=objective, paper_root=paper_root,
            seed=seed, device=args.device,
            target_binding=inputs['exact_target_identity']['seeds'][str(seed)])
        for item in inputs['settings']:
            if int(item['problem_seed']) != seed:
                continue
            gamma, beta = numerical._angle_arrays(item)
            source = json.loads((root / item['source_row_relative_path']).read_text())
            rows.append(numerical._evaluate_unique_setting(
                module=module, context=context, input_row=item,
                gamma_values=gamma, beta_values=beta, source_row=source,
                plan_sha256=sample['plan_sha256'],
                input_manifest_sha256=sample['input_manifest_sha256']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'settings': rows}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
