"""Run an appendix BCST job using repository-local inputs and the saved settings."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import math
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--stage1-state', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'reviewer_workspace' / 'corrected_bcst_campaign_20260730_v2' / 'source_v2'))
    sys.path.insert(0, str(ROOT / 'reviewer_workspace' / 'reviewer_appendix_campaign_20260731' / 'source'))
    import numpy as np
    from appendix_v3 import adam_protocol, runner
    from bcst_v2.instance_core import build_training_instance, derive_targets

    payload = json.loads(args.job.read_text(encoding='utf-8'))
    record = payload.get('manifest', payload)
    # Keep the saved initializer domain; deployment/file seals are not numerical inputs.
    adam_protocol.PLAN_SHA256 = record['plan_sha256'].lower()
    manifest = SimpleNamespace(**record)
    manifest.resources = runner.LogicalResources(**record['resources'])
    runtime = runner._build_target_blind_runtime(manifest, device=args.device)
    phi = None
    if args.stage1_state:
        phi = np.load(args.stage1_state, allow_pickle=False)
    if manifest.method in runner.PHI_INITIAL_METHODS and phi is None:
        parser.error('--stage1-state is required for this method')
    if manifest.method == 'stage1_only':
        state = phi
        optimization = None
    else:
        optimization, state, _, _ = runner._run_adam_cell(
            manifest, runtime.dynamics, phi=phi, device=args.device,
            activation_checkpointing=payload.get('activation_checkpointing', True))
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / 'selected_state.npy', state, allow_pickle=False)
    result = {'manifest': record, 'optimization': optimization}
    if manifest.cell_type != 'stage1':
        instance = build_training_instance(manifest.N, manifest.problem_seed, 1, 2, 1, 16)
        targets = derive_targets(instance)
        probability = float(np.abs(state[targets.ground]) @ np.abs(state[targets.ground]))
        resources = runner._resource_record(manifest.resources, manifest.method, manifest.depth)
        result.update(success_probability=probability, resources=resources)
        repetitions = None if probability <= 0 else (1 if probability >= 1 else math.ceil(math.log(0.01) / math.log1p(-probability)))
        result.update(RTS99=repetitions, total_cost_RU=None if repetitions is None else repetitions * resources['terminal_RU'])
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
