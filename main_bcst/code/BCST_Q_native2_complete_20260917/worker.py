"""Production BCST LP experiments using archived kernels and 1-RU quadratic phases."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback

MODULE_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('LPQAOA_OUTPUT', str(MODULE_ROOT / 'output')))
ENERGY = MODULE_ROOT.parent / 'BCST_stage2_energy_20260917'
sys.path.insert(0, str(ENERGY))
import run_final_depth_scan as scan
import run_experiment as r
import continue_experiment as continuation
import torch

def resource(method, depth):
    reference = 1265+3550*14 if method=='three_stage' else 1265
    phase = {'three_stage':15420, 'two_stage':15540, 'two_stage_merged':15480}[method]
    return dict(stage1_preparation=1265, reference_preparation=reference,
        stage2_Q_phase=120, phase_separators_per_layer=phase, selective_phase=900,
        LP_mixer_per_layer=2*reference+900, terminal_evaluation=15420,
        per_shot=reference+depth*(phase+2*reference+900)+15420,
        convention='Occupation-product phases: quadratic terms 1 RU; other nonconstant degrees k cost 4k-2 RU')

def objective_record(seed):
    data = MODULE_ROOT.parents[1] / 'data'
    for path in sorted((data / 'raw' / 'baselines').glob(f'{seed}_*.json')):
        record = json.loads(path.read_text())
        if record.get('status') == 'COMPLETE':
            return record
    for sub in ['test','refinement','validation']:
        for path in sorted((r.BASE/'results'/sub).glob(f'{seed}_*.json')):
            record=json.loads(path.read_text())
            if record.get('status')=='COMPLETE':
                return record
    raise FileNotFoundError(f'Existing objective and target for seed {seed}')

def run(method,seeds,depths):
    progress=ROOT/'progress'/f'{method}.json'
    ctx,initial_d,_=r.setup()
    if method=='three_stage':
        reference,stage2,_=continuation.load_reference(ctx,initial_d,'Q',14,progress,require_saved=True)
    else:
        reference,stage2=ctx.c12_state,None
    for seed in seeds:
        old=objective_record(seed)
        table=old['coefficient_table']
        ground=old['score']['ground_indices']
        d=r.c.diagonals(ctx,table,1024)
        initial=r.measure(ctx.screen,d,reference,ground)
        if method=='three_stage':
            phases,names=(d.o_tensor,),['O/2048']
        elif method=='two_stage':
            phases,names=(d.q_tensor,d.o_tensor),['Q_raw','O/2048']
        else:
            phases,names=(d.q_tensor+d.o_tensor,),['Q_raw+O/2048']
        for depth in depths:
            output=ROOT/'results'/f'{seed}_{method}_p{depth}.json'
            if output.exists():
                continue
            started=time.time()
            if depth==0:
                measurements,training=initial,None
            else:
                state,training=scan.optimize(ctx.screen,reference,phases,d.h_tensor,
                    scan.initialize(method,depth),depth,progress,f'{seed} {method} p={depth}')
                measurements=r.measure(ctx.screen,d,state,ground)
                del state
            probabilities=measurements['success_probability']
            repeats=[r.c.rts99(p) for p in probabilities]
            ledger=resource(method,depth)
            costs=[None if n is None else n*ledger['per_shot'] for n in repeats]
            record=dict(status='COMPLETE',seed=seed,method=method,final_phase='O',
                final_stage_depth=depth,stage1_depth=12,
                frozen_stage2_depth=14 if method=='three_stage' else None,
                phase_separators=names,loss='(O+1024(C+Q))/2048',
                restart_seeds=r.RESTARTS,reference_measurements=initial,
                coefficient_table=table,ground_indices=ground,
                frozen_stage2=stage2,training=training,measurements=measurements,
                variational_parameter_count=(3 if method=='two_stage' else 2)*depth,
                initialization_stream=95002 if method=='two_stage_merged' else (95003 if method=='two_stage' else 95001),
                probability_by_restart=probabilities,probability_geomean=r.gm(probabilities),
                RTS99_by_restart=repeats,RU=ledger,total_cost_by_restart=costs,
                total_cost_geomean=r.gm(costs),elapsed_sec=time.time()-started,
                completed_unix=time.time(),reused_existing_result=False)
            r.write(output,record)
            r.write(progress,dict(status='POINT_COMPLETE',seed=seed,method=method,depth=depth,
                probability=record['probability_geomean'],total_cost=record['total_cost_geomean'],updated_unix=time.time()))
            print('COMPLETE',seed,method,depth,record['probability_geomean'],record['total_cost_geomean'],flush=True)
    r.write(progress,dict(status='COMPLETE',method=method,seeds=seeds,depths=depths,updated_unix=time.time()))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',required=True,choices=['three_stage','two_stage','two_stage_merged'])
    parser.add_argument('--seeds',required=True,type=int,nargs='+')
    parser.add_argument('--depths',required=True,type=int,nargs='+')
    args=parser.parse_args()
    try:
        run(args.method,args.seeds,args.depths)
    except Exception:
        r.write(ROOT/'progress'/f'{args.method}.json',dict(status='FAILED',error=traceback.format_exc(),updated_unix=time.time()))
        raise
