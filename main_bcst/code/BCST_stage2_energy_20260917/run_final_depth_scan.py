"""Requested final-stage depth studies using frozen references and energy loss."""
from pathlib import Path
import argparse
import json
import math
import os
import time
import traceback

import numpy as np
import torch
import run_experiment as r
import continue_experiment as continuation

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'final_depth_scan_20260917'


def initialize(method, depth):
    if method == 'two_stage_merged':
        # Archived joint-phase family stream, using the same depth-prefix rule.
        gamma, beta = [], []
        for seed in r.RESTARTS:
            gamma.append(np.random.default_rng(seed+1_000_003*(2*95002)).uniform(-math.pi,math.pi,64)[:depth])
            beta.append(np.random.default_rng(seed+1_000_003*(2*95002+1)).uniform(-math.pi,math.pi,64)[:depth])
        return np.asarray(gamma), np.asarray(beta)
    family = r.c.old.COLLAPSE_SEPARATE if method == 'two_stage' else r.c.old.FULL_LP
    gamma, beta = r.c.old._initial_angle_arrays(family, 64)
    count = 2 if method == 'two_stage' else 1
    return gamma.reshape(4,count,64)[:,:,:depth].reshape(4,count*depth).copy(), beta[:,:depth].copy()


def optimize(screen, reference, phases, loss, arrays, depth, progress, label):
    gamma, beta = [torch.tensor(a,dtype=screen.real_dtype,device=screen.device,requires_grad=True) for a in arrays]
    optimizer = torch.optim.Adam((gamma,beta),lr=0.035)
    best_energy = torch.full((4,),math.inf,dtype=screen.real_dtype,device=screen.device)
    best_gamma,best_beta = torch.zeros_like(gamma),torch.zeros_like(beta)
    best_index = torch.zeros(4,dtype=torch.int64,device=screen.device)
    best_state = reference.detach().clone()
    trace=[]
    started=time.time()
    for i in range(1,2401):
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(i<2400):
            energies,states=r.c.kernels.history_circuit(screen,reference,phases,loss,gamma,beta,depth)
        with torch.no_grad():
            improved=energies.detach()<best_energy
            best_state[improved]=states.detach()[improved]
            best_gamma[improved]=gamma.detach()[improved]
            best_beta[improved]=beta.detach()[improved]
            best_index[improved]=i
            best_energy=torch.minimum(best_energy,energies.detach())
        if i==1 or i==2400 or i%200==0:
            row=dict(evaluation=i,energies=energies.detach().cpu().tolist(),
                best_energies=best_energy.cpu().tolist(),elapsed_sec=time.time()-started)
            trace.append(row)
            r.write(progress,dict(status='RUNNING',phase=label,process_id=os.getpid(),
                evaluations=2400,updated_unix=time.time(),**row))
            print(label,i,row['best_energies'],flush=True)
        if i<2400:
            energies.sum().backward()
            optimizer.step()
    return best_state.detach(),dict(evaluations_per_restart=2400,updates_per_restart=2399,
        selected_energy=best_energy.cpu().tolist(),best_evaluation=best_index.cpu().tolist(),
        gamma=best_gamma.detach().cpu().tolist(),beta=best_beta.detach().cpu().tolist(),
        trace=trace,elapsed_sec=time.time()-started)


def resource(method,phase_name,depth,reference_depth):
    final_phase=15420 if phase_name=='O' else 15480
    if method=='two_stage_merged':
        reference=1265
        # Q and O share the 30 unary supports: count the union once.
        phase_cost=15480
    elif method=='two_stage':
        reference=1265
        phase_cost=120+final_phase
    else:
        reference=1265+3550*reference_depth
        phase_cost=final_phase
    return dict(stage1_preparation=1265,reference_preparation=reference,
        phase_separators_per_layer=phase_cost,selective_phase=900,
        LP_mixer_per_layer=2*reference+900,terminal_evaluation=15420,
        per_shot=reference+depth*(phase_cost+2*reference+900)+15420)


def run(method,phase_name,depths,reference_depth):
    ctx,d,baseline=r.setup()
    key=f'{method}_{phase_name}'
    progress=OUT/'progress'/f'{key}.json'
    if method in ('two_stage','two_stage_merged'):
        reference=ctx.c12_state
        stage2_record=None
        if method=='two_stage_merged':
            if phase_name!='O':
                raise ValueError('Only the requested Q_raw+O/2048 merged phase is defined.')
            phase_names=['Q_raw+O/2048']
            phases=(d.q_tensor+d.o_tensor,)
        else:
            phase_names=['Q_raw','O/2048' if phase_name=='O' else '(O+1024(C+Q))/2048']
            phases=(d.q_tensor,d.o_tensor if phase_name=='O' else d.h_tensor)
    else:
        reference,stage2_record,_=continuation.load_reference(ctx,d,'Q',reference_depth,progress,require_saved=True)
        phase_names=['O/2048' if phase_name=='O' else '(O+1024(C+Q))/2048']
        phases=(d.o_tensor if phase_name=='O' else d.h_tensor,)
    initial_metrics=r.measure(ctx.screen,d,reference,baseline['score']['ground_indices'])
    for depth in depths:
        output=OUT/'results'/f'{key}_p{depth}.json'
        if output.exists():
            continue
        started=time.time()
        state,training=optimize(ctx.screen,reference,phases,d.h_tensor,
            initialize(method,depth),depth,progress,f'{key} p={depth}')
        measurements=r.measure(ctx.screen,d,state,baseline['score']['ground_indices'])
        probabilities=measurements['success_probability']
        repetitions=[r.c.rts99(p) for p in probabilities]
        ledger=resource(method,phase_name,depth,reference_depth)
        costs=[None if n is None else n*ledger['per_shot'] for n in repetitions]
        probability,cost=r.gm(probabilities),r.gm(costs)
        result=dict(status='COMPLETE',seed=r.SEED,method=method,final_phase=phase_name,
            final_stage_depth=depth,stage1_depth=12,
            frozen_stage2_depth=reference_depth if method=='three_stage' else None,
            phase_separators=phase_names,loss='(O+1024(C+Q))/2048',
            restart_seeds=r.RESTARTS,reference_measurements=initial_metrics,
            variational_parameter_count=(3 if method=='two_stage' else 2)*depth,
            initialization_stream=95002 if method=='two_stage_merged' else (95003 if method=='two_stage' else 95001),
            frozen_stage2=stage2_record,training=training,measurements=measurements,
            probability_by_restart=probabilities,probability_geomean=probability,
            RTS99_by_restart=repetitions,RU=ledger,total_cost_by_restart=costs,
            total_cost_geomean=cost,elapsed_sec=time.time()-started)
        r.write(output,result)
        r.write(progress,dict(status='POINT_COMPLETE',method=method,final_phase=phase_name,
            final_stage_depth=depth,success_probability=probability,total_cost=cost,updated_unix=time.time()))
        print('COMPLETE',key,depth,probability,cost,flush=True)
        del state
    r.write(progress,dict(status='COMPLETE',method=method,final_phase=phase_name,
        completed_depths=depths,updated_unix=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['two_stage','three_stage','two_stage_merged'],required=True)
    parser.add_argument('--phase',choices=['O','H'],required=True)
    parser.add_argument('--depths',type=int,nargs='+',required=True)
    parser.add_argument('--reference-depth',type=int,default=14)
    args=parser.parse_args()
    try:
        run(args.method,args.phase,args.depths,args.reference_depth)
    except Exception:
        r.write(OUT/'progress'/f'{args.method}_{args.phase}.json',
            dict(status='FAILED',error=traceback.format_exc(),updated_unix=time.time()))
        raise
