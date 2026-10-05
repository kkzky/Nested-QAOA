"""Shared BCST resource accounting in the occupation-product representation."""

MODEL_ID='bcst_occupation_products_two_body_1RU'

def monomial_RU(degree):
    if degree==0:
        return 0
    return 1 if degree==2 else 4*degree-2

def components(two_body_RU=1):
    return dict(initial=5,C=30,Q=30*2+60*two_body_RU,
        O=30*2+240*10+720*18,XY=75,selective_phase=900,
        shared_Q_O_unary=30*2)

def ledger(method,depth,p2=14,final_phase='O',two_body_RU=1):
    c=components(two_body_RU)
    p1=c['initial']+12*(c['C']+c['XY'])
    merged=c['O']+c['Q']-c['shared_Q_O_unary']
    if method=='lp_three_stage':
        ref=p1+p2*(c['Q']+2*p1+c['selective_phase'])
        phase=c['O'] if final_phase=='O' else merged
        mixer=2*ref+c['selective_phase']
    elif method in ('lp_two_stage_separate','lp_two_stage_combined','omit_feasibility'):
        ref=p1
        phase={'lp_two_stage_separate':c['Q']+c['O'],
               'lp_two_stage_combined':merged,'omit_feasibility':c['O']}[method]
        mixer=2*ref+c['selective_phase']
    elif method in ('block_xy_separate','block_xy_combined','warm_start_block_xy','uniform_projector'):
        ref=p1 if method=='warm_start_block_xy' else c['initial']
        phase=c['Q']+c['C']+c['O'] if method in ('block_xy_separate','uniform_projector') else merged
        mixer=2*ref+c['selective_phase'] if method=='uniform_projector' else c['XY']
    else:
        raise ValueError(method)
    return dict(model_id=MODEL_ID if two_body_RU==1 else f'occupation_products_quadratic_{two_body_RU}RU',
        reference_preparation=ref,phase_per_layer=phase,mixer_per_layer=mixer,
        terminal_evaluation=c['O'],depth=depth,
        per_shot=ref+depth*(phase+mixer)+c['O'])

def n18_RU(depth,two_body_RU=1):
    return 2622+depth*(2622+63*two_body_RU)
