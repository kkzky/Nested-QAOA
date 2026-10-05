# Paper experiments, figures and tables

Paths are relative to the repository root.

## Environment and running the experiments

Reference environment: Python 3.12.7, NumPy 2.1.2 and PyTorch 2.7.0
with CUDA 11.8. The recorded GPU was an NVIDIA A800 with 80 GB memory.
Matplotlib is also imported by the numerical modules; its compatible installation
range is listed in `requirements.txt`, rather than an unrecorded original version.
The state-vector experiments require a CUDA-capable GPU with sufficient memory;
the largest jobs use tens of GB. Run the commands below from the repository root.
They run the full numerical jobs, not reduced demonstration examples.

```sh
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install numpy==2.1.2 "matplotlib>=3.8,<4"
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

### Main BCST

The runners read the frozen preparations in `main_bcst/data/inputs/`, the
coefficient tables in `main_bcst/data/raw/baselines/`, and the input records in
`main_bcst/data/runtime_inputs/`. Realization seeds are listed in
`main_bcst/data/current/per_realization.json`; final-stage depths and optimizer
settings are stored in the corresponding result records.

```sh
python main_bcst/code/BCST_Q_native2_complete_20260917/worker.py --method three_stage --seeds 2026091100 --depths 11
python main_bcst/code/BCST_Q_native2_complete_20260917/worker.py --method two_stage --seeds 2026091100 --depths 40
python main_bcst/code/BCST_Q_native2_complete_20260917/worker.py --method two_stage_merged --seeds 2026091100 --depths 12
python main_bcst/code/BCST_full_depth_20260913/full_depth_worker.py --seed 2026091100 --method xy_separate --depth 16
python main_bcst/run_n18.py --table-seed 2026081804 --depth 1 --output runs/n18_seed2026081804_p1.json
```

LP output defaults to `main_bcst/code/BCST_Q_native2_complete_20260917/output/`;
set `LPQAOA_OUTPUT` to select another location. The baseline runner writes to
`main_bcst/code/BCST_full_depth_20260913/results/validation/`.
Other baseline method keys are `xy_combined`, `no_feasibility`, `warm_xy` and
`uniform_projector`. Pass the desired seeds and depths to run additional cells.
The current occupation-product RU rules are implemented in
`main_bcst/code/resource_accounting.py`; saved costs use these same rules.

### Additional BCST and gradients

Saved Adam job specifications contain the instance, method, depth, initializer
and training budget. `supplementary_bcst/run_saved_job.py` accepts a job JSON and,
for methods requiring a learned reference, the corresponding saved stage-1
`selected_state.npy`. Its output directory must not already exist.

```sh
python supplementary_bcst/run_saved_job.py --job supplementary_bcst/inputs/reviewer_jobs/learned_projector_N25_seed840025102_p16_b800.json --stage1-state supplementary_bcst/raw/reviewer/stage1_budget_400_800_v3/cells/stage1_N25_seed840025000_b400/attempt-0001/artifacts/selected_state.npy --output runs/appendix_N25
python supplementary_bcst/run_gradient_settings.py --output runs/gradients.json
python supplementary_bcst/historical_spsa/source/campaign_runner.py scan --help
```

For the other Adam cells, select their saved job specifications from the input
and raw-data directories indexed below. SPSA configurations, initial angles and
stage-1 checkpoints are in `supplementary_bcst/historical_spsa/inputs/` and
`supplementary_bcst/historical_spsa/raw/`; pass the stored instance, schedule,
depths, optimizer settings and checkpoint to the `scan` subcommand.
Gradient inputs are the saved angles in `FROZEN_ANGLE_INPUT_MANIFEST.json`, with
the associated instance and stage-1 records under `gradient_inputs/`.
The gradient command evaluates those settings without optimizing or resampling.

### SBM

The configuration files list every argument used for each graph. Select a
zero-based task index and a new output file. This preserves the saved graph,
depths, restart count and optimizer configuration.

```sh
python sbm/run_saved_task.py --config sbm/config/main_runtime_tasks.json --index 0 --output runs/sbm_seed100.json
python sbm/run_saved_task.py --config sbm/config/p8_runtime_tasks.json --index 0 --output runs/sbm_seed100_p8.json
```

## Numerical figure data

| Figure | CSV files in `source_data/` |
| --- | --- |
| 2 | `figure_2_realizations.csv`; `figure_2_initializations.csv`; `figure_2_curve_summary.csv` |
| 3 | `figure_3_realizations.csv`; `figure_3_initializations.csv`; `figure_3_curve_summary.csv` |
| 4 | `figure_4_gradients.csv` |
| 5 | `figure_5_sbm.csv` |
| 7 | `figure_7_bcst_validation.csv` |
| 8 | `figure_8_bcst.csv` |
| 9 | `figure_9_bcst.csv` |

## License

Code and accompanying data are distributed under the MIT license in `LICENSE`.

## Experiment index

## Multilevel BCST — Results 2.3 and Appendix G

| Paper item | Code | Data |
| --- | --- | --- |
| Benchmark construction; Appendix G.1 | [Objective and simulation modules](main_bcst/code/BCST_corrected_loss_20260911/original_source/) | `main_bcst/data/inputs/original_manifest.json` |
| Three-stage and two-stage LP-QAOA; Fig. 2; Appendix G.2, G.4; Tables 1, 3, 4 | [Final-stage runner](main_bcst/code/BCST_Q_native2_complete_20260917/worker.py); [stage-2 and final-stage modules](main_bcst/code/BCST_stage2_energy_20260917/); [shared simulation modules](main_bcst/code/BCST_corrected_loss_20260911/) | `main_bcst/data/inputs/Q_p2_14.json`; `main_bcst/data/raw/current_campaign/results/`; `main_bcst/data/current/per_realization.json`; `main_bcst/data/current/per_initialization.json` |
| Block-XY QAOA baselines; Fig. 2; Appendix G.2, G.4; Tables 1, 3 | [Numerical runner](main_bcst/code/BCST_corrected_loss_20260911/campaign.py); [additional-depth runner](main_bcst/code/BCST_full_depth_20260913/full_depth_worker.py) | `main_bcst/data/raw/baselines/`; `main_bcst/data/current/per_realization.json`; `main_bcst/data/current/per_initialization.json` |
| Omitted joint-feasibility stage, warm-start block-XY and uniform-projector ablations; Fig. 3; Appendix G.2, G.4; Tables 1, 3 | [Numerical runner and circuit modules](main_bcst/code/BCST_corrected_loss_20260911/) | `main_bcst/data/raw/baselines/`; `main_bcst/data/current/per_realization.json`; `main_bcst/data/current/per_initialization.json` |
| Circuit and optimization inputs; Appendix G.2; Table 1 | [Circuit and optimization modules](main_bcst/code/) | `main_bcst/data/inputs/`; `main_bcst/data/runtime_inputs/` |
| Resource expressions and cost evaluation; Appendix F.2, G.3; Table 2 | [Resource accounting](main_bcst/code/resource_accounting.py) | `main_bcst/data/resource_model.json`; `main_bcst/data/current/` |
| Standard QAOA diagnostic at N=18; Appendix G.5 | [QAOA simulation](main_bcst/code/BCST_corrected_loss_20260911/original_source/targeted_coherent_im3_vanilla_qaoa.py); [probability evaluation](main_bcst/code/BCST_corrected_loss_20260911/original_source/targeted_coherent_im3_vanilla_qaoa_score.py); [resource accounting](main_bcst/code/resource_accounting.py) | `main_bcst/data/n18/inputs/`; `main_bcst/data/n18/results/`; `main_bcst/data/n18/vanilla_curve_cells.csv`; `main_bcst/data/n18/corrected_numerical_metrics.json` |

The main BCST metric files identify each record by realization seed, method and depth. Method keys are `lp_three_stage`, `lp_two_stage_separate`, `lp_two_stage_combined`, `block_xy_separate`, `block_xy_combined`, `omit_feasibility`, `warm_start_block_xy` and `uniform_projector`.

## Additional two-level BCST — Appendix H

| Paper item | Code | Data |
| --- | --- | --- |
| Hybrid degree-four/degree-six benchmark; Appendix H.1 | [Instance construction](supplementary_bcst/reviewer_workspace/corrected_bcst_campaign_20260730_v2/source_v2/bcst_v2/instance_core.py); [historical objective modules](supplementary_bcst/historical_spsa/source/source/) | `supplementary_bcst/results/adam400_stage1_N25.json`; `supplementary_bcst/raw/reviewer/stage1_budget_400_800_v3/` |
| SPSA robustness, depth and target-set sensitivity; Appendix H.2; Table 5 | [Numerical runners and modules](supplementary_bcst/historical_spsa/source/) | `supplementary_bcst/historical_spsa/inputs/`; `supplementary_bcst/historical_spsa/raw/`; `supplementary_bcst/historical_spsa/results/` |
| Fixed-budget Adam comparison at N=25; Appendix H.3; Tables 5, 6 | [Production runner](supplementary_bcst/reviewer_workspace/corrected_bcst_campaign_20260730_v2/source_v2/bcst_v2/production_runner.py); [simulation and optimization modules](supplementary_bcst/reviewer_workspace/corrected_bcst_campaign_20260730_v2/source_v2/bcst_v2/) | `supplementary_bcst/reviewer_workspace/corrected_bcst_campaign_20260730_v2/LEAN_EXECUTION_PLAN_20260730.json`; `supplementary_bcst/results/adam400_stage1_N25.json`; `supplementary_bcst/raw/adam400/`; `supplementary_bcst/results/adam400_N25_LP_WarmXY.json` |
| Matched comparisons: first five realizations per size, including target-set sensitivity; Appendix H.4; Tables 5, 7, 8 | [Numerical runner and evaluation modules](supplementary_bcst/reviewer_workspace/reviewer_appendix_campaign_20260731/source/appendix_v3/) | `supplementary_bcst/inputs/reviewer_jobs/`; `supplementary_bcst/raw/reviewer/five_seed_matched_validation_combined_execution_50_v3/`; `supplementary_bcst/results/matched_first_five_instance_results.json`; `supplementary_bcst/figure_data/matched_method_cells.csv`; `supplementary_bcst/figure_data/matched_targets_long.csv` |
| Matched comparisons: remaining fifteen realizations per size; Appendix H.4; Table 5 | [Numerical entry point](supplementary_bcst/bcst_unique_strengthened_validation_20260810/source/bcst_unique_campaign/cell_entry.py); [simulation modules](supplementary_bcst/bcst_unique_strengthened_validation_20260810/source/) | `supplementary_bcst/bcst_unique_strengthened_validation_20260810/VALIDATION_PLAN_V1.json`; `supplementary_bcst/bcst_unique_strengthened_validation_20260810/run/preflight/STAGE1_IMPORT_CERTIFICATE.json`; `supplementary_bcst/raw/validation_second_fifteen/`; `supplementary_bcst/results/validation_second_fifteen.json` |
| Combined twenty-realization comparison; Fig. 7; Appendix H.4 | Code for the two matched-comparison sets above | `supplementary_bcst/figure_data/bcst_unique_optimum_validation_20_instances.csv`; `supplementary_bcst/figure_data/bcst_unique_optimum_validation_5_plus_15.csv` |
| Initial-state projector control; Appendix H.4, H.5; Tables 5, 7, 8 | [Numerical runner and evaluation modules](supplementary_bcst/reviewer_workspace/reviewer_appendix_campaign_20260731/source/appendix_v3/) | `supplementary_bcst/inputs/reviewer_jobs/`; `supplementary_bcst/raw/reviewer/external_validation_native_grover_initial_v3/`; `supplementary_bcst/figure_data/matched_method_cells.csv`; `supplementary_bcst/figure_data/matched_targets_long.csv` |
| Depth and resource analysis at N=30; Fig. 8; Appendix H.5; Tables 5, 9 | [Numerical runner and evaluation modules](supplementary_bcst/reviewer_workspace/reviewer_appendix_campaign_20260731/source/appendix_v3/) | `supplementary_bcst/raw/reviewer/depth_reselection_base_b800_v3/`; `supplementary_bcst/raw/reviewer/depth_reselection_first_extension_b800_v3/`; `supplementary_bcst/raw/reviewer/depth_reselection_second_extension_b800_v3/`; `supplementary_bcst/raw/reviewer/depth_reselection_capped_112_128_b800_v3/`; `supplementary_bcst/results/depth/`; `supplementary_bcst/figure_data/bcst_unique_optimum_n30_depth_scan.csv`; `supplementary_bcst/figure_data/n30_lp_depth_cells.csv` |
| Mixer, reference-state and objective ablations; Fig. 9; Appendix H.6; Tables 5, 10, 11 | [Numerical runner and evaluation modules](supplementary_bcst/reviewer_workspace/reviewer_appendix_campaign_20260731/source/appendix_v3/) | `supplementary_bcst/raw/reviewer/reduced_causal_ablation_adam800_36_v3/`; `supplementary_bcst/results/causal/`; `supplementary_bcst/figure_data/bcst_unique_optimum_causal_ablation.csv`; `supplementary_bcst/figure_data/bcst_unique_optimum_causal_contrasts.csv` |
| Finite-depth gradient diagnostics; Fig. 4; Results 2.3; Appendix H.7; Tables 5, 12 | [Gradient runner](supplementary_bcst/bcst_unique_primary_campaign_20260809/gradient_inputs/historical_source/trainability_runner.py); [unique-optimum evaluation](supplementary_bcst/bcst_unique_primary_campaign_20260809/source/bcst_unique_campaign/supplements.py); [simulation dependencies](supplementary_bcst/bcst_unique_primary_campaign_20260809/gradient_inputs/historical_source/paper_source/source/) | `supplementary_bcst/bcst_unique_primary_campaign_20260809/gradient_inputs/FROZEN_ANGLE_INPUT_MANIFEST.json`; `supplementary_bcst/bcst_unique_primary_campaign_20260809/gradient_inputs/original_random_rows/`; `supplementary_bcst/bcst_unique_primary_campaign_20260809/run/preflight/STAGE1_IMPORT_CERTIFICATE.json`; `supplementary_bcst/gradient_results/`; `supplementary_bcst/figure_data/bcst_degree6_trainability_seed_summaries.csv` |

## SBM — Results 2.4 and Appendix I

| Paper item | Code | Data |
| --- | --- | --- |
| Graph construction, circuits and optimization; Appendix I.1, I.2 | [Simulation and graph construction](sbm/code/sbm_standard_core.py); [main numerical runner](sbm/code/run_depth_scan.py); [depth-8 runner](sbm/code/run_natural_sbm_adam_direct_microbatch.py) | `sbm/config/main_runtime_tasks.json`; `sbm/config/p8_runtime_tasks.json` |
| Nine aligned graphs; Fig. 5; Results 2.4; Appendix I.3; Table 13 | SBM code above | `sbm/data/main_results/*seed10[0-7].json`; `sbm/data/main_results/*seed109.json`; `sbm/data/p8_results/*seed10[0-7].json`; `sbm/data/p8_results/*seed109.json`; `sbm/data/logs/main/*seed10[0-7].log`; `sbm/data/logs/main/*seed109.log`; `sbm/data/logs/p8/*seed10[0-7].log`; `sbm/data/logs/p8/*seed109.log` |
| Coarse–fine mismatch control; Appendix I.4 | SBM code above | `sbm/data/main_results/*seed108.json`; `sbm/data/p8_results/*seed108.json` |
