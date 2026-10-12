# Source Data

Building QAOA from QAOA: Learned-projector QAOA for Hierarchical Optimization.
Numbering follows the main manuscript and the separate Supplementary Information.
CSV files use UTF-8 encoding and one header row. Probabilities are dimensionless;
resource and total-cost columns are expressed in logical resource units (RU).

## Figures

| Paper item | Data files |
| --- | --- |
| Figure 2a,b: selected-depth BCST comparison | `figure_2_realizations.csv`; `figure_2_initializations.csv` |
| Figure 2c,d: BCST depth curves | `figure_2_curve_summary.csv`; `figure_2_realizations.csv`; `figure_2_initializations.csv` |
| Figure 3: BCST ablation comparison | `figure_3_realizations.csv`; `figure_3_initializations.csv`; `figure_3_curve_summary.csv` |
| Figure 4: gradient diagnostics | `figure_4_gradients.csv` |
| Figure 5: SBM comparison | `figure_5_sbm.csv` |
| Supplementary Figure 2: two-level BCST validation | `supplementary_figure_2_bcst_validation.csv` |
| Supplementary Figure 3: two-level BCST depth curves | `supplementary_figure_3_bcst.csv` |
| Supplementary Figure 4: two-level BCST ablations | `supplementary_figure_4_bcst.csv` |

## Numerical tables and additional results

| Paper item | Data files |
| --- | --- |
| Supplementary Table 3 | `supplementary_table_3_selected_depths.csv`; `figure_2_realizations.csv`; `figure_3_realizations.csv` |
| Supplementary Table 4 | `supplementary_table_4_lp_depth.csv`; `figure_2_initializations.csv` |
| Supplementary Table 6 | `supplementary_table_6_fixed_budget_instances.csv` |
| Supplementary Tables 7 and 8 | `supplementary_tables_7_8_matched_instances.csv` |
| Supplementary Table 9 | `supplementary_figure_3_bcst.csv` |
| Supplementary Table 11 | `supplementary_table_11_probability_contrasts.csv`; `supplementary_figure_4_bcst.csv` |
| Supplementary Table 12 | `supplementary_table_12_gradient_comparisons.csv`; `figure_4_gradients.csv` |
| Supplementary Table 13 | `supplementary_table_13_sbm_summary.csv`; `figure_5_sbm.csv` |
| Supplementary Section S7.5: standard QAOA diagnostic | `supplementary_section_S7_5_standard_qaoa.csv` |
| Supplementary Section S8.2: SPSA results | `supplementary_section_S8_2_spsa_confirmation.csv`; `supplementary_section_S8_2_spsa_tuning.csv` |
| Supplementary Section S8.2: target-set sensitivity | `supplementary_section_S8_2_target_sensitivity.csv`; `supplementary_section_S8_2_tuning_target_sensitivity.csv` |
| Supplementary Section S9.4: coarse-fine mismatch control | `supplementary_section_S9_4_sbm_control.csv` |

## Data fields and aggregation

- `seed`, `realization_seed`, `problem_seed` and `graph_seed` identify the recorded random instances; `restart_index`, `restart_seed` and `angle_index` identify optimizer starts or sampled angle vectors. `N` is the number of binary variables and `depth` is the active circuit depth.
- `selected_depth` marks the depth used in the selected-depth comparison. `selected_unique_operating_depth` has the same role in Supplementary Figure 3. `p1` and `p2` give the two stage depths for SBM circuits; standard QAOA stores its circuit depth in `p2`.
- `probability`, `success_probability` and `ground_probability` give the probability of measuring the target. `ground_space_success_probability` gives this probability in the standard-QAOA diagnostic. `feasible_probability` and `feasible_mass` give the probability of satisfying the constraints.
- `RTS99` is the integer number of independent preparations needed for 99% success. `per_shot_RU`, `RU`, `terminal_RU`, `terminal_RU_integer_decimal` and `resource_units_per_run` give the cost of one complete preparation and terminal evaluation.
- `total_cost`, `total_cost_RU`, `RTS99_x_RU` and `ground_total_logical_cost_integer_decimal` give the BCST total cost. `continuous_c99` and `continuous_total_cost_RU` give the continuous repetition-cost measure used for SBM.
- Main-BCST realization values are geometric means over four optimizer starts. Curve summaries are geometric means over ten realizations; the `q25`, `q75`, `min` and `max` suffixes identify quartiles and extrema over those realizations. The `n_realizations` column gives the number of contributing realizations.
- Two-level BCST rows report the minimum-energy circuit selected across the recorded restarts. `target` distinguishes the unique ground state (`ground`), the two lowest-cost states (`best2`) and the eight lowest-cost states (`best8`); `target_count` gives the number of target configurations. A blank refinement depth or update count denotes the stage-1-only result.
- Gradient data report normalized root-mean-square derivatives and the cosine between training-loss descent and increasing target probability. `common_objective_gradient_rms_normalized` refers to the final objective; `training_loss_gradient_rms_normalized` refers to the training loss. `log_unique_probability_gradient_rms` gives the gradient magnitude of the logarithm of target probability. `parameter_count` is the number of active circuit angles and `alignment_defined` records whether the cosine is defined. Supplementary Table 12 first takes the median over 12 angle vectors per instance, then the geometric mean of gradient ratios and the arithmetic mean of alignment differences over six instances.
- SBM summaries use medians over nine aligned graphs. The standard-QAOA envelope takes the largest target probability and the smallest total cost across depths 4, 6 and 8 for each graph. `LP_higher_probability_count` and `LP_lower_cost_count` count paired graph or realization comparisons won by LP-QAOA. Blank comparison counts identify the LP-QAOA reference row.

## Method identifiers

| Identifier | Method |
| --- | --- |
| `lp_three_stage` | Three-stage LP-QAOA |
| `lp_two_stage_separate` | Two-stage LP-QAOA with separate phases |
| `lp_two_stage_combined` | Two-stage LP-QAOA with a combined phase |
| `block_xy_separate` | Block-XY QAOA with separate phases |
| `block_xy_combined` | Block-XY QAOA with a combined phase |
| `omit_feasibility` | Learned-projector circuit omitting the joint-feasibility stage |
| `warm_start_block_xy` | Warm-start block-XY QAOA |
| `uniform_projector` | Uniform-state projector circuit |
| `learned_projector`, `LP` | XY-LP-QAOA in the two-level BCST experiments |
| `ordinary_native_direct` | Block-XY QAOA with a combined phase |
| `decoupled_native_direct` | Block-XY QAOA with separate phases |
| `same_stage1_state_direct`, `WarmXY` | Warm-start block-XY QAOA |
| `stage1_only` | Optimized stage-1 preparation |
| `native_grover_d`, `known_projector_hd` | Projector onto the block-cardinality initial state, with the penalized objective |
| `projector_hd` | Learned projector with the penalized objective |
| `xy_hf` | Warm-start block-XY mixer with the final objective |

SBM method suffixes specify the stage depths. Its `stage1_only` method optimizes
the coarse objective; its `lp_qaoa`, `warm_start_qaoa` and `standard_qaoa` labels
identify the circuits in Supplementary Section S9.2. SPSA `configuration`
identifiers retain the settings stored in the experiment records.

## Supporting experiment records

The repository and submission archive retain these paths:

| Content | Paths |
| --- | --- |
| Main BCST realizations, optimizer starts, frozen preparations and RU model; Supplementary Tables 1 and 2 | `main_bcst/data/current/`; `main_bcst/data/inputs/`; `main_bcst/data/resource_model.json` |
| Standard-QAOA instances and summaries | `main_bcst/data/n18/inputs/`; `main_bcst/data/n18/corrected_numerical_metrics.json` |
| Two-level BCST instance outcomes and optimization diagnostics; Supplementary Tables 5 and 10 | `supplementary_bcst/results/` |
| SPSA selected-depth summary | `supplementary_bcst/historical_spsa/results/EXACT_GROUND_REANALYSIS.json` |
| SBM graph settings, optimized angles and outcomes | `sbm/config/`; `sbm/data/main_results/`; `sbm/data/p8_results/` |

The repository root `LICENSE` contains the MIT license for the code and data.
