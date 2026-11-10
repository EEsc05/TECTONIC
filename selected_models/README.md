# Selected frozen forecasting models

This is a portable subset of the existing experiments, with no new model selection or retraining. Start with the [repository instructions](../README.rst), then run `python -m selected_models.reproduce` from the repository root. The compact data and all three primary CVAE seed draws are included in ordinary Git.

| Model | Exact implementation / artifact | Evidence |
|---|---|---|
| Quality-aware smoothing | `volcanic_forecasting_v2/src/stage1/models.py`, `LocalFit`, `quality_v2_h6_half30_w0`: six maps, 30-day half-life, robust MAD clipping, no trend gain | `results/v9_point_metrics.csv`; 10.522253 mm MAE |
| EMA30 | Same `LocalFit` / `smooth`; six acquisitions, actual-day exponential weights with 30-day half-life | `results/v9_point_metrics.csv`; 10.549943 mm MAE |
| Robust low-rank | `original_lowrank.py`: exact function bodies from `volcanic_experiment/scripts/engines.py` and original history/mask helpers; rank 3, six maps, robust clipping, actual-time ridge coefficient forecast | `results/original64_summary.csv`; 14.431700 mm MAE at 64×64 / next actual acquisition |
| Four-branch hybrid | V9 `four_horizon`: EMA temporal, block16 linear-coefficient spatial, residual SVR context, frozen quality branch; shrunk horizon-bin weights from original 2022 calibration | `artifacts/four_horizon_weights.json`, original V9 component checkpoints, `artifacts/v9_points.npz`; 10.412709 mm MAE |
| CVAE | V9 `neural.VAE`, configuration `cvae_coarse3`, original seeds 20261206/7/8; conditional prior, decoder, frozen spatial preprocessor and EMA30 residual parent | Original `cvae_coarse3_seed*.pt`, `_draws.npz`, calibration and scored CSVs in V9 paths; mean-seed CRPS 8.172474 mm |
| Persistence | Original past-filled last map, unchanged over requested horizons | `original_lowrank.py`, `results/original64_summary.csv`; 15.671541 mm MAE |

The original V9 files are under `volcanic_forecasting_v2/experiments/v9_model_optimization/`, including selections, timings, uncertainty and reports. Their shared historical tables contain comparator context; only the selected models and necessary source dependencies/checkpoints are packaged. Original reports are copied unchanged. They can reference figures/artifacts outside this subset; the complete archive preserves that historical context.

## What the replay verifies

- Checksums for all published files except the self-referential manifest itself.
- Quality and EMA recomputation over all 1,714 original matched dataset pairs.
- Exact V9 validation volcano-macro MAEs from frozen prediction arrays.
- Actual four-branch checkpoint inference on original calibration, validation and test cases; agreement with the frozen `four_horizon` maps.
- Original robust low-rank and persistence forecasts at all 143 supported original test origins, scored with the inherited masks and six frame-group macro aggregation.
- CVAE CRPS from all original V9 validation draws across three training seeds, plus deterministic CPU sampling checks from every frozen checkpoint.
- Separate replay runtime measurements. These measure replay/inference/scoring, not original training; original training timings remain in V9 diagnostics.

`artifacts/four_horizon_weights.json` reconstructs the already-frozen horizon weights using the exact original `fusion_growth.py` recipe and 2022 calibration cases. Agreement was checked against every original saved hybrid map. No 2023 outcomes were used to select weights, no experts were refitted, and no new winner was chosen.

`data/original64/` stores exact central 64×64 crops of the original native256 caches, with actual timestamps and spacing. No resampling or interpolated targets were introduced. The original evaluation's zero-as-missing convention is preserved by its extracted history function; the V9 adapter keeps its distinct inherited validity convention. Do not merge their scores or change missingness rules and claim the original results.

Original V6 source manifests retain archival absolute raw-cache paths for provenance. The replay reads the included `examples.npz` and static geometry metadata; it does not need those old absolute paths. Historical full-run hash checks/HPO runners reference artifacts intentionally omitted from the subset. Use the selected replay command, or restore the complete release archive for the full historical workspace.

## Interpretation

V9 2023 validation is an already-inspected development assessment, not independent confirmation. The original 64×64 volcanic evaluation uses a different cohort, mask convention, horizon definition and frame-group weighting. CVAE CRPS measures probabilistic forecast quality, not point MAE. Signed LOS displacement is not vertical displacement. Historical statistical patterns do not establish eruptions or operational forecasting readiness. Stage 2 is not included or run.

The repository contains no virtual environments, installed CUDA libraries, raw duplicate ZIP downloads, unrelated monitoring/event models, or later V10–V12 experiments. Those files remain unchanged locally and in the complete GitHub release backup.
