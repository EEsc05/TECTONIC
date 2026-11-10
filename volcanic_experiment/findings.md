# Volcanic deformation forecasting: findings

Completed retrospective experiment; all new artifacts remain in `volcanic_experiment/`. The primary model was fixed before test evaluation. Source data are archived LOS estimates, not verified real-time monitoring products. No eruption classifier was trained.

## Primary results

At 64×64 native pixels, one acquisition ahead (usually 6 days in this archive), robust rank 3 MAE is **14.432 mm**, persistence **15.672 mm**, EMA120 **14.463 mm**, and four-branch equal **14.847 mm**. Scores average maps within source-frame groups, then average the 6 independent held-out groups. These are new volcanic results, not directly comparable to earlier 17-region MAEs.

| model | MAE_mm | RMSE_mm | medianAE_mm | p90AE_mm | q90_change_MAE_mm |
| --- | --- | --- | --- | --- | --- |
| persistence | 15.672 | 17.338 | 15.233 | 24.631 | 27.612 |
| exponential_level | 14.463 | 16.598 | 13.688 | 24.433 | 21.786 |
| lowrank_r3_robust | 14.432 | 16.583 | 13.650 | 24.421 | 21.784 |
| heterogeneous_equal | 14.847 | 17.166 | 13.954 | 25.430 | 22.275 |
| heterogeneous_constrained | 14.696 | 16.971 | 13.854 | 25.096 | 22.093 |
| temporal_spatial | 15.245 | 17.684 | 14.345 | 26.196 | 22.672 |
| without_context | 14.834 | 17.148 | 13.979 | 25.362 | 22.226 |
| without_quality | 15.115 | 17.518 | 14.183 | 26.012 | 22.562 |
| matched_cnn | 20.065 | 26.479 | 15.947 | 44.382 | 33.742 |
| standalone_ensemble_constrained | 17.772 | 22.102 | 15.330 | 35.164 | 28.244 |
| selected_lowrank | 14.435 | 16.573 | 13.657 | 24.397 | 21.786 |
| temporal | 16.606 | 19.388 | 15.517 | 28.799 | 24.115 |
| context | 15.337 | 17.695 | 14.420 | 26.345 | 22.795 |
| quality | 14.243 | 16.337 | 13.508 | 24.020 | 21.640 |

RMSE, medianAE and p90AE here are means of mapwise statistics, not quantiles of a globally pooled error array. Paired frame-bootstrap95% intervals and exact frame-level sign-flip tests are in `paired_comparisons.csv`. Six groups make intervals and tests fragile. Adjacent pixels and overlapping windows are never independent trials. Only original lowrank versus persistence at 64px/h1 is the primary hypothesis; all other tests are exploratory with Holm adjustment.

The apparent gain over persistence is uncertain (paired frame-level p=0.1875), and the difference from EMA is only 0.031 mm. The hybrid gives no improvement. These results provide little evidence of added spatial forecasting skill beyond smoothing; apparent anticipation may include noise filtering and mean reversion.

## Calendar horizons of 12–48 days

The primary acquisition-based comparison is often shorter than 12 days in this archive. These separate calendar results directly address the 12–48day question, with explicit±6day matching and counts in `calendar_matching.csv`. They remain secondary rather than replacing the predeclared primary:

| model | horizon | mae_mm | q90_change_mae_mm |
| --- | --- | --- | --- |
| exponential_level | 12 | 16.326 | 26.801 |
| exponential_level | 24 | 17.817 | 33.287 |
| exponential_level | 48 | 22.021 | 45.633 |
| heterogeneous_equal | 12 | 16.840 | 27.497 |
| heterogeneous_equal | 24 | 18.223 | 34.051 |
| heterogeneous_equal | 48 | 22.504 | 46.320 |
| lowrank_r3_robust | 12 | 16.346 | 26.831 |
| lowrank_r3_robust | 24 | 17.800 | 33.314 |
| lowrank_r3_robust | 48 | 21.873 | 45.452 |
| persistence | 12 | 18.150 | 31.314 |
| persistence | 24 | 19.098 | 35.534 |
| persistence | 48 | 21.913 | 45.271 |

## Answers to the 19 research questions

1. **Included systems and rationale.** All 16 documented source regions were retained across archive roles: agung, campi_flegrei, corbetti, domuyo, erta_ale, etna, fujisan, galapagos, kilauea, kverkfjoll, laguna_del_maule, lawu, reykjanes, rinjani, taal, tambora. The shared Galapagos frame contributes one Sierra Negra crop; it is not five independent volcanoes. Shared Agung/Rinjani and Domuyo/Laguna frame IDs remain in the same roles. The archive deliberately includes deformation/artifact examples and is selection-biased. Full inventory, resolution and dates: `data/dataset_inventory.csv`. No existing Japanese city was silently relabelled as a summit-centred region.

2. **Independent activity episodes.** Evaluated fixed-model annotations yield {'documented_background': 1, 'unrest_or_eruption': 2}. The annotation catalogue contains 3 unrest/crisis episodes and 1 narrow documented background episode. The fixed stride supplies origin-labelled forecasts in 2 of those unrest episodes; Agung's short high-alert window is represented by a forecast issued one day before onset whose target lies inside the episode, recorded separately in `activity_target_episode_features.csv`. Thus target-labelled characterization includes 3 unrest episodes without pretending that origin-labelled samples do. Phase ranges are in `data/activity_annotations.json`. Counts of forecast windows are in `activity_episode_features.csv`; they do not increase the event count. Unknown dates remain unknown. Classification is not supported.

3. **Actual forecast areas.** Native32,64,128,256 crops are all direct extractions. Across regions,64px widths range 3.06–7.12 km and heights 7.08–7.14 km. Areas and supported fractions for every crop are in `data/crop_footprints.csv`; these are bounding-box areas, not guaranteed coherent land areas. WGS84 local curvature converts the native geographic increments; no upsampling. Source boxes approximate named summit centres; Galapagos uses declared Sierra Negra(-0.83,-91.17). Larger domains can still truncate broad signals, as shown by edge-contact statistics; full deformation coverage is not established.

4. **Actual horizons.** Testh1 has actual median 6 days, range 6–24 days. `acquisition_elapsed_days.csv` gives each volcano/horizon. Calendar12/24/48 days selects nearest strictly future acquisition within±6 days and excludes unmatched pairs. See `calendar_elapsed_days.csv`, `calendar_summary.csv`. No acquisition horizon is labelled an exact day horizon. `calendar_matching.csv` reports unmatched-origin counts. Calendar comparisons are conditional hindcasts using actual future observation timestamps; availability of that schedule as of the origin is not demonstrated. Acquisition-slot interval widths are transferred to calendar slots without date-specific recalibration, a further limitation.

5. **Low rank versus simple baselines.** Primary MAEs above show the direction and magnitude. Lowrank minus persistence is -1.240 mm, bootstrap95% [-2.491,+0.212], exact sign-flip p=0.1875. The assumed symmetric independent frame effects may be questionable at n=6; statistical support is limited to this dataset and processing.

6. **Hybrid incremental value.** Four-branch equal minus lowrank is +0.416 mm. Branch residual correlations are in `branch_residual_correlations.csv`; near-one correlations mean common errors dominate, not complementary skill. Full versus temporal+spatial and leave-context/leave-quality contrasts are in `paired_comparisons.csv`. Frozen constrained weights were reused, not searched on volcano tests. Large neural/tree transfer failures are reported rather than discarded. The prior official benchmark remains `standalone_ensemble_constrained`, volcanic MAE 17.772 mm.

7. **Spatial tradeoff.** Original lowrank h1 MAEs at 32/64/128/256 are 15.573, 14.432, 13.117, 12.215 mm. Figure11 and footprints show coverage/error tradeoffs. This is a predeclared comparison, not test-based selection of an optimal operational crop. Changing crop also changes scored pixels; accuracy differences are not evidence of improved skill on a common footprint. Native resolution was retained. An exploratory fixed equal blend of native64/128/256 lowrank forecasts was evaluated on the same central64 support: h1 MAE 14.435 mm (`exploratory_multiscale_summary.csv`). It adds no trained weights. This diagnostic was declared after the primary selection freeze, during test computation, before inspecting numerical test metrics; it is not a new selected winner. `matched_central32_context_summary.csv` scores all context sizes on identical central32 pixels, separating spatial-context effects from changing target footprints.

8. **History and rank.** The18-setting validation grid selected lr_h6_r5_robust0 (validation frame-macro MAE 10.780 mm), frozen before test. Its test MAE is 14.435 mm. Robust/nonrobust ranks 1/3/5 and histories 6/12/18 use identical eligible validation origins and6-map support. With six maps, centred rank 5 spans the full temporal subspace, so this selected setting is not evidence that strong spatial compression is optimal. This remains a secondary selected configuration; original robust rank 3/history 6 stays the fixed primary. Only2 independent validation frame groups make selection unstable. The ridge penalty stays4, so long histories change both information and effective trend leverage.

9. **Accuracy during unrest.** `activity_stratified_metrics.csv` and `activity_episode_features.csv` separate origin and target activity. Historical2017/2018 crisis comparisons use only immutable, untrained original lowrank/persistence/EMA; later-selected volcanic settings and frozen Japanese learned models are excluded from those historical hindcasts. With only one background episode and volcano confounding, a general unrest penalty cannot be estimated credibly.

10. **Anticipating change.** At primary setting, mean predicted/observed change-pattern correlation is 0.247, cosine similarity 0.381, sign agreement on observed ≥5 mm pixels 0.707, coherent-region recall 0.694 and precision 0.720. The5 mm threshold is fixed; four-connected components must contain ≥4 native pixels. Actual target changes define evaluation strata only. The model predicts complete maps without an oracle future-event mask. Stable-map MAE alone does not establish anticipation. Subtracting the same origin gives identical change and absolute-map errors, so no duplicate improvement is claimed.

11. **Captured/missed structures.** Figures1–4 show an independently chosen near-January2020 Taal forecast, including past-only components. The trend assumes previously observed spatial modes persist; it cannot discover a wholly new source pattern from future observations. Rapid changes, wrap errors, atmosphere, ice motion and tectonic signals may all cause misses. SourceLOS sign is retained; positive/negative areas are not labelled vertical uplift/subsidence. Gradient and centroid errors are in the metric CSVs; undefined centres and zero-variance correlations remain missing rather than zero skill.

12. **Error relative to signal/noise.** Primary observed mean absolute change is 15.672 mm versus forecast MAE 14.432 mm. Magnitude bias is -3.520 mm. Error-on-observed-top10%-change is 21.784 mm. This does not measure physical measurement uncertainty: per-acquisition coherence, atmospheric covariance and calibrated producer uncertainty are absent. Source inversion may underestimate extreme deformation, notably Galapagos unwrap aliasing.

13. **Prediction intervals.** Nominal 90% empirical residual intervals cover 82.4% of test pixels and 68.7% of observed-top10%-change pixels; mean width 52.37 mm. Widths come exclusively from train-volcano residuals with targets before 2018, separated from validation and held-out volcanoes. Equal deterministic pixel samples per calibration map approximate residual distributions; unequal available records and geographic shift remain. These are empirical marginal intervals, not guaranteed conformal or physical measurement confidence bounds. Quality/availability proxies and seed variability are not presented as calibrated uncertainty.

14. **Forecast features versus activity.** Pure past-supported features, including predicted magnitude, rates, dispersion, local component amplitudes, coherent area and gradients, are saved in `forecast_features.csv`. Unlike scoring diagnostics, these features never use target validity. Episode-level contrasts are in `activity_feature_contrasts.csv`. Descriptive differences are provided without p-values or bootstrap confidence intervals because only one background episode exists. Volcano and episode differences are inseparable. `activity_narrow_definition_sensitivity.csv` excludes the long Taal low-alert phase. No forecast-derived unrest claim is supported by feature screening. Component trajectories are arbitrary-sign local bases and cannot be compared across volcanoes without alignment.

15. **Statistical support.** `paired_comparisons.csv` gives effect sizes, frame-bootstrap intervals, exact sign-flip p-values and exploratory Holm adjustment. At most6 independent held-out frame groups support the forecasting comparisons. The confirmatory hypothesis was frozen original lowrank versus persistence, h1/64px; all other scientific metrics and activity characterization are exploratory. Event counts preclude reliable quiet-versus-unrest tests; thousands of pixels do not fix this.

16. **Geographic generalization.** Seven untouched test regions represent6 independent source-frame groups. Train, validation and test frame groups are disjoint. Validation input support begins 2017 and targets finish before 2019; test origins begin 2019. Unseen-volcano histories may include older maps, which are legitimately known at the origin. No held-out-volcano outcomes train global weights, calibration or hyperparameter selection. Shared global atmospheric effects and source curation limit independence and external validity.

17. **Eruption classification.** Deferred: only 3 positively annotated episodes,1 documented background episode, and no aligned independent seismic/gas/thermal streams. Proposed architecture and feasibility criteria are in `classifier_feasibility.md`. Ground-deformation skill is not eruption-prediction skill. The absence of a classifier is an explicit evidence-based outcome.

18. **Operational limitations.** Whole-record InSAR inversion, incomplete reference/sign/coherence metadata, unknown acquisition-time availability, zero-encoded unsupported pixels, possible real-zero exclusion, deformation-enriched source selection, sparse independent labels, irregular cadence, geographic transfer, and unwrapping aliasing. Source files have no explicit missing mask: exact zeros are conservatively excluded per acquisition, a documented heuristic. Native boxes are larger than priorcity32 crops but not guaranteed complete physical source coverage. No external weather input is added, though upstream archive processing may include atmospheric correction. No operational warning or eruption prediction claim is justified.

19. **Priorities.** Acquire acquisition-vintage InSAR with coherence/reference metadata and published monitoring histories for substantially more independent unrest and verified background episodes. Freeze prospective volcano/time splits; test whether physically meaningful changes improve over zero-change at matched supports. Examine regularization-induced trend attenuation and mode stability on validation only. Add aligned monitoring modalities and calibrated covariance before any eruption classifier.

## Runtime and reproducibility

No new global neural/tree model was trained: new training time is0 seconds for those frozen checkpoints. Statistical models fit locally at each forecast origin; `model_runtimes.csv` distinguishes local fitting/prediction from global training and gives inclusive branch costs. Fusion overhead alone is tiny because branch work is shared; inclusive costs avoid presenting that as full hybrid runtime. `runtime_by_origin.csv` includes prediction and scoring time, with final archive compression included only in overall execution wall time. Original checkpoint training times are copied read-only into `frozen_checkpoint_training_history.csv`. `validation_grid.csv` records rank/history local timings (shared decomposition costs counted per configuration as standalone estimates). Download/preparation, version and seed records are under `data/`; `execution.json` records this modelling invocation. Repeated/restarted runs reuse prediction caches; existing artifacts stay unchanged.

To reproduce: follow `README.md`. All quantitative conclusions above are generated from saved CSVs and predictions; no target information enters forecasting inputs. Historical challenge results remain separate from the2019+ sealed geographic test.

Frozen-branch numerical note: the original context feature calculation can take a square root of a tiny negative uniform-filter roundoff residual. Those feature values become missing and are handled by the frozen LightGBM model’s native missing-value routing. Final prediction arrays were checked finite; prior source code was not changed.
