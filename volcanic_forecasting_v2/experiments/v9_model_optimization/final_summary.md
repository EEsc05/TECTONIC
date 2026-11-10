# V9 systematic Stage1 optimization — executed results

**Stage2 suspended. No Stage2 seismic, monitoring-fusion or eruption-prediction experiments were run.** Parent V3–V8 hashes are unchanged. Version **v9_optimization_v1**, dataset **global_comet_LOS_v6_2_immutable**, split **v9_nested_training_cv_locked2023_v1**. Peru not scored. All candidates/checkpoints/calibration factors were locked before V9 final2023 scoring. That cohort was already used by previous research, so V9 provides no new confirmatory validation.

## Answers to the27 research questions

1. **Prior systematic optimization:** low-rank had narrow structured rank/history/robustness grids; most CNN, context and neural training parameters were fixed. Prior seed/data-size repeats were not hyperparameter optimization. See previous_hyperparameter_audit.md and its executed-configuration inventory.
2. **Previously unexplored:** broad conventional linear/tree/kernel comparisons, scaler/target controls, latent regressors, CNN architecture/training settings, neural LR/depth/width/annealing combinations, variance shrinkage/df and quantiles. V9 tests bounded subsets; it does not claim exhaustive optimization.
3. **Optimization gains:** diagnostics/optimization_gains.csv reports like-for-like internal screening gains, which are optimistic search differences. Locked tables compare new models to fixed references on the same future observations. Changes combine feature/target/HP choices; not every gain can be attributed to one parameter.
4. **Conventional ML:** internal3fold champion **SVR / SVR_0**, mean fold MAE **18.553mm**. All13 eligible pixel families were genuinely benchmarked; GPR/PLS were used in suitable latent tasks. Internal and final-development rankings need not agree.
5. **Versus simple controls:** the lowest descriptive new2023 MAE is **four_horizon: 10.413mm**. Compare against frozen quality/EMA in the tables and multiplicity-aware intervals, not a post-hoc retuned baseline. This descriptive ranking does not change the prelocked shortlist or imply promotion.
6. **Feature representations:** shared31-feature past-only library includes level, rates, variability, masks, physical gradients, cadence, geometry and local low-rank information. Four matched Ridge ablations and grouped whole-origin permutation sensitivities are in the outputs. Importance is predictive, not causal; correlated-feature permutation can create out-of-domain combinations.
7. **Target formulation:** absolute, delta, EMA residual, Kalman residual and train-scaled delta were tested with matched Ridge settings. Latent coefficient changes were a separate task. Temporal/spatial/blend residual context targets were tested on chronological/region-held internal folds. See target_formulation_results.csv and context_branch_results.csv; kernels use bounded samples, not unconstrained full-map fits.
8. **Spatial decomposition:** primary locked representation/forecaster **spatial_block16_linear**. Eight representations were screened separately from11 coefficient forecasters on two shortlisted bases. Actual observed-history reconstruction and principal-angle stability are distinct diagnostics; low reconstruction error is not a future forecasting score.
9. **Coefficient algorithm:** selected **linear** on **block16** by internal future MAE. Local constant/EMA/Kalman/regularized-linear controls and global Ridge/Huber/forest/ExtraTrees/LightGBM/GPR/PLS were compared using common historical coefficient inputs. Nonlinear trees are not assumed to extrapolate outside their training target range.
10. **Temporal branch:** **EMA_h30_trend0**, selected only by equal-fold training-internal macro MAE. Thirty actual-time EMA/Kalman combinations were evaluated; frozen robust-trend and selected smoothing references remain in the final benchmark. State/noise/damping choices, not merely model labels, were varied.
11. **Spatial branch:** future-MAE champion plus a gradient specialist within10% MAE and a Ridge reference were retained where distinct. Six maps are available to all; older12/18map references are excluded from the primary equal-information comparison.
12. **Context branch:** **context_SVR_residual_temporal**, target **residual_temporal**. Algorithms/target forms were selected on2018–19 temporal/held-region training CV. Its learned map remains separately evaluated, not automatically trusted because it is more flexible.
13. **Reliability:** **HGB_relative**, chosen using independently OOS2022 trust-weighted forecast MAE. All quality heads fit chronological OOF expert errors, never in-sample expert residuals. Absolute/squared/relative targets and Ridge/Huber/forest/ExtraTrees/boosting/quantile alternatives were tested. Error estimates are not calibrated sensor-quality probabilities.
14. **Complementarity:** the residual-correlation arm alone failed; the spatial+context equal pair nevertheless improved training-OOF MAE by about2%, satisfying the registered alternative gain gate. Both arms are audited; original pair-stage outputs are retained. Correlation alone would have stopped a justified combination.
15. **Two branches:** TS/TC/SC equal/regularized and quality-controlled TS were tested, with single experts as ablations. Their2023/old-assessment results and frame/block intervals determine descriptive value; no repeated tuning on those results occurred.
16. **Three branches:** enabled by the OOF gain gate and tested with equal/regularized/horizon/regime/reliability/small-gate fusion. Any additional gain must be compared to two experts on matched cases and adjusted intervals.
17. **Four branches:** four-map and reliability alternatives plus leave-one-branch-out ablations were run. A smaller system remains an acceptable outcome; the results do not promote four branches by default.
18. **Fusion strategy:** weights and lightweight gates use independently OOS2022 forecasts, not2023 labels. Final descriptive rankings follow below. They are not used to rewrite the locked model choice. Hyperparameter reuse within internal OOF errors can still impart selection optimism, so new independent evidence remains necessary.
19. **CNN HPO:** six coarse plus two LR refinements, depth2/3/4,width12/16/24,kernel3/5,residual skips,lr/batch/decay/loss/normalization; two configs each with three full-training seeds. All-lag/held-region review selected **cnn_coarse0_refine** as the internal primary. Full-fit epoch counts came from internal epochs, with no2022 early-stop tuning.
20. **CVAE HPO:** latent8/16/32,width64/128,depth1/2,beta.03/.1/.3,annealing,LR/batch/decay and Gaussian-NLL/Huber alternatives were evaluated; primary **cvae_coarse3_refine**. Shared config labels mae/mse map to the implemented Gaussian NLL here; no Laplace-loss result is claimed. Posterior diagnostics use future targets only to diagnose, never to forecast.
21. **Flow HPO:** width/depth,LR/batch/decay,velocity loss,normalization/conditioning and Euler/Heun varied; primary **flow_coarse1_refine**. Separate8/16/32step and16/32/64draw calibration diagnostics did not change primary32draw/16step final forecasts. Sampling seeds and training seeds are distinguished.
22. **Conventional probability:** six mean+Gaussian controls, heteroscedastic/structured/Student, BayesianRidge posterior, latent-GP posterior and quantile boosting were tested. Covariance shrinkage/df/horizon variance/heteroscedastic regularization were searched internally, then2022 CRPS scaling was separate. Proper scores, coverage and sharpness must be read together, not as proof Gaussian errors are physically true.
23. **Seed stability:** two selected configs per neural family were trained with three independent seeds each. See diagnostics/seed_stability.csv; do not infer a family improvement from its single best seed. CNN predictive variation also remains in the locked point cases.
24. **Generalization:** final tables show2023, old temporal, old geography and exact observed lags. The geography contains few shared radar-frame clusters; all these populations were previously inspected in earlier phases. There is no untouched confirmatory assessment or NISAR validation.
25. **Accuracy versus complexity:** cheap stateless EMA/quality and linear heads remain serious controls against kernel inference, ensembles and neural sampling. Training/inference/sampling/calibration/scoring clocks retain their actual scopes; CPU/GPU and combined scopes must not be compared as identical.
26. **Promotion:** insufficient new independent evidence to replace any frozen model. V6/V7/V8 references and selections remain unchanged, regardless of descriptive V9 winners. Shared-frame/six-map block maxT and Bonferroni-normal sensitivity address multiple comparisons but do not create fresh confirmation.
27. **Remaining limits:** architecture/optimization interact with sparse irregular history, missing-target latent encoding, nonstationary LOS/reference/geometry, correlated examples and unknown historical SAR issue times. No physical mechanism or intrinsic predictability limit is identified solely by model rankings. Pure neural geometry and fully masked latent objectives remain gaps; conventional geometry ablations are genuine and separately labeled.

## Locked2023 point results (descriptive)

| role | model_id | mae | rmse | extreme_mae | gradient_mae | localization_IoU |
| --- | --- | --- | --- | --- | --- | --- |
| validation | four_horizon | 10.4127 | 12.6324 | 36.9295 | 9.1178 | 0.0233 |
| validation | ablate_1 | 10.4246 | 12.6497 | 36.1635 | 9.1794 | 0.0249 |
| validation | four_regularized | 10.4283 | 12.6506 | 37.0263 | 9.1206 | 0.0230 |
| validation | ablate_0 | 10.4314 | 12.6565 | 37.2525 | 9.1348 | 0.0239 |
| validation | four_regime | 10.4320 | 12.6555 | 37.0274 | 9.1257 | 0.0229 |
| validation | four_smallgate | 10.4339 | 12.6711 | 37.5531 | 9.0705 | 0.0227 |
| validation | three_horizon | 10.4343 | 12.6504 | 37.3547 | 9.1288 | 0.0210 |
| validation | pair_TC_equal | 10.4391 | 12.6588 | 36.2879 | 9.1770 | 0.0228 |

## Previously inspected temporal point assessment

| role | model_id | mae | rmse | extreme_mae | gradient_mae | localization_IoU |
| --- | --- | --- | --- | --- | --- | --- |
| temporal_evaluation | ML_SVR_0 | 15.6876 | 19.5719 | 47.7085 | 14.5610 | 0.0246 |
| temporal_evaluation | context_SVR_residual_temporal | 15.6876 | 19.5719 | 47.7085 | 14.5610 | 0.0246 |
| temporal_evaluation | expert_context | 15.6876 | 19.5719 | 47.7085 | 14.5610 | 0.0246 |
| temporal_evaluation | pair_TC_weighted | 15.7714 | 19.6564 | 48.1495 | 14.4477 | 0.0246 |
| temporal_evaluation | ablate_1 | 15.7952 | 19.6785 | 48.1288 | 14.4454 | 0.0259 |
| temporal_evaluation | pair_TC_equal | 15.7993 | 19.6827 | 48.2273 | 14.4335 | 0.0246 |
| temporal_evaluation | pair_SC_weighted | 15.8055 | 19.6850 | 49.9244 | 14.5251 | 0.0078 |
| temporal_evaluation | ablate_0 | 15.8073 | 19.6835 | 49.1486 | 14.4502 | 0.0161 |

## Previously inspected geographic point assessment

| role | model_id | mae | rmse | extreme_mae | gradient_mae | localization_IoU |
| --- | --- | --- | --- | --- | --- | --- |
| geographic_evaluation | ablate_2 | 9.5499 | 12.4278 | 43.8757 | 20.6938 | 0.0016 |
| geographic_evaluation | expert_temporal | 9.5648 | 12.4070 | 42.7855 | 20.6848 | 0.0021 |
| geographic_evaluation | ref_ema_half30 | 9.5648 | 12.4070 | 42.7855 | 20.6848 | 0.0021 |
| geographic_evaluation | temporal_EMA_h30_trend0 | 9.5648 | 12.4070 | 42.7855 | 20.6848 | 0.0021 |
| geographic_evaluation | ref_ema_half120 | 9.5678 | 12.4001 | 41.4632 | 20.4918 | 0.0105 |
| geographic_evaluation | ref_V8_hybrid | 9.5793 | 12.4346 | 42.9260 | 20.7605 | 0.0032 |
| geographic_evaluation | pair_TS_weighted | 9.5795 | 12.4449 | 44.8628 | 20.7417 | 0.0004 |
| geographic_evaluation | expert_quality | 9.5933 | 12.6010 | 42.4565 | 20.9607 | 0.0170 |

## Locked2023 probabilistic results

| role | model_id | calibration | mae | rmse | crps | coverage50 | coverage80 | coverage90 | coverage95 | interval_width90 | energy_score | variogram_score | extreme_mae | event_brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | EMA_heteroscedastic | scale_CRPS | 10.9426 | 13.3766 | 8.0418 | 0.5263 | 0.7816 | 0.8644 | 0.9038 | 45.3252 | 9.6624 | 1.4649 | 37.3125 | 0.0277 |
| validation | cvae_coarse3_seed20261207 | scale_CRPS | 10.8175 | 13.0878 | 8.1081 | 0.4764 | 0.7264 | 0.8071 | 0.8477 | 43.7293 | 9.5771 | 1.3688 | 34.9853 | 0.0268 |
| validation | cvae_coarse3_seed20261207 | raw | 10.8175 | 13.0878 | 8.1081 | 0.4764 | 0.7264 | 0.8071 | 0.8477 | 43.7293 | 9.5771 | 1.3688 | 34.9853 | 0.0268 |
| validation | QuantileBoosting_EMA | raw | 11.1490 | 13.5325 | 8.1292 | 0.5342 | 0.7986 | 0.8804 | 0.9203 | 47.1398 | 9.7807 | 1.4083 | 37.6072 | 0.0277 |
| validation | QuantileBoosting_EMA | scale_CRPS | 11.1490 | 13.5325 | 8.1316 | 0.4227 | 0.6789 | 0.7747 | 0.8303 | 35.3549 | 9.7655 | 1.3126 | 37.6072 | 0.0277 |
| validation | cvae_coarse3_refine_seed20261207 | scale_CRPS | 10.8967 | 13.1561 | 8.1704 | 0.4772 | 0.7289 | 0.8096 | 0.8512 | 44.5910 | 9.6422 | 1.3682 | 34.5277 | 0.0285 |
| validation | cvae_coarse3_refine_seed20261207 | raw | 10.8967 | 13.1561 | 8.1704 | 0.4772 | 0.7289 | 0.8096 | 0.8512 | 44.5910 | 9.6422 | 1.3682 | 34.5277 | 0.0285 |
| validation | cvae_coarse3_seed20261206 | raw | 10.8352 | 13.1477 | 8.1718 | 0.4819 | 0.7264 | 0.8060 | 0.8478 | 43.3840 | 9.6478 | 1.3758 | 35.2755 | 0.0261 |

## Old temporal probabilistic assessment

| role | model_id | calibration | mae | rmse | crps | coverage50 | coverage80 | coverage90 | coverage95 | interval_width90 | energy_score | variogram_score | extreme_mae | event_brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal_evaluation | hybrid_Gaussian | scale_CRPS | 16.3020 | 20.3696 | 12.6054 | 0.4751 | 0.7124 | 0.7901 | 0.8312 | 55.5415 | 15.3688 | 2.6887 | 49.3191 | 0.0980 |
| temporal_evaluation | EMA_heteroscedastic | scale_CRPS | 16.4541 | 20.4730 | 12.6360 | 0.4688 | 0.7148 | 0.7973 | 0.8395 | 50.3657 | 15.4357 | 2.6176 | 48.8690 | 0.0770 |
| temporal_evaluation | QuantileBoosting_EMA | raw | 16.4586 | 20.4848 | 12.6659 | 0.4944 | 0.7405 | 0.8191 | 0.8601 | 53.1547 | 15.5295 | 2.6159 | 48.2603 | 0.0939 |
| temporal_evaluation | EMA_heteroscedastic | raw | 16.4541 | 20.4730 | 12.7313 | 0.5796 | 0.8144 | 0.8769 | 0.9082 | 67.1543 | 15.4804 | 2.7485 | 48.8690 | 0.0848 |
| temporal_evaluation | EMA_Gaussian | scale_CRPS | 16.5174 | 20.5646 | 12.7738 | 0.4626 | 0.7013 | 0.7800 | 0.8216 | 53.5345 | 15.5567 | 2.6765 | 48.8529 | 0.0927 |
| temporal_evaluation | EMA_student | raw | 16.5841 | 20.5270 | 12.8062 | 0.4041 | 0.6578 | 0.7611 | 0.8219 | 51.7799 | 15.4463 | 2.5139 | 48.5121 | 0.0677 |
| temporal_evaluation | EMA_student | scale_CRPS | 16.5841 | 20.5270 | 12.8062 | 0.4041 | 0.6578 | 0.7611 | 0.8219 | 51.7799 | 15.4463 | 2.5139 | 48.5121 | 0.0677 |
| temporal_evaluation | QuantileBoosting_EMA | scale_CRPS | 16.4586 | 20.4848 | 12.8209 | 0.3905 | 0.6280 | 0.7177 | 0.7686 | 39.8661 | 15.7458 | 2.5864 | 48.2603 | 0.0838 |

## Old geographic probabilistic assessment

| role | model_id | calibration | mae | rmse | crps | coverage50 | coverage80 | coverage90 | coverage95 | interval_width90 | energy_score | variogram_score | extreme_mae | event_brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geographic_evaluation | QuantileBoosting_EMA | scale_CRPS | 9.9571 | 12.8587 | 7.5018 | 0.4451 | 0.6932 | 0.7830 | 0.8298 | 29.9895 | 9.4358 | 1.7228 | 43.5569 | 0.0002 |
| geographic_evaluation | EMA_student | raw | 9.7659 | 12.6277 | 7.5062 | 0.4196 | 0.6644 | 0.7670 | 0.8293 | 33.0654 | 9.4247 | 1.7101 | 42.7169 | 0.0011 |
| geographic_evaluation | EMA_student | scale_CRPS | 9.7659 | 12.6277 | 7.5062 | 0.4196 | 0.6644 | 0.7670 | 0.8293 | 33.0654 | 9.4247 | 1.7101 | 42.7169 | 0.0011 |
| geographic_evaluation | QuantileBoosting_EMA | raw | 9.9571 | 12.8587 | 7.5090 | 0.5549 | 0.8012 | 0.8726 | 0.9080 | 39.9860 | 9.3335 | 1.8675 | 43.5569 | 0.0002 |
| geographic_evaluation | EMA_heteroscedastic | scale_CRPS | 9.9451 | 12.8565 | 7.5633 | 0.5235 | 0.7632 | 0.8367 | 0.8746 | 38.2785 | 9.4315 | 1.9729 | 43.0853 | 0.0002 |
| geographic_evaluation | EMA_Gaussian | scale_CRPS | 9.8817 | 12.7785 | 7.5930 | 0.4693 | 0.7026 | 0.7803 | 0.8225 | 33.7000 | 9.5150 | 1.8915 | 43.0665 | 0.0002 |
| geographic_evaluation | EMA_structured | scale_CRPS | 9.7966 | 12.6573 | 7.6054 | 0.4275 | 0.6510 | 0.7312 | 0.7761 | 28.8464 | 9.6132 | 1.6952 | 42.9217 | 0.0004 |
| geographic_evaluation | EMA_structured | raw | 9.7966 | 12.6573 | 7.6299 | 0.5260 | 0.7508 | 0.8188 | 0.8561 | 38.4618 | 9.4952 | 1.7193 | 42.9217 | 0.0011 |

## Neural training seed stability

| role | configuration | calibration | training_seeds | CRPS_mean | CRPS_std | coverage90 |
| --- | --- | --- | --- | --- | --- | --- |
| geographic_evaluation | cvae_coarse3 | raw | 3 | 8.0026 | 0.0108 | 0.7235 |
| geographic_evaluation | cvae_coarse3 | scale_CRPS | 3 | 8.0026 | 0.0108 | 0.7235 |
| geographic_evaluation | cvae_coarse3_refine | raw | 3 | 8.2165 | 0.1446 | 0.7196 |
| geographic_evaluation | cvae_coarse3_refine | scale_CRPS | 3 | 8.2165 | 0.1446 | 0.7196 |
| geographic_evaluation | flow_coarse0_refine | raw | 3 | 8.2505 | 0.2833 | 0.8410 |
| geographic_evaluation | flow_coarse0_refine | scale_CRPS | 3 | 8.2171 | 0.3235 | 0.7478 |
| geographic_evaluation | flow_coarse1_refine | raw | 3 | 8.0088 | 0.0958 | 0.8614 |
| geographic_evaluation | flow_coarse1_refine | scale_CRPS | 3 | 7.8793 | 0.1210 | 0.7798 |
| temporal_evaluation | cvae_coarse3 | raw | 3 | 13.0056 | 0.1616 | 0.7294 |
| temporal_evaluation | cvae_coarse3 | scale_CRPS | 3 | 13.0056 | 0.1616 | 0.7294 |
| temporal_evaluation | cvae_coarse3_refine | raw | 3 | 13.0711 | 0.1775 | 0.7262 |
| temporal_evaluation | cvae_coarse3_refine | scale_CRPS | 3 | 13.0711 | 0.1775 | 0.7262 |
| temporal_evaluation | flow_coarse0_refine | raw | 3 | 13.6283 | 0.3168 | 0.8307 |
| temporal_evaluation | flow_coarse0_refine | scale_CRPS | 3 | 13.5889 | 0.3149 | 0.7446 |
| temporal_evaluation | flow_coarse1_refine | raw | 3 | 13.1937 | 0.0436 | 0.8642 |
| temporal_evaluation | flow_coarse1_refine | scale_CRPS | 3 | 12.9761 | 0.0339 | 0.7868 |
| validation | cvae_coarse3 | raw | 3 | 8.1725 | 0.0647 | 0.8092 |
| validation | cvae_coarse3 | scale_CRPS | 3 | 8.1725 | 0.0647 | 0.8092 |
| validation | cvae_coarse3_refine | raw | 3 | 8.2298 | 0.0631 | 0.8058 |
| validation | cvae_coarse3_refine | scale_CRPS | 3 | 8.2298 | 0.0631 | 0.8058 |
| validation | flow_coarse0_refine | raw | 3 | 9.0032 | 0.1770 | 0.8958 |
| validation | flow_coarse0_refine | scale_CRPS | 3 | 8.6285 | 0.1962 | 0.8172 |
| validation | flow_coarse1_refine | raw | 3 | 9.1023 | 0.0654 | 0.9193 |
| validation | flow_coarse1_refine | scale_CRPS | 3 | 8.5147 | 0.0558 | 0.8480 |


Full scores, all attempted configurations/failures, active-parameter caveats, exact folds, locked candidates, source/configuration hashes, checkpoints and reusable fit/predict functions are preserved. No Stage2 experiments or new seismic results exist in V9.

## Quantitative interpretation

The internal conventional champion remains SVR_0. Its locked2023 MAE is 10.589mm versus frozen quality 10.522mm, so conventional ML did not beat that strong simple control on this cohort. The descriptively lowest four-branch MAE is 10.413mm, a 1.04% macro-volcano difference. This tiny difference is not a promotion result. Frame/block-weighted estimates use a different estimand and do not establish an improvement after multiple-comparison adjustment; see diagnostics/paired_cluster_uncertainty.csv.

The lowest descriptive calibrated-distribution CRPS is EMA_heteroscedastic / scale_CRPS: 8.042mm with90% interval coverage 86.4%. Neural config means across all three training seeds are below; selecting a best seed would overstate performance.

| role | configuration | calibration | training_seeds | CRPS_mean | CRPS_std | coverage90 |
| --- | --- | --- | --- | --- | --- | --- |
| validation | cvae_coarse3 | scale_CRPS | 3 | 8.1725 | 0.0647 | 0.8092 |
| validation | cvae_coarse3_refine | scale_CRPS | 3 | 8.2298 | 0.0631 | 0.8058 |
| validation | flow_coarse0_refine | scale_CRPS | 3 | 8.6285 | 0.1962 | 0.8172 |
| validation | flow_coarse1_refine | scale_CRPS | 3 | 8.5147 | 0.0558 | 0.8480 |

Whole-volcano, geographic-region and historical-rate/turning breakdowns are in diagnostics/point_by_*.csv and probability_by_*.csv. Regime thresholds use only training histories. Frozen V7/V8 CVAE/flow and Gaussian references are separately preserved in diagnostics/frozen_probability_reference_summary.csv and its case table: these are unchanged archived scores with original sampling/calibration, not new V9 reruns or model-selection evidence. CNN seed variation is in diagnostics/cnn_seed_stability.csv.
