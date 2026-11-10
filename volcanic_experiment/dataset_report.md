# Dataset, processing, and label audit

Primary source: Fabien Albino, Datacubes of InSAR time series for active volcanoes, Zenodo7213408, DOI10.5281/zenodo.7213408, CC-BY4.0. Metadata and file checksums are cached. https://zenodo.org/records/7213408

Related paper: Biggs etal2022, Bulletin of Volcanology84:100, DOI10.1007/s00445-022-01608-x. https://link.springer.com/article/10.1007/s00445-022-01608-x . The archive contains geocoded LOS estimates and deformation/artifact examples; it is not a randomly sampled global monitoring cohort. Inversion uses retrospective records. Its Galapagos example has documented severe unwrapping limitations during large deformation. Artifact-prone examples remain included as contrast cases (not independently verified quiet controls), not excluded based on test errors. The archive README cites a different journal than the actual DOI; source metadata and the DOI are retained rather than silently reconciling this discrepancy.

## Measured source metadata

`data/dataset_inventory.csv` records all 16 region centres, timestamps, true cadence/gaps, LOS units, view geometry parsed from frame IDs, WGS84 pixel dimensions and source references. All cubes use0.001degree latitude/longitude increments, with metres per longitude pixel varying with latitude. The first source epoch is identically zero, consistent with a temporal reference; the physical spatial reference/sign convention is incompletely documented. No LOS-to-vertical conversion is made. Native32/64/128/256 nested squares are extracted without interpolation; `data/crop_footprints.csv` gives their cell-edge dimensions and areas. Full source and crop coverage are distinct.

The NetCDF files have no explicit coherence/missingness mask and encode unsupported regions as exact zeros. The raw cache preserves original values. The loader treats exact zeros as missing per acquisition, including the first reference epoch. This conservative heuristic excludes actual zero-valued observations too; validity fractions describe this rule, not producer-certified coherence. No full-record static mask is fed to models. Support is inferred from each origin's past alone; the target validity mask is scoring-only. Future validity, future image values, full-record velocity and whole-record quality fields never reach predictors. No claim of resolving upstream processing leakage is made.

## Selected regions and splits

| name | role | group | n_acquisitions | start | end | median_gap_days | max_gap_days | spacing_y_m | spacing_x_m | valid_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agung | test | bali_lombok_frame | 187 | 2015-11-19 | 2020-07-25 | 6.000 | 48 | 110.598 | 110.149 | 0.853 |
| campi_flegrei | validation | campi_flegrei | 233 | 2015-11-07 | 2020-07-07 | 6.000 | 24 | 111.051 | 84.356 | 0.522 |
| corbetti | train | corbetti | 115 | 2015-12-17 | 2020-07-11 | 12.000 | 48 | 110.592 | 110.453 | 0.910 |
| domuyo | validation | chile_083D | 179 | 2015-12-17 | 2020-10-27 | 6.000 | 48 | 110.971 | 89.431 | 0.980 |
| erta_ale | train | erta_ale | 119 | 2015-11-23 | 2020-09-21 | 12.000 | 48 | 110.636 | 108.219 | 0.992 |
| etna | train | etna | 228 | 2015-11-02 | 2020-10-30 | 6.000 | 66 | 110.992 | 88.133 | 0.992 |
| fujisan | test | fujisan | 125 | 2015-11-20 | 2020-09-06 | 12.000 | 96 | 110.947 | 90.887 | 0.964 |
| galapagos | test | galapagos | 182 | 2015-11-14 | 2020-06-20 | 6.000 | 48 | 110.575 | 111.308 | 0.796 |
| kilauea | test | kilauea | 222 | 2015-11-11 | 2020-10-27 | 6.000 | 24 | 110.697 | 105.025 | 0.932 |
| kverkfjoll | train | kverkfjoll | 163 | 2015-11-18 | 2020-10-16 | 12.000 | 72 | 111.487 | 47.788 | 0.356 |
| laguna_del_maule | validation | chile_083D | 174 | 2015-12-17 | 2020-10-27 | 6.000 | 108 | 110.960 | 90.096 | 0.917 |
| lawu | train | lawu | 217 | 2015-11-14 | 2020-08-31 | 6.000 | 48 | 110.594 | 110.341 | 0.968 |
| reykjanes | test | reykjanes | 219 | 2016-06-07 | 2020-12-31 | 6.000 | 36 | 111.475 | 49.196 | 0.627 |
| rinjani | test | bali_lombok_frame | 188 | 2015-11-19 | 2020-07-25 | 6.000 | 48 | 110.598 | 110.127 | 0.832 |
| taal | test | taal | 141 | 2015-11-19 | 2020-10-23 | 12.000 | 48 | 110.639 | 108.034 | 0.529 |
| tambora | train | tambora | 54 | 2015-11-26 | 2018-06-07 | 12.000 | 48 | 110.597 | 110.175 | 0.346 |

All 16 archive files are retained in assigned roles. The Galapagos multi-volcano frame is represented by one declared Sierra Negra centre; other overlapping systems are not independent test regions. Agung and Rinjani share a source frame and remain together, as do Domuyo and Laguna del Maule. Whole volcano/frame geographic roles and chronology are declared in `protocol.json`. Tambora is retained for older calibration despite ending before 2019; it cannot contribute a2019+ test. No performance-based dataset exclusions occur. Kverkfjoll32 has no supported pixels and is not scored; Laguna del Maule32 has only about 1.8% supported pixels, so its small-crop numbers are fragile. These are documented support limitations, not zero-error forecasts.

## Sources investigated but excluded from primary modelling

- Existing Kagoshima/Beppu/Aso/Kirishima city footprints: requested summit coordinates outside old arrays; `existing_footprint_audit.csv` and `repository_audit.md` document checks.
- USGS Mauna Loa2015–2024 (DOI10.5066/P13ZFTSX): documented open data, but different time segments/processors/projections and filtered products would require another harmonization study. https://www.usgs.gov/data/insar-time-series-data-mauna-loa-hawaii-2015-2024
- LiCSBAS Campi Flegrei tutorial: downloaded unfiltered cumulative HDF5 for format compatibility, but narrower121×191 support, alternate viewing track and overlapping volcano make it supplementary only; never stitched to the primary cube. https://github.com/yumorishita/LiCSBAS/wiki
- Single-epoch/velocity-only products and fused GNSS records do not supply a comparable radar-only displacement sequence for this experiment.

## Activity annotations

Annotations and source URLs are in `data/activity_annotations.json`. USGS independently documents Kilauea2018 activity and NORMAL status from March26,2019; the background window is conservatively limited to publication month. Indonesian authorities document Agung's high-alert period in 2017. NAST/PHIVOLCS records independently document Taal's low-level unrest from March28,2019 and January2020 eruption; the pre-eruption and eruption phases remain one episode. Unlabelled dates are unknown, never automatically quiet. These few labels support descriptions, not a validated activity classifier. Labels at origin and target are stored separately; neither is a forecasting feature.

A second narrow-label sensitivity excludes Taal's long low-alert phase. Results cannot disentangle volcano, processing and activity effects. Atmospheric Fuji/Agung/Lawu, tectonic Rinjani and ice-related Kverkfjoll patterns require physical caution; anomalies are not automatically volcanic.

Full source cell-edge bounds and approximate WGS84 full-frame areas are separately recorded in `data/full_source_footprints.csv`; crop coverage should not be confused with the whole source frame.
