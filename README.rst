TECTONIC — six deformation forecasting models
=============================================

This repository contains the source, frozen checkpoints, compact evaluation data,
forecast samples and results for six selected signed line-of-sight (LOS)
deformation models. Files are stored in ordinary Git: **Git LFS is not required**.

.. list-table:: Reported results
   :header-rows: 1

   * - Model
     - Metric and cohort
     - Result
   * - Quality-aware smoothing
     - MAE, V9 2023 validation
     - 10.522 mm
   * - EMA30
     - MAE, V9 2023 validation
     - 10.550 mm
   * - Robust rank-3 low-rank
     - MAE, original 64x64 held-out volcanic evaluation
     - 14.432 mm
   * - Four-branch hybrid (four_horizon)
     - MAE, V9 2023 validation
     - 10.413 mm
   * - CVAE (cvae_coarse3, three seeds)
     - CRPS, V9 2023 validation
     - 8.172 mm
   * - Persistence control
     - MAE, original 64x64 held-out volcanic evaluation
     - 15.672 mm

These are different evaluation populations and metrics. CRPS and MAE are not
interchangeable, and the original 64x64 scores cannot be ranked against the V9
32x32 results. V9 assessments were previously inspected; the small descriptive
hybrid gain does not constitute independent confirmation or model promotion.
Historical map availability at issue time remains unverified.

Clone and reproduce
-------------------

The repository is private; use a GitHub account with repository access.
The recorded environment is Linux with Python 3.13. From a fresh clone:

.. code-block:: bash

   git clone https://github.com/EEsc05/TECTONIC.git
   cd TECTONIC
   python3.13 -m venv .venv
   source .venv/bin/activate
   python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -r selected_models/requirements-lock.txt
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m selected_models.reproduce

The replay verifies file hashes, reproduces the reported point MAEs and CVAE
CRPS from the original samples, recomputes smoothing and original low-rank
forecasts, and exercises frozen hybrid and CVAE checkpoint inference. It does
not train new models or rerun the original model search. Results and timings are
written under ``selected_models/generated/``. A shorter smoke test is available
with ``--quick``; its partial-cohort numbers are not full evaluation results.

See `selected_models/README.md <selected_models/README.md>`_ for the model/file
map, provenance, limitations and original reports. ``selected_models/provenance.json``
identifies original files and derived subsets; ``file_manifest.json`` records
checksums of the published files. Shared historical modules are included only
as dependencies or source evidence; their original full experiment runners
are not the reproduction entry point for this selected bundle.

Other experiments, installed environments, redundant raw archives and feature
caches are intentionally outside this Git tree. They remain preserved in the
`complete workspace backup <https://github.com/EEsc05/TECTONIC/releases/tag/tectonic-complete-backup-2026-11-09>`_.
No Git LFS allowance or paid storage is needed for this selected repository.
