# V9 model optimization

**Stage2 suspended; Stage1 only.** Read [final_summary.md](final_summary.md), [prior optimization audit](previous_hyperparameter_audit.md), model_search_space.json and search_history.csv. Everything new is isolated here; source snapshots are in code/.

Resume from repository root:
```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 overnight/.venv/bin/python -u -m volcanic_forecasting_v2.src.research_v9.run
```

Scripts: volcanic_forecasting_v2/src/research_v9/. Model-level search caches and phase statuses preserve completed work. Prediction interfaces are conventional.predict, spatial.predict, temporal.forecast, neural.cnn_predict/decode on the common versioned benchmark arrays. They are research interfaces; no arbitrary sensor/location/horizon deployment is validated. Unsupported sensors/history/horizons retain prior abstention limits. No prior experiment was modified.
