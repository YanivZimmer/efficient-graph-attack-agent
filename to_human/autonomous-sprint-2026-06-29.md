# Autonomous Research Sprint — 2026-06-29

## Headline

The best full-slice fair AIT-ADS result remains cluster F1 `0.9363` for the multiview temporal-episode family. After chronological validation, the best holdout-mean method is `multiview_hgat_v2 + temporal_only_macro_elbow_floor4`, with mean F1 `0.9588` across four time folds.

This preserves the full-slice `0.9363` result, beats fixed `4h` on holdout mean (`0.9588` vs `0.9515`), and fixes the unconstrained macro-elbow instability (`0.7692` holdout mean).

## What Improved

- Added `4h` temporal episode strategies to the clustering ablation grid.
- Confirmed that `dbscan_split_time_4h` and `temporal_only_4h` are identical for `multiview_hgat_v2`.
- Added a weak-label-only `macro_elbow` estimator that ignores micro-burst gaps and detects prominent macro-gap jumps.
- Confirmed that `temporal_only_macro_elbow` matches the fixed `4h` result for `multiview_hgat_v2`.
- Added chronological 4-fold validation.
- Found unconstrained macro-elbow over-fragments late folds.
- Added `macro_elbow_floor4`, which uses a minimum macro-gap floor to avoid chasing small local gaps.
- This means DBSCAN is unnecessary for the current best AIT-ADS incident construction; HGAT is acting as a triage/candidate selector, while temporal episode segmentation forms incidents. The temporal floor is still an empirical prior that needs external validation or data-driven estimation.

## What Failed

- Adaptive quantile gaps did not robustly infer incident scale. On `multiview_hgat_v2`, q90/q95 temporal splitting reached only F1 `0.5764`.
- Raw Bayesian-block event segmentation over-fragmented incidents into roughly `1000+` micro-clusters. It achieved very high precision but near-zero recall, with best multiview F1 `0.0195`.
- Unconstrained macro-elbow looked strong on the full slice but failed on chronological holdout, dropping to mean F1 `0.7692`.

## Research Interpretation

The strongest story is shifting from "better HGAT clustering embeddings" to "weakly supervised alert triage plus robust temporal episode priors." The cleanest current claim is not that the temporal prior is fully learned, but that temporal episode scale is the dominant bottleneck and a robust macro-gap floor improves chronological stability.

The important caveat: the `4h` floor is still an empirical prior. The next paper-quality step is inferring that floor from data or testing whether it transfers to non-AIT datasets.

## Best Next Experiments

- Infer the minimum macro-gap floor without incident labels.
- Validate `macro_elbow_floor4` on another dataset.
- Cache cluster-evaluation pair structures so validation sweeps are cheap enough for multi-seed runs.
- Turn the Bayesian-block failure into constrained segmentation by enforcing minimum episode duration/size or learning adjacent-block merges.
- If the `4h` floor fails out of sample, learn the gap with a stability objective rather than direct incident labels.
