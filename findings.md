# Research Findings

## Research Question

Can weakly supervised alert-entity graph models discover coherent security incidents from alert-level labels alone, and how close can they get to rule-based and incident-supervised baselines?

## Current Understanding

The repo already supports a meaningful research story: learned graph models can separate malicious from benign alerts well, but weakly supervised incident discovery is currently bottlenecked by clustering quality rather than raw alert discrimination. On AIT-ADS, multiple weakly supervised methods achieve very high node-level AUC/F1, yet they mostly collapse predicted-malicious alerts into one large cluster. That yields high cluster recall but poor precision. In contrast, the rule-based GraphWeaver baseline fragments alerts into many tighter clusters, and the incident-supervised GRAIN-style baseline produces the best cluster F1.

On the primary tenant dataset, the picture is less stable. There is at least one strong standalone HGAT result, but side-by-side baseline comparison runs vary substantially. That makes the current evidence suggestive rather than publication-ready.

The latest AIT-ADS ablations materially changed the story. The strongest fair result is no longer "HGAT plus DBSCAN"; it is `multiview_hgat_v2` for alert triage followed by temporal episode segmentation. On the full AIT-ADS slice, macro-elbow temporal splitting, macro-elbow with a `4h` floor, and fixed `4h` splitting all reach cluster F1 `0.9363`. Chronological holdout is more revealing: unconstrained macro-elbow is unstable, while `macro_elbow_floor4` reaches the best mean holdout F1 (`0.9588`) and fixed `4h` remains the strongest simple baseline (`0.9515`).

## Key Results

- Primary standalone HGAT run: AUC `0.9945`, F1 `0.9032`, one 80-alert cluster in [outputs/benchmarks/primary/evaluation_report.json](/Users/yanivzimmer/Code/efficient-graph-attack-agent/outputs/benchmarks/primary/evaluation_report.json).
- Primary comparison runs: HGAT ranges from strong AUC with zero F1 to weaker but still high AUC in [outputs/baseline_comparison_full/primary_baseline_comparison.json](/Users/yanivzimmer/Code/efficient-graph-attack-agent/outputs/baseline_comparison_full/primary_baseline_comparison.json) and [outputs/baseline_smoke/primary_baseline_comparison.json](/Users/yanivzimmer/Code/efficient-graph-attack-agent/outputs/baseline_smoke/primary_baseline_comparison.json).
- AIT-ADS weakly supervised HGAT: cluster precision `0.2537`, recall `0.9991`, F1 `0.4046` in [outputs/baseline_comparison_full/ait_ads/hgat/evaluation_report.json](/Users/yanivzimmer/Code/efficient-graph-attack-agent/outputs/baseline_comparison_full/ait_ads/hgat/evaluation_report.json).
- AIT-ADS GraphWeaver: cluster precision `0.7838`, recall `0.6579`, F1 `0.6533` in [outputs/baseline_comparison_full/ait_ads/graphweaver/evaluation_report.json](/Users/yanivzimmer/Code/efficient-graph-attack-agent/outputs/baseline_comparison_full/ait_ads/graphweaver/evaluation_report.json).
- AIT-ADS GRAIN-style supervised upper bound: cluster precision `0.8184`, recall `0.9995`, F1 `0.8867` in [outputs/baseline_comparison_full/ait_ads/grain/evaluation_report.json](/Users/yanivzimmer/Code/efficient-graph-attack-agent/outputs/baseline_comparison_full/ait_ads/grain/evaluation_report.json).
- AIT-ADS best fair weakly supervised result: `multiview_hgat_v2 + temporal_only_macro_elbow`, `temporal_only_4h`, `dbscan_split_time_macro_elbow`, and `dbscan_split_time_4h` all reach precision `0.9517`, recall `0.9392`, F1 `0.9363` in [cluster_ablation_macro_elbow_ait_ads_e6](/Users/yanivzimmer/Code/efficient-graph-attack-agent/experiments/h1-incident-aware-hgat/results/cluster_ablation_macro_elbow_ait_ads_e6/ablation_summary.json).
- AIT-ADS chronological holdout: `multiview_hgat_v2 + temporal_only_macro_elbow_floor4` reaches mean F1 `0.9588`, while fixed `4h` reaches `0.9515` and unconstrained macro-elbow reaches only `0.7692` in [temporal_holdout_macro_elbow_floor4_ait_ads_e6](/Users/yanivzimmer/Code/efficient-graph-attack-agent/experiments/h1-incident-aware-hgat/results/temporal_holdout_macro_elbow_floor4_ait_ads_e6/temporal_holdout_summary.json).
- AIT-ADS Bayesian-block segmentation: best multiview row reaches only F1 `0.0195` despite precision `0.9923`, because unconstrained event-rate segmentation fragments incidents into roughly `1000+` micro-clusters.

## Patterns and Insights

- Weakly supervised methods appear much better at triage than at incident grouping.
- Current post-processing dominates downstream clustering quality, but the encoder still matters because temporal episode segmentation only produces the best result for the `multiview_hgat_v2` candidate set.
- The best current incident construction policy is temporal episode segmentation, not density clustering over learned embeddings.
- Rule-based grouping remains surprisingly competitive when the main failure mode is over-merging.
- Raw adaptive gap quantiles, unconstrained macro-elbow, and raw Bayesian-block segmentation can all over-fragment by modeling telemetry micro-bursts or local gaps. A minimum macro-gap floor is currently needed for chronological robustness.

## Lessons and Constraints

- Do not treat near-perfect node classification as evidence of strong incident discovery.
- Separate evaluation of triage quality and cluster formation is necessary.
- Primary-dataset claims should be framed cautiously until run-to-run variance is explained.
- Incident-supervised baselines are useful as upper bounds, but they are not apples-to-apples comparisons.
- `macro_elbow_floor4` improves chronological robustness, but the `4h` floor is still an empirical prior. It should be inferred from data or tested on other datasets before becoming a headline deployment claim.

## Open Questions

- Can we infer the minimum macro-gap floor from unlabeled/weakly labeled data instead of setting it to `4h`?
- Does the `4h` floor transfer to non-AIT datasets?
- Why does `multiview_hgat_v2` pair so well with `4h` temporal episodes while saved HGAT does not?
- Does heterogeneous structure help mainly at triage time, candidate calibration time, or representation alignment time?
- What causes the primary-dataset instability across runs with nominally similar settings?

## Optimization Trajectory

Current fair weakly supervised best on the full AIT-ADS slice is cluster F1 `0.9363` for the multiview temporal-episode family. On four chronological folds, `multiview_hgat_v2 + temporal_only_macro_elbow_floor4` has the best mean F1 (`0.9588`). The next optimization step should infer or validate the `4h` floor rather than tuning more post-hoc splitters on the same slice.
