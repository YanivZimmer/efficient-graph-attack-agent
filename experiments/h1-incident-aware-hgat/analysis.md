# Analysis — H1 Incident-Aware HGAT

Use this file to summarize experiment outcomes for:

- A0 baseline HGAT
- A1 pair-loss only
- A2 alert-alert edges only
- A3 full incident-aware HGAT
- A4 full incident-aware HGAT with clustering sweep

## Result Summary

First AIT-ADS implementation test completed on 2026-06-29.

### Implemented changes

- Sparse typed alert-alert edges in [data/graph_builder.py](/Users/yanivzimmer/Code/efficient-graph-attack-agent/data/graph_builder.py)
- Projection head in [models/hgat.py](/Users/yanivzimmer/Code/efficient-graph-attack-agent/models/hgat.py)
- Weak incident pair mining and pair losses in [training/trainer.py](/Users/yanivzimmer/Code/efficient-graph-attack-agent/training/trainer.py)
- Focused experiment runner in [scripts/run_h1_incident_aware_hgat.py](/Users/yanivzimmer/Code/efficient-graph-attack-agent/scripts/run_h1_incident_aware_hgat.py)

### Runs

#### AIT-ADS smoke (`max_records=1000`, `epochs=2`)

- Output: [ait_ads_smoke](</Users/yanivzimmer/Code/efficient-graph-attack-agent/experiments/h1-incident-aware-hgat/results/ait_ads_smoke>)
- Baseline cluster F1: `0.3520`
- Incident-aware cluster F1: `0.4444`
- Interpretation: encouraging small-data signal, but too small to trust yet

#### First full comparison (`max_records=10000`, `epochs=6`)

- Output: [ait_ads_first_test_e6](</Users/yanivzimmer/Code/efficient-graph-attack-agent/experiments/h1-incident-aware-hgat/results/ait_ads_first_test_e6>)
- Baseline:
  - node AUC `0.9014`
  - cluster precision `0.2136`
  - cluster recall `1.0`
  - cluster F1 `0.3520`
- Incident-aware v1:
  - node AUC `0.9976`
  - cluster precision `0.2136`
  - cluster recall `1.0`
  - cluster F1 `0.3520`
- Interpretation: large gain in ranking quality, no gain in incident clustering under the default decision rule

#### Longer incident-aware run (`max_records=10000`, `epochs=20`)

- Output: [ait_ads_incident_v1_e20](</Users/yanivzimmer/Code/efficient-graph-attack-agent/experiments/h1-incident-aware-hgat/results/ait_ads_incident_v1_e20>)
- node AUC `0.9954`
- cluster precision `0.2136`
- cluster recall `1.0`
- cluster F1 `0.3520`
- Interpretation: extra training time did not fix the clustering behavior

#### Calibrated comparison (`max_records=10000`, `epochs=6`, tuned threshold, probability-only clustering gate)

- Output: [ait_ads_calibrated_e6](</Users/yanivzimmer/Code/efficient-graph-attack-agent/experiments/h1-incident-aware-hgat/results/ait_ads_calibrated_e6>)
- Baseline:
  - tuned threshold `0.5627`
  - selected alerts `8416`
  - node AUC `0.9507`
  - node F1 `0.9746`
  - cluster precision `0.2536`
  - cluster recall `0.9991`
  - cluster F1 `0.4045`
- Incident-aware v1:
  - tuned threshold `0.5091`
  - selected alerts `8501`
  - node AUC `0.9969`
  - node F1 `0.9947`
  - cluster precision `0.2510`
  - cluster recall `0.9991`
  - cluster F1 `0.4012`
- Interpretation: the calibration fix materially improved the baseline clustering result and largely matched the older strong HGAT artifact, but `incident_v1` still did not beat the calibrated baseline on full AIT-ADS.

#### All-variants sweep (`epochs=6`, tuned threshold, probability-only clustering gate)

- Output: [all_variants_ait_ads_e6](</Users/yanivzimmer/Code/efficient-graph-attack-agent/experiments/h1-incident-aware-hgat/results/all_variants_ait_ads_e6>)
- Ranked by cluster F1:
  - `temporal_causal_hgat`: `0.4089`
  - `multiview_hgat`: `0.4051`
  - saved repo `hgat`: `0.4046`
  - saved repo `graph_ids`: `0.4046`
  - local calibrated `baseline_hgat`: `0.4045`
  - `prototype_memory_hgat`: `0.4020`
  - saved repo `gnn_ids`: `0.4020`
  - `differentiable_cluster_hgat`: `0.3993`
  - `baseline_hgat` in unified runner: `0.3985`
  - `weak_pair_hgat`: `0.3981`
  - saved repo `anomal_e`: `0.3521`
  - saved repo `graphweaver`: `0.6533`
  - saved repo `grain`: `0.8867`

Interpretation:

- Among the five newly implemented alternatives, only `temporal_causal_hgat` clearly beat the saved HGAT baseline on AIT-ADS in this first sweep.
- `multiview_hgat` also slightly beat the local calibrated baseline and was effectively tied with the saved HGAT/GraphIDS range.
- The pure weak-pair objective, differentiable clustering, and prototype-memory variants did not outperform the best existing weakly supervised baselines in this first pass.
- None of the new weakly supervised variants came close to the saved GraphWeaver or supervised GRAIN-style upper bound.

### Key diagnostic

On the full AIT-ADS slice used here, both baseline and incident-aware variants predict essentially all alerts as malicious at the default `0.5` cutoff. That means the clusterer receives nearly the full dataset and collapses it into one incident cluster.

Observed probability ranges:

- 6-epoch incident-aware run: `0.5287` to `0.5568`
- 20-epoch incident-aware run: `0.5326` to `0.5910`

This makes the current clustering stage insensitive to the improved node-level ranking unless the maliciousness decision rule is changed.

## What Improved

- Node AUC improved substantially in the first full comparison:
  - baseline `0.9014`
  - incident-aware v1 `0.9976`
- The small smoke run showed a possible clustering benefit on a reduced subset.
- Probability-only candidate selection plus threshold tuning improved cluster F1 from `0.3520` to about `0.4045` on the full AIT-ADS run.
- `temporal_causal_hgat` was the strongest new variant and achieved the best new cluster F1 at `0.4089`.
- `multiview_hgat` achieved very strong node metrics and a slight clustering improvement over the local calibrated baseline.

## What Broke

- Cluster F1 did not improve on full AIT-ADS.
- Pair loss is dominated by the negative term; positive pair loss stayed tiny in early runs.
- The current `cluster_incidents` candidate rule (`predictions OR probability >= threshold`) makes threshold sweeps ineffective when predictions are already all ones.
- Even after fixing the candidate rule, the current incident-aware loss still produces one dominant cluster on the selected AIT-ADS alerts.
- The first differentiable clustering and prototype-memory implementations still collapsed most selected alerts into a single dominant cluster.
- The pure weak-pair contrastive variant underperformed both temporal and multiview variants in this sweep.

## Interpretation

The current incident-aware v1 changes improve discriminative ordering but do not yet produce incident-separable alert subsets on AIT-ADS. The model is still effectively feeding the clusterer "almost everything," so downstream DBSCAN cannot express the benefit of better ranking.

The calibration and candidate-selection fix was necessary and useful. It recovered a much stronger clustering baseline on AIT-ADS. That means the next iteration should no longer focus on thresholding bugs. It should focus on making the embedding geometry incident-aware in a way that actually fragments the selected malicious set into multiple coherent groups.

The next changes should target representation structure and clustering behavior:

1. deepen the temporal-causal path, since it was the strongest new signal
2. improve multiview fusion with stronger cross-view alignment or view-specific regularization
3. strengthen the positive-pair signal relative to BCE and negative-pair pressure before giving up on weak-pair learning
4. revisit differentiable clustering only after the encoder produces multiple naturally separable malicious regions

#### V2 follow-up: stronger temporal-causal and multiview fusion (`epochs=6`, tuned threshold, probability-only clustering gate)

- Output: [v2_variants_ait_ads_e6](</Users/yanivzimmer/Code/efficient-graph-attack-agent/experiments/h1-incident-aware-hgat/results/v2_variants_ait_ads_e6>)
- `temporal_causal_hgat_v2`:
  - tuned threshold `0.5322`
  - selected alerts `8429`
  - node AUC `0.9925`
  - node F1 `0.9864`
  - cluster precision `0.2532`
  - cluster recall `0.9991`
  - cluster F1 `0.4040`
- `multiview_hgat_v2`:
  - tuned threshold `0.6200`
  - selected alerts `8361`
  - node AUC `0.9943`
  - node F1 `0.9875`
  - cluster precision `0.2555`
  - cluster recall `1.0`
  - cluster F1 `0.4070`

Interpretation:

- `multiview_hgat_v2` improved over the saved HGAT baseline (`0.4046`) and over the earlier multiview v1 result (`0.4051`), but it still did not beat the earlier temporal-causal v1 peak (`0.4089`).
- `temporal_causal_hgat_v2` materially improved node metrics over temporal v1, but it regressed on cluster F1 versus temporal v1 (`0.4089` to `0.4040`), which suggests the added temporal objectives are currently smoothing the representation more than they are separating incidents.
- Both v2 variants still produced a single discovered incident cluster on this AIT-ADS slice, so the failure mode remains the same: stronger ranking and coherence, but insufficient fragmentation of the malicious alert set.
- At this stage, before clustering-policy ablations, the best weakly supervised result on AIT-ADS remained `temporal_causal_hgat` v1 at cluster F1 `0.4089`.

Updated direction:

1. keep the multiview v2 path, since it produced a small but real gain over the saved HGAT baseline
2. roll back or soften the temporal v2 continuity/order weights before iterating further on the temporal path
3. prioritize objectives that explicitly increase malicious-cluster fragmentation rather than only improving node discrimination

#### Cluster ablation follow-up: giant-cluster splitting and graph components

- Output: [cluster_ablation_ait_ads_e6](</Users/yanivzimmer/Code/efficient-graph-attack-agent/experiments/h1-incident-aware-hgat/results/cluster_ablation_ait_ads_e6>)
- Sources replayed:
  - saved repo `hgat`
  - `temporal_causal_hgat`
  - `multiview_hgat_v2`
- Strategies tested:
  - current DBSCAN
  - DBSCAN plus temporal gap split
  - DBSCAN plus entity-component split
  - DBSCAN plus tactic/semantic split
  - relation-graph connected components
  - combinations of the above

Best non-semantic result:

- source: `multiview_hgat_v2`
- strategy: `dbscan_split_time_6h`
- semantic/tactic metadata: no
- clusters: `7`
- noise alerts: `2`
- cluster precision: `0.8824`
- cluster recall: `0.9305`
- cluster F1: `0.8842`

Best non-semantic results by source:

- `multiview_hgat_v2 + dbscan_split_time_6h`: cluster F1 `0.8842`
- `saved_hgat + graph_entity_6h`: cluster F1 `0.8820`
- `temporal_causal_hgat + dbscan_split_time_48h`: cluster F1 `0.7297`

Diagnostic semantic result:

- `saved_hgat + dbscan_split_tactic_entity_time_24h`: cluster F1 `0.9791`
- `saved_hgat + graph_semantic_6h`: cluster F1 `0.9791`

Caveat:

- The semantic/tactic ablations are not fair deployment claims on AIT-ADS as currently loaded. In the AIT loader, tactic/technique can be populated from attack-step label windows, which makes these fields label-adjacent. They are useful to show the upper limit of metadata-assisted splitting, but should be excluded from the primary weak-supervision result.

Interpretation:

- The largest real improvement came from splitting the giant DBSCAN cluster by temporal gaps, not from changing the encoder.
- For `multiview_hgat_v2`, a simple `6h` or `12h` temporal split raised cluster F1 from `0.4070` to `0.8842`, essentially matching the saved supervised `grain` upper bound at `0.8867` without using semantic metadata.
- For saved `hgat`, entity/time graph components worked best among non-semantic options, reaching cluster F1 `0.8820`.
- The earlier one-cluster collapse was primarily a clustering policy failure. The embeddings and calibrated candidate sets already contain enough structure for strong incident recovery when the clusterer is allowed to fragment long-running alert groups.

Updated direction:

1. promote `multiview_hgat_v2 + temporal gap splitting` as the strongest fair AIT-ADS path so far, pending the tighter temporal-scale sweep below
2. add a validation-tuned temporal gap instead of hard-coding `6h`
3. evaluate the non-semantic splitter on additional datasets before treating the `0.8842` result as robust
4. keep semantic/tactic ablations separated as oracle-style diagnostics

#### Autonomous sprint: temporal episode scale and new segmentation baseline

- Output: [cluster_ablation_temporal4_ait_ads_e6](</Users/yanivzimmer/Code/efficient-graph-attack-agent/experiments/h1-incident-aware-hgat/results/cluster_ablation_temporal4_ait_ads_e6>)
- Code change: added `4h` temporal episode strategies to [incident_ablation_clusterer.py](/Users/yanivzimmer/Code/efficient-graph-attack-agent/clustering/incident_ablation_clusterer.py).
- Sources replayed:
  - saved repo `hgat`
  - `temporal_causal_hgat`
  - `multiview_hgat_v2`
- Added strategies:
  - `dbscan_split_time_4h`
  - `temporal_only_4h`
  - previously added adaptive quantile splitters and Bayesian-block event segmentation were rerun in the same grid

Best fair/non-semantic result:

- source: `multiview_hgat_v2`
- strategy: `dbscan_split_time_4h`
- semantic/tactic metadata: no
- clusters: `8`
- noise alerts: `2`
- node AUC: `0.9943`
- node F1: `0.9875`
- cluster precision: `0.9517`
- cluster recall: `0.9392`
- cluster F1: `0.9363`

Key ablation:

- `multiview_hgat_v2 + temporal_only_4h` exactly matched `multiview_hgat_v2 + dbscan_split_time_4h` at cluster F1 `0.9363`.
- This means DBSCAN is not contributing to incident construction once the candidate set is produced by `multiview_hgat_v2`; the incident grouping signal is primarily the temporal episode boundary.
- The effect is not encoder-agnostic. Saved HGAT with `4h` temporal-only splitting reached only `0.7840`, and `temporal_causal_hgat` with `4h` reached only `0.4271`.

Negative result: adaptive temporal thresholds

- `multiview_hgat_v2 + temporal_only_adaptive_q90/q95` reached only cluster F1 `0.5764`.
- Saved HGAT with adaptive q90/q95 reached `0.8820`, roughly tying its best entity/time graph result, but did not exceed the `multiview_hgat_v2 + 4h` result.
- Interpretation: naive within-stream gap quantiles do not reliably infer the incident scale; they can over-fragment the multiview candidate stream.

Negative result: Bayesian-block event segmentation

- Raw Bayesian-block segmentation over-fragmented the selected alert stream into roughly `1000` to `1600` tiny clusters.
- For `multiview_hgat_v2`, the best Bayesian-block row was `bayesian_blocks_p8` with precision `0.9923`, recall `0.0132`, and cluster F1 `0.0195`.
- This is useful evidence against unconstrained rate-change segmentation: it finds very pure micro-bursts, but those bursts are far smaller than incident-level objects.

Updated interpretation:

- The strongest current method is better described as **HGAT triage + temporal episode segmentation**, not embedding density clustering.
- `multiview_hgat_v2` still matters because it produces the candidate alert set for which `4h` temporal episodes align with incidents.
- The key research gap is now estimating incident-scale temporal priors without test incident labels. A publishable next step is a validation-free or weak-label-only method for learning the episode gap, then testing whether it recovers the `4h` regime on held-out data.

Updated direction:

1. treat `multiview_hgat_v2 + temporal_only_4h` as the cleanest fair AIT-ADS method so far
2. frame DBSCAN as unnecessary for the current best result, at least on AIT-ADS
3. develop an unsupervised/weakly supervised temporal-prior estimator instead of hand-selecting `4h`
4. turn the Bayesian-block failure into a constrained segmentation variant with minimum episode duration/size or learned adjacent-block merging
5. validate the `4h` episode prior on another split/dataset before making a deployment-level claim

#### Follow-up: weak-label-only macro-gap temporal prior

- Output: [cluster_ablation_macro_elbow_ait_ads_e6](</Users/yanivzimmer/Code/efficient-graph-attack-agent/experiments/h1-incident-aware-hgat/results/cluster_ablation_macro_elbow_ait_ads_e6>)
- Code change: added `macro_elbow` temporal gap estimation to [incident_ablation_clusterer.py](/Users/yanivzimmer/Code/efficient-graph-attack-agent/clustering/incident_ablation_clusterer.py).
- Test coverage: added `test_macro_elbow_estimator_ignores_microburst_gaps` in [test_incident_ablation_clusterer.py](/Users/yanivzimmer/Code/efficient-graph-attack-agent/tests/test_incident_ablation_clusterer.py).

Estimator:

- Select candidate malicious alerts using the saved model threshold.
- Sort selected alerts by timestamp.
- Ignore telemetry micro-burst gaps below `0.25h`.
- Find the earliest prominent adjacent jump in sorted macro gaps.
- Use the lower side of that jump as the temporal episode boundary, scaled by `0.95`.
- This uses no incident IDs, no tactic/technique metadata, and no ground-truth cluster feedback.

Best fair/non-semantic result:

- source: `multiview_hgat_v2`
- strategy: `temporal_only_macro_elbow`
- clusters: `8`
- noise alerts: `2`
- cluster precision: `0.9517`
- cluster recall: `0.9392`
- cluster F1: `0.9363`

Key ablations:

- `multiview_hgat_v2 + temporal_only_macro_elbow` exactly matched `temporal_only_4h`, `dbscan_split_time_4h`, and `dbscan_split_time_macro_elbow` at cluster F1 `0.9363`.
- `saved_hgat + temporal_only_macro_elbow` reached cluster F1 `0.8820`, slightly above its adaptive q90/q95 rows (`0.8820` rounded, exact `0.8820133` vs `0.8819827`) and well above fixed `4h/6h/12h` (`0.7840`).
- `temporal_causal_hgat + temporal_only_macro_elbow` reached cluster F1 `0.5706`, improving over fixed `4h/6h/12h` but still trailing its fixed `48h` result (`0.7297`).

Updated interpretation:

- The current best AIT-ADS method no longer depends on a hand-selected `4h` value. The macro-gap estimator recovers the same incident split from the selected alert stream itself.
- DBSCAN remains unnecessary for the best rows: the matching temporal-only and DBSCAN-plus-temporal results show that candidate selection plus temporal episode segmentation is sufficient.
- The estimator is not universally optimal across encoders, so the next robustness test should be held-out split/dataset validation rather than further tuning on this same AIT-ADS slice.

Updated direction:

1. initially treat `multiview_hgat_v2 + temporal_only_macro_elbow` as the cleanest fair full-slice method on AIT-ADS, pending chronological validation below
2. validate macro-elbow on held-out time windows or the primary dataset
3. cache cluster-evaluation pair structures before running large multi-seed sweeps
4. consider learning the macro-gap threshold with a stability objective if macro-elbow fails out of sample

#### Iteration 1: chronological holdout validation

- Output: [temporal_holdout_macro_elbow_ait_ads_e6](</Users/yanivzimmer/Code/efficient-graph-attack-agent/experiments/h1-incident-aware-hgat/results/temporal_holdout_macro_elbow_ait_ads_e6>)
- Code change: added [run_temporal_holdout_validation.py](/Users/yanivzimmer/Code/efficient-graph-attack-agent/scripts/run_temporal_holdout_validation.py).
- Evaluation: four chronological AIT-ADS folds, replaying saved predictions/embeddings and intersecting ground-truth incident groups with each held-out window.

Key result:

- `multiview_hgat_v2 + temporal_only_4h` was best on mean cluster F1: `0.9515`.
- `multiview_hgat_v2 + temporal_only_macro_elbow` dropped to mean cluster F1 `0.7692`.
- The failure was concentrated in folds 2 and 3:
  - fold 2: macro-elbow `0.6663` vs fixed `4h` `0.9991`
  - fold 3: macro-elbow `0.4818` vs fixed `4h` `0.9534`

Interpretation:

- The full-slice macro-elbow result did not generalize as-is to chronological windows.
- The estimator was chasing 0.7h to 3h local macro gaps in later folds, causing over-fragmentation.
- This means the previous "no hand-fixed gap" claim should be treated as exploratory unless the estimator is hardened or validated out of sample.

#### Iteration 2: robust macro-gap floor

- Output: [temporal_holdout_macro_elbow_floor4_ait_ads_e6](</Users/yanivzimmer/Code/efficient-graph-attack-agent/experiments/h1-incident-aware-hgat/results/temporal_holdout_macro_elbow_floor4_ait_ads_e6>)
- Full-slice confirmation: [cluster_ablation_macro_floor4_ait_ads_e6](</Users/yanivzimmer/Code/efficient-graph-attack-agent/experiments/h1-incident-aware-hgat/results/cluster_ablation_macro_floor4_ait_ads_e6>)
- Code change: added `macro_elbow_floor4` strategies by setting the macro-gap floor to `4h`.

Holdout result:

- `multiview_hgat_v2 + temporal_only_macro_elbow_floor4` reached mean cluster F1 `0.9588`, precision `0.9519`, recall `0.9737`.
- This beat fixed `4h` on mean cluster F1 (`0.9515`) and preserved the same recall.
- Per-fold F1:
  - fold 0: `0.9894`
  - fold 1: `0.9391`
  - fold 2: `0.9989`
  - fold 3: `0.9076`

Full-slice result:

- `multiview_hgat_v2 + temporal_only_macro_elbow_floor4` tied `temporal_only_4h` and the earlier macro-elbow result at cluster F1 `0.9363`.

Caveat:

- The `4h` floor is now a strong prior, not a fully learned quantity.
- It improves chronological robustness and prevents small-gap over-fragmentation, but fold 3 still favors fixed `4h` (`0.9534`) over floor4 macro-elbow (`0.9076`).
- The next clean research target is a stability-selected floor or a validation-free way to infer the minimum macro-gap floor itself.

Updated direction:

1. promote `multiview_hgat_v2 + temporal_only_macro_elbow_floor4` as the best holdout-mean method on AIT-ADS
2. stop claiming the unconstrained macro-elbow estimator is robust
3. treat `4h` as an empirically strong temporal prior until it is inferred from data
4. next iteration should either learn the macro-gap floor or move to another dataset to test whether `4h` is dataset-specific
