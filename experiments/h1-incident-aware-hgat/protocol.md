# Protocol — H1 Incident-Aware HGAT

## Goal

Test whether the current HGAT underperforms on incident discovery mainly because its embeddings are optimized only for alert-level maliciousness, not for clusterability.

This experiment keeps the current HGAT backbone mostly intact and adds the smallest plausible set of changes needed to make alert embeddings more incident-aware:

1. weak incident pair supervision
2. explicit alert-alert temporal/entity edges
3. a cleaner clustering/evaluation sweep

## Method Name

`hgat_incident_v1`

## Research Hypothesis

Weakly supervised incident discovery will improve if alert embeddings are trained to:

- separate malicious from benign alerts, and
- pull together alerts that are likely to belong to the same incident under weak evidence

The expected signature is:

- node AUC stays strong
- cluster precision rises materially
- cluster F1 rises above the current HGAT baseline on AIT-ADS

## Current Baseline

From existing artifacts:

- `hgat` on AIT-ADS cluster F1: `0.4046`
- `graphweaver` on AIT-ADS cluster F1: `0.6533`
- `grain` on AIT-ADS cluster F1: `0.8867`

The baseline to beat for this experiment is the current weakly supervised HGAT result, with a secondary goal of narrowing the gap to GraphWeaver.

## Version 1 Design

### 1. Encoder

Keep the current 2-layer HGT encoder in [models/hgat.py](/Users/yanivzimmer/Code/efficient-graph-attack-agent/models/hgat.py).

Minimal additions:

- add a projection head for metric learning:
  - `Linear(out_channels, proj_dim)`
  - L2 normalize projected embeddings before pairwise similarity

This lets us use one embedding space for clustering and one projection space for pair losses if needed.

### 2. Graph Changes

Extend the graph in [data/graph_builder.py](/Users/yanivzimmer/Code/efficient-graph-attack-agent/data/graph_builder.py) with alert-alert edges.

#### New edge types

- `("alert", "same_host", "alert")`
- `("alert", "same_user", "alert")`
- `("alert", "same_process", "alert")`
- `("alert", "same_ip", "alert")`
- `("alert", "precedes", "alert")`

#### Edge construction rules

For alerts `i` and `j`:

- `same_{entity}` edge if both alerts contain the same entity value for that type.
- `precedes` edge if:
  - both alerts have timestamps
  - `t_j > t_i`
  - `t_j - t_i <= max_precedes_hours`
  - and they share at least one entity of any tracked type

Default:

- `max_precedes_hours = 6`

Why this is intentionally small:

- it injects temporal continuity and entity continuity without redesigning the whole data model
- HGT already handles typed relations, so this slots into the existing architecture naturally

### 3. Weak Incident Pair Mining

Use only training alerts to create weak pair supervision.

#### Positive pairs `P`

A pair `(i, j)` is a weak positive if all are true:

- `y_i = 1` and `y_j = 1`
- they share at least one entity
- `abs(t_i - t_j) <= tau_pos_hours` when timestamps exist
- optional strengthening rule: same MITRE tactic OR shared entity type in `{host, user}`

Default:

- `tau_pos_hours = 6`

#### Hard negative pairs `N`

A pair `(i, j)` is a weak negative if either:

- one alert is malicious and the other is benign, or
- both are malicious but:
  - share no entity
  - `abs(t_i - t_j) >= tau_neg_hours`
  - tactic mismatch

Default:

- `tau_neg_hours = 24`

#### Pair sampling

Do not use all pairs.

Per epoch:

- sample up to `k_pos = 8` positives per anchor
- sample up to `k_neg = 16` negatives per anchor

This keeps the loss stable and cheap enough for the current full-graph training loop.

### 4. Losses

Let:

- `logits_i` be the alert classification logit
- `z_i` be the L2-normalized projected embedding
- `s(i, j) = z_i dot z_j`

#### Classification loss

`L_cls = BCEWithLogitsLoss(logits[train_mask], y[train_mask])`

#### Positive pair loss

For weak positives:

`L_pos = mean_(i,j in P) (1 - s(i, j))`

This directly pulls likely incident-mates together.

#### Negative pair loss

For weak negatives:

`L_neg = mean_(i,j in N) relu(s(i, j) - m_neg)`

Default:

- `m_neg = 0.2`

This only penalizes negatives that are too similar.

#### Total loss

`L_total = L_cls + lambda_pos * L_pos + lambda_neg * L_neg`

Default:

- `lambda_pos = 0.5`
- `lambda_neg = 0.25`

This is the exact version-1 objective. It is intentionally simpler than InfoNCE or Sinkhorn clustering so we can isolate whether weak incident-aware geometry helps at all.

### 5. Inference and Clustering

Keep the current clustering path for the first pass, but run a controlled sweep.

#### Default inference path

- encode all alerts
- classify malicious alerts
- cluster predicted-malicious alerts

#### Clustering sweep

Evaluate:

- DBSCAN with `eps in {0.15, 0.2, 0.25, 0.3, 0.35}`
- `min_samples in {2, 3, 5}`
- threshold in `{0.4, 0.5, 0.6}`

If available later, add:

- HDBSCAN
- agglomerative clustering with cosine distance and time-window constraints

The point of version 1 is to test whether better embeddings improve clustering even before changing the clusterer.

## Evaluation Plan

### Primary metric

On AIT-ADS:

- `cluster_f1`

### Secondary metrics

- `cluster_precision`
- `cluster_recall`
- node `auc`
- node `f1`
- cluster count
- mean tactic coherence

### Success criteria

Minimal success:

- beat current HGAT cluster F1 `0.4046`

Strong success:

- cluster F1 `>= 0.50`
- no major regression in node AUC

Very strong success:

- cluster precision improves materially while recall remains high
- gap to GraphWeaver narrows substantially

## Ablation Matrix

### A0. Current HGAT baseline

- current graph
- BCE only
- DBSCAN default

### A1. Pair loss only

- current graph
- BCE + pair losses
- DBSCAN default

Purpose:

- isolate whether the embedding objective alone helps

### A2. Alert-alert edges only

- graph with typed alert-alert edges
- BCE only
- DBSCAN default

Purpose:

- isolate whether temporal/entity structure alone helps

### A3. Full `hgat_incident_v1`

- graph with alert-alert edges
- BCE + pair losses
- DBSCAN default

Purpose:

- test the core method

### A4. Full `hgat_incident_v1` + clustering sweep

- same encoder as A3
- threshold and DBSCAN sweep

Purpose:

- measure how much remaining error is post-processing rather than representation learning

## Minimal Code Edit Plan

### [data/graph_builder.py](/Users/yanivzimmer/Code/efficient-graph-attack-agent/data/graph_builder.py)

Add:

- helper to build typed alert-alert edges
- optional config args:
  - `include_alert_alert_edges: bool = False`
  - `max_precedes_hours: float = 6.0`

Implementation note:

- keep current alert-entity graph intact
- append new edge types into the same `HeteroData`

### [models/hgat.py](/Users/yanivzimmer/Code/efficient-graph-attack-agent/models/hgat.py)

Add:

- projection head:
  - `self.projection = Linear(out_channels, proj_dim)`
- method:
  - `project(data) -> normalized projected alert embeddings`

Leave the classifier head unchanged.

### [training/trainer.py](/Users/yanivzimmer/Code/efficient-graph-attack-agent/training/trainer.py)

Add:

- pair mining helper over `artifacts.alert_records`
- pair loss helpers
- trainer config args:
  - `use_incident_pair_loss`
  - `lambda_pos`
  - `lambda_neg`
  - `tau_pos_hours`
  - `tau_neg_hours`
  - `m_neg`
- record pair-loss terms in history

### [main.py](/Users/yanivzimmer/Code/efficient-graph-attack-agent/main.py)

Add CLI flags for:

- alert-alert edges on/off
- pair loss on/off
- pair-loss hyperparameters

### [clustering/incident_clusterer.py](/Users/yanivzimmer/Code/efficient-graph-attack-agent/clustering/incident_clusterer.py)

No architecture change required for version 1.

Optional:

- add small parameter sweep helper in a benchmark script rather than inside the clusterer itself

## Exact Implementation Order

1. Add alert-alert edges to graph construction.
2. Confirm current HGAT still trains with the richer graph and reproduces baseline-ish node metrics.
3. Add projection head and pair mining.
4. Add pair losses behind a flag.
5. Run A0/A1/A2/A3 on AIT-ADS.
6. Run A4 clustering sweep only if A3 beats A0.

## Failure Modes to Watch

- Pair mining is too noisy and collapses unrelated malicious alerts together.
- Added alert-alert edges oversmooth embeddings and reduce node classification.
- Pair losses produce one dense malicious manifold, making DBSCAN even worse.
- Improvement on AIT-ADS does not transfer to primary.

## What Would Falsify the Hypothesis

If A1 and A3 both fail to beat A0 on AIT-ADS cluster F1, and especially if cluster precision does not improve, then the problem is probably not just the loss geometry. At that point the next move should be:

- differentiable clustering, or
- prototype-memory incident heads, or
- a more strongly temporal model

## Deliverables

- code path for `hgat_incident_v1`
- ablation table for A0-A4
- short result memo in `experiments/h1-incident-aware-hgat/analysis.md`
- if promising, promote this into the main benchmark suite
