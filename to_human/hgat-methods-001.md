# HGAT Methods Analysis 001

## Current HGAT Bottlenecks

- The current model in [models/hgat.py](/Users/yanivzimmer/Code/efficient-graph-attack-agent/models/hgat.py) learns alert embeddings and a binary maliciousness head, but no incident-aware objective.
- The current training loop in [training/trainer.py](/Users/yanivzimmer/Code/efficient-graph-attack-agent/training/trainer.py) optimizes only BCE loss on alert labels.
- The current graph builder in [data/graph_builder.py](/Users/yanivzimmer/Code/efficient-graph-attack-agent/data/graph_builder.py) uses a static alert-entity graph with one-hot tactic/technique plus severity and normalized time delta, but no explicit temporal or causal edges between alerts.
- The current incident discovery stage in [clustering/incident_clusterer.py](/Users/yanivzimmer/Code/efficient-graph-attack-agent/clustering/incident_clusterer.py) selects alerts at a fixed `0.5` threshold and applies fixed-parameter DBSCAN.

## Best Improvement Directions

### 1. Incident-aware HGAT with weak positive pairs

Add a contrastive or metric-learning head on top of the alert embeddings. Define weak positive pairs from shared high-confidence entities plus short time windows and weak negative pairs from disjoint entities or long temporal gaps. Keep BCE for alert classification, but add a supervised-contrastive or triplet-style loss so embeddings become incident-friendly, not just triage-friendly.

Why this is promising:
- Current weakly supervised methods already classify well on AIT-ADS.
- The failure mode appears downstream in clustering.
- This is the cleanest way to inject grouping signal without requiring incident labels.

### 2. Replace post-hoc DBSCAN with end-to-end differentiable clustering

Instead of learning embeddings and hoping DBSCAN works, introduce learnable incident slots or cluster nodes and optimize soft assignment jointly with the encoder. This can be done with a bipartite alert-to-cluster graph, Sinkhorn / OT-style assignment, or another differentiable clustering layer.

Why this is promising:
- It attacks the likely bottleneck directly.
- It can let the model trade off classification confidence and cluster compactness during training.

### 3. Temporal-causal HGAT

Augment the heterogeneous graph with alert-to-alert edges that encode temporal precedence, shared-entity continuity, and stage progression. The repo already has a causal-alert-graph idea in the GRAIN-style baseline path; adapt a weakly supervised version of that idea into HGAT rather than leaving temporal structure only inside alert features.

Why this is promising:
- Security incidents are temporal objects.
- Today time is only a scalar feature, not a first-class structural bias.

### 4. Prototype-memory incident learning

Maintain a memory bank of learned incident prototypes. During training, let alerts attend to prototype vectors and penalize diffuse assignments. At inference, cluster by prototype affinity first, then optionally refine with local density clustering.

Why this is promising:
- It provides global incident structure.
- It may prevent the current “one huge malicious cluster” collapse.

### 5. Multi-view HGAT

Build parallel views:
- heterogeneous alert-entity graph
- alert-alert temporal/causal graph
- alert-alert semantic graph from tactic/technique similarity

Fuse them with cross-view attention or co-contrastive learning. The model should learn where each view is trustworthy rather than collapsing everything into one graph.

Why this is promising:
- It matches the fact that incidents are simultaneously entity-linked, temporally ordered, and semantically staged.

## Suggested Priority Order

1. Add weak-pair contrastive loss to current HGAT.
2. Add temporal/causal alert-alert edges.
3. Run clustering ablations with HDBSCAN or constrained agglomerative clustering.
4. If clustering still dominates error, move to differentiable clustering or prototype-memory methods.

## Good Paper-Framing Angle

The strongest paper angle is probably not “better node classification.”

It is:

“Weak supervision already gives strong alert triage; the key unsolved problem is making heterogeneous alert embeddings incident-aware enough for clustering. We propose to bridge that gap with incident-aware weak supervision.”
