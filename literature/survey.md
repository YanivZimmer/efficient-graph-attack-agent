# Literature Survey

Initial bootstrap survey for the current repo's research direction.

## Problem Frame

The project is best framed as weakly supervised incident discovery from SOC alerts, not generic flow-based intrusion detection. The main distinction is that the model must group related alerts into incidents while training only on alert-level labels.

## Key References

### GraphWeaver (Freitas and Gharib, 2024)

- Problem fit: direct alert correlation / incident construction.
- Why it matters here: best external conceptual match for a rule-based lower-bound baseline.
- Repo relevance: the local baseline simplifies GraphWeaver down to shared-entity plus time-window correlation.
- URL: https://arxiv.org/abs/2406.01842

### Heterogeneous Graph Transformer (Hu et al., 2020)

- Problem fit: heterogeneous graphs with different node and relation types.
- Why it matters here: architectural backbone for alert-entity graphs containing alerts, hosts, users, processes, and IPs.
- Repo relevance: informs the `hgat` model used for weakly supervised alert scoring.
- URL: https://arxiv.org/abs/2003.01332

### GraphSAGE (Hamilton, Ying, and Leskovec, 2017)

- Problem fit: inductive neighborhood aggregation on graphs.
- Why it matters here: baseline message-passing backbone for several homogeneous graph baselines.
- Repo relevance: the local GRAIN-style implementation uses GraphSAGE layers over a causal alert graph.

### Bayesian Blocks (Scargle et al., 2013)

- Problem fit: event-stream segmentation without predefining equally spaced bins.
- Why it matters here: a genuinely different approach to incident discovery from temporal alert arrivals.
- Repo relevance: the raw version was implemented as `bayesian_blocks_p*` clustering ablations and failed by over-fragmenting AIT-ADS incidents into micro-bursts.
- URL: https://arxiv.org/abs/1304.2818

### Dynamic Graph Transformer with Correlated Spatial-Temporal Positional Encoding (Wang et al., 2024)

- Problem fit: continuous-time dynamic graph representation learning with explicit spatial-temporal positional structure.
- Why it matters here: supports the hypothesis that alert-entity incident discovery should model time as structure, not only as a scalar feature.
- Repo relevance: useful for a future model-side version of the current temporal episode result.
- URL: https://arxiv.org/abs/2407.16959

### SCGC: Self-Supervised Contrastive Graph Clustering (Kulatilleke et al., 2022)

- Problem fit: graph clustering with contrastive objectives and iteratively refined soft cluster labels.
- Why it matters here: supports the broader idea that explicit clustering guidance can matter more than node-classification-only training.
- Repo relevance: a possible future direction if we revisit representation learning after validating the temporal episode prior.
- URL: https://arxiv.org/abs/2204.12656

## Immediate Gaps

- We need stronger literature-backed framing for why post-hoc clustering should work after alert-level weak supervision.
- We need a better explanation for when rule-based incident grouping beats learned embeddings.
- We need clearer literature positioning for the local GRAIN-style implementation versus the exact original paper.
- We need a weak-label-only way to estimate the temporal episode scale; the current best `4h` result is discovered on AIT-ADS and should not be overclaimed until validated elsewhere.
