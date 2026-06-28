# Technical Specification: Graph-Native Incident Detection System (GIDS)

## 1. Project Overview

GIDS is an event-driven cybersecurity framework that transforms **CrowdStrike Falcon Alerts** into directional edges in an organizational entity graph. It uses a three-layer detection pipeline to move from atomic alert triaging to contextual incident reconstruction.

### The Detection Pipeline

1. **Layer 1: Subgraph Matching Engine (SME)** - Deterministic (Cypher patterns).
2. **Layer 2: Community Detection & Correlation (CDC)** - Structural (Clustering/Grouping).
3. **Layer 3: GNN-LLM Hybrid Reasoner (GHR)** - Cognitive (Classification & Narrative).

---

## 2. Data Schema (Entity-Relation Model)

### Nodes ($V$)

| Label | Properties | Description |
| --- | --- | --- |
| **Host** | `aid`, `hostname`, `os`, `risk_score` | A device with the Falcon sensor installed. |
| **User** | `sid`, `username`, `is_privileged` | Active Directory or Local user account. |
| **Process** | `name`, `command_line`, `hash` | A running process or binary execution. |
| **IP** | `address`, `reputation`, `is_external` | Internal or External network address. |

### Edges ($E$)

| Type | Properties | Direction |
| --- | --- | --- |
| **ALERT** | `alert_id`, `tactic`, `technique`, `severity`, `ts` | `(Source)-[:ALERT]->(Target)` |

---

## 3. Layer 1: Subgraph Matching Engine (SME)

**Objective:** Detect known attack sequences (e.g., MITRE ATT&CK chains) in real-time.

* **Logic:** Triggered upon every new edge insertion.
* **Anchor:** The `Target` node of the latest alert.
* **Cypher Pattern Example (Ransomware Preparation):**

```cypher
MATCH (u:User)-[e1:ALERT {tactic: 'Credential Access'}]->(h:Host)-[e2:ALERT {tactic: 'Execution'}]->(p:Process)
WHERE e2.ts > e1.ts 
AND duration.between(datetime({epochSeconds: e1.ts}), datetime({epochSeconds: e2.ts})).minutes < 60
AND p.name IN ['vssadmin.exe', 'powershell.exe']
RETURN u, h, p, e1, e2

```

---

## 4. Layer 2: Community Detection & Correlation (CDC)

**Objective:** Group disjoint alerts into high-density "Incident Clusters."

* **Algorithm:** Leiden or Louvain (MAGE optimized).
* **Edge Weighting ($W$):** 
$$W = \frac{Severity}{\log(\Delta t + 2)}$$



*Where $\Delta t$ is the seconds elapsed since the event.*
* **Heuristics:**
* Communities with $>1$ host and $>3$ alerts are promoted to `Incident Subgraphs`.
* Assign a unique `incident_id` to all edges within the cluster.



---

## 5. Layer 3: GNN-LLM Hybrid Reasoner (GHR)

**Objective:** Classify the "shape" of the incident and generate a human-readable report.

### Part A: GNN Encoder (Structural)

* **Architecture:** 2-Layer **GraphSAGE** (PyTorch Geometric).
* **Input:** Subgraph from CDC.
* **Node Features:** One-hot encoding of Node Label + `risk_score`.
* **Output:** Binary classification score $[0.0, 1.0]$.

### Part B: LLM Decoder (Cognitive)

* **Prompt Template:**

```text
SYSTEM: You are an L3 Incident Responder.
DATA: {JSON_SUBGRAPH_METADATA}
GNN_SCORE: {GNN_OUTPUT}
TASK: Analyze the relationship between the {node_count} nodes. 
1. Determine if this is a True Positive or False Positive.
2. Write a narrative of the attacker's journey.
3. List the first three remediation steps.

```

---

## 6. Implementation Roadmap for Agent

### Module 1: `graph_db.py`

* Initialize Memgraph connection using `gqlalchemy`.
* Create unique constraints on `aid`, `sid`, and `alert_id`.

### Module 2: `ingestor.py`

* Transform Falcon JSON alerts into Cypher `MERGE` statements.
* **Logic:** `MERGE (s:Host {aid: ...}) MERGE (t:IP {address: ...}) CREATE (s)-[:ALERT {...}]->(t)`.

### Module 3: `detectors.py`

* `class SME`: Load patterns from `patterns.yaml` and execute via Bolt driver.
* `class CDC`: Trigger `leiden.get()` every 60 seconds and extract results to `NetworkX`.

### Module 4: `inference.py`

* Load `model.onnx` for GNN inference.
* Function `generate_report(subgraph)`: Format metadata and call OpenAI/Anthropic API.

---

## 7. Operational Targets

* **Ingestion Latency:** < 100ms per alert.
* **Inference Latency:** < 2s per incident cluster.
* **Max Graph Size:** 1M+ nodes (maintained via a 30-day sliding window/TTL).

---