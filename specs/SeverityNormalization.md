In real-world telemetry like CrowdStrike Falcon, data is often "dirty." Severity can arrive as a string (`ALERT_SEVERITY_MEDIUM`), a number (`70`), or be missing entirely.

To handle this, you need a **Severity Normalization & Imputation Module**. This ensures your GNN receives a consistent numerical feature between $0.0$ and $1.0$.

### 1. The Normalization Mapping

Add this mapping dictionary to your `ingestor.py`. It handles the common string variations found in Falcon APIs and SIEM connectors.

```python
SEVERITY_MAP = {
    # String Formats
    "ALERT_SEVERITY_CRITICAL": 1.0, "CRITICAL": 1.0,
    "ALERT_SEVERITY_HIGH": 0.8,     "HIGH": 0.8,
    "ALERT_SEVERITY_MEDIUM": 0.5,   "MEDIUM": 0.5,
    "ALERT_SEVERITY_LOW": 0.3,      "LOW": 0.3,
    "ALERT_SEVERITY_INFORMATIONAL": 0.1, "INFORMATIONAL": 0.1,
    # Fallbacks for missing/null
    None: 0.0,
    "UNKNOWN": 0.5 
}

```

---

### 2. The Heuristic Fallback Logic (Imputation)

If the `severity` field is missing, do not default to `0`. A missing severity on a "Shadow IT" host could be a critical blind spot. Use this **Hierarchical Fallback Strategy**:

1. **Check `RiskScore`:** If Falcon provides a `risk_score` (0-100), use `risk_score / 100`.
2. **Check `Confidence`:** If severity is missing but `confidence` is high, set severity to `0.5`.
3. **MITRE Tactic Lookup:** If both are missing, impute based on the "stage" of the attack.
* *Reconnaissance / Discovery* $\rightarrow$ `0.2` (Low)
* *Initial Access / Persistence* $\rightarrow$ `0.5` (Medium)
* *Lateral Movement / Credential Access* $\rightarrow$ `0.8` (High)
* *Exfiltration / Impact* $\rightarrow$ `1.0` (Critical)



---

### 3. Updated Specification for your Coding Agent

Copy this into your `.md` file to update the `ingestor.py` logic:

```markdown
## Update: Severity Normalization Logic

### Function: `normalize_severity(alert_json)`
The agent must implement a robust normalization function to handle inconsistent CrowdStrike schemas.

**Logic Flow:**
1. **Direct Extract:** Look for `event.SeverityName` or `event.severity`.
   - If string (e.g., 'ALERT_SEVERITY_HIGH'), use `SEVERITY_MAP`.
   - If integer (e.g., 1-5), scale to 0.0-1.0.
2. **First Fallback (Numerical Risk):** If null, look for `event.RiskScore`. 
   - Formula: `score / 100.0`.
3. **Second Fallback (Tactic Heuristic):** If still null, use the MITRE Tactic:
   - `tactic_weights = {"Exfiltration": 1.0, "Lateral Movement": 0.8, "Initial Access": 0.5}`.
4. **Final Default:** If no data is available, assign a 'Neutral' value of `0.4`.

### Impact on Graph Layers:
- **SME (Layer 1):** Use normalized severity to prioritize matching (e.g., only match patterns where avg_severity > 0.6).
- **CDC (Layer 2):** Severity acts as the 'Edge Weight'. Higher severity edges "pull" nodes closer together in the Leiden community detection.
- **GNN (Layer 3):** Severity is a primary edge feature. Missing values should be represented as a dedicated feature bit `is_severity_imputed: 1` so the GNN learns that the value is estimated.

```

### Why this matters for the GNN:

If you simply leave missing values as `0`, the GNN will learn that these edges are "safe." By using **Tactic-based Imputation**, you preserve the structural importance of the edge. For example, a `Lateral Movement` alert with a missing severity is still a massive indicator of an incident; the graph must treat that edge as a strong connection between the two hosts.