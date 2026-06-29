# Autoresearch Bootstrap 001

## What I Set Up

- Installed Orchestra Research AI Research Skills into the local agent homes.
- Read the `autoresearch` skill and initialized its workspace in this repo.
- Seeded project state from the repo's existing code, specs, and result artifacts.

## Current Best Read

- The strongest current evidence is not "the model works end-to-end."
- It is "weakly supervised models can classify alerts well, but incident clustering is still the bottleneck."
- AIT-ADS is currently the clearest benchmark for that story because it has incident ground truth.

## Practical Next Step

Prioritize a clustering-focused experiment track:

1. Keep alert embeddings fixed.
2. Sweep clustering choices and thresholds.
3. Measure whether cluster precision can rise without collapsing recall.

If that fails, the next pivot is to make the embedding objective more incident-aware rather than purely alert-label-aware.
