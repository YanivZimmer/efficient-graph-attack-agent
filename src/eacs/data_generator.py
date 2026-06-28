from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ALERTS = 2_000
DEFAULT_INCIDENTS = 20
DEFAULT_SEED = 20260531
START_TIME = datetime(2026, 5, 1)

TACTICS = {
    "Initial Access": ["T1190", "T1566"],
    "Execution": ["T1059", "T1204"],
    "Persistence": ["T1547", "T1037"],
    "Privilege Escalation": ["T1068", "T1548"],
    "Credential Access": ["T1003", "T1552"],
    "Lateral Movement": ["T1021", "T1570"],
    "Exfiltration": ["T1041", "T1048"],
}

ATTACK_CHAINS = (
    ("Initial Access", "Execution", "Credential Access", "Lateral Movement", "Privilege Escalation", "Exfiltration"),
    ("Initial Access", "Execution", "Persistence", "Credential Access", "Lateral Movement"),
    ("Credential Access", "Execution", "Lateral Movement", "Exfiltration"),
    ("Execution", "Privilege Escalation", "Credential Access", "Lateral Movement", "Exfiltration"),
)

PROCESS_BY_TACTIC = {
    "Initial Access": ["winword.exe", "outlook.exe", "browser.exe"],
    "Execution": ["powershell.exe", "cmd.exe", "wscript.exe"],
    "Persistence": ["schtasks.exe", "reg.exe", "services.exe"],
    "Privilege Escalation": ["fodhelper.exe", "tokenbroker.exe", "installer.exe"],
    "Credential Access": ["lsass.exe", "authsvc.exe", "vaultsvc.exe"],
    "Lateral Movement": ["wmic.exe", "psexec.exe", "remoteadmin.exe"],
    "Exfiltration": ["rclone.exe", "curl.exe", "sync.exe"],
}

EASY_INCIDENT_SEVERITY = (("High", 0.55), ("Critical", 0.45))
EASY_NOISE_SEVERITY = (("Informational", 0.34), ("Low", 0.33), ("Medium", 0.33))

HARD_INCIDENT_SEVERITY = (
    ("Low", 0.10),
    ("Medium", 0.35),
    ("High", 0.35),
    ("Critical", 0.20),
)
HARD_NOISE_SEVERITY = (
    ("Informational", 0.10),
    ("Low", 0.18),
    ("Medium", 0.34),
    ("High", 0.26),
    ("Critical", 0.12),
)

FIELDNAMES = [
    "alert_id",
    "timestamp",
    "source_node",
    "target_node",
    "user",
    "tactic",
    "technique",
    "severity",
    "process",
    "is_incident",
    "incident_id",
]


def generate_falcon_graph_alerts(
    *,
    num_alerts: int = DEFAULT_ALERTS,
    num_incidents: int = DEFAULT_INCIDENTS,
    profile: str = "easy",
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    if num_alerts <= 0:
        raise ValueError("num_alerts must be positive")
    if num_incidents <= 0:
        raise ValueError("num_incidents must be positive")
    if profile not in {"easy", "hard"}:
        raise ValueError("profile must be 'easy' or 'hard'")

    rng = random.Random(seed)
    hosts = [f"AID-{idx:04d}" for idx in range(1000, 1150)]
    users = [f"USER-{idx:03d}" for idx in range(100, 200)]
    rows: list[dict[str, Any]] = []

    for incident_idx in range(1, num_incidents + 1):
        rows.extend(_incident_rows(rng, incident_idx, hosts, users, profile))
        if len(rows) >= num_alerts:
            break

    while len(rows) < num_alerts:
        if profile == "hard" and rng.random() < 0.18:
            rows.extend(_decoy_burst_rows(rng, hosts, users, remaining=num_alerts - len(rows)))
        else:
            rows.append(_noise_row(rng, hosts, users, profile))

    return sorted(rows[:num_alerts], key=lambda row: (row["timestamp"], row["alert_id"]))


def write_alerts_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def summarize_alerts(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    incident_rows = [row for row in materialized if row["is_incident"]]
    noise_rows = [row for row in materialized if not row["is_incident"]]
    incident_high = sum(1 for row in incident_rows if row["severity"] in {"High", "Critical"})
    noise_high = sum(1 for row in noise_rows if row["severity"] in {"High", "Critical"})
    return {
        "alerts": len(materialized),
        "incident_alerts": len(incident_rows),
        "noise_alerts": len(noise_rows),
        "incidents": len({row["incident_id"] for row in incident_rows}),
        "incident_severity": dict(Counter(row["severity"] for row in incident_rows)),
        "noise_severity": dict(Counter(row["severity"] for row in noise_rows)),
        "high_critical_incident_rate": incident_high / len(incident_rows) if incident_rows else 0.0,
        "high_critical_noise_rate": noise_high / len(noise_rows) if noise_rows else 0.0,
    }


def _incident_rows(
    rng: random.Random,
    incident_idx: int,
    hosts: list[str],
    users: list[str],
    profile: str,
) -> list[dict[str, Any]]:
    start_time = START_TIME + timedelta(hours=rng.randint(1, 500), minutes=rng.randint(0, 50))
    current_host = rng.choice(hosts)
    current_user = rng.choice(users)
    chain = list(rng.choice(ATTACK_CHAINS))
    step_count = rng.randint(6, 12)
    while len(chain) < step_count:
        chain.insert(rng.randint(1, len(chain)), rng.choice(["Execution", "Credential Access", "Persistence"]))

    rows = []
    for step_idx, tactic in enumerate(chain[:step_count]):
        target_host = current_host
        if tactic == "Lateral Movement":
            target_host = rng.choice([host for host in hosts if host != current_host])
        elif tactic == "Exfiltration" and rng.random() < 0.30:
            target_host = rng.choice([host for host in hosts if host != current_host])

        rows.append(
            _row(
                rng,
                timestamp=start_time + timedelta(minutes=step_idx * rng.randint(7, 14)),
                source_node=current_host,
                target_node=target_host,
                user=current_user,
                tactic=tactic,
                severity=_severity(rng, profile, incident=True),
                is_incident=True,
                incident_id=f"INC-{incident_idx}",
            )
        )
        current_host = target_host
    return rows


def _noise_row(rng: random.Random, hosts: list[str], users: list[str], profile: str) -> dict[str, Any]:
    source = rng.choice(hosts)
    target = rng.choice(hosts)
    tactic = rng.choice(list(TACTICS))
    return _row(
        rng,
        timestamp=START_TIME + timedelta(hours=rng.randint(1, 600), minutes=rng.randint(0, 59)),
        source_node=source,
        target_node=target,
        user=rng.choice(users),
        tactic=tactic,
        severity=_severity(rng, profile, incident=False),
        is_incident=False,
        incident_id="",
    )


def _decoy_burst_rows(
    rng: random.Random,
    hosts: list[str],
    users: list[str],
    *,
    remaining: int,
) -> list[dict[str, Any]]:
    burst_size = min(remaining, rng.randint(2, 4))
    source = rng.choice(hosts)
    target = rng.choice([host for host in hosts if host != source])
    user = rng.choice(users)
    start_time = START_TIME + timedelta(hours=rng.randint(1, 600), minutes=rng.randint(0, 40))
    tactics = rng.sample(list(TACTICS), k=burst_size)
    rows = []
    for idx, tactic in enumerate(tactics):
        rows.append(
            _row(
                rng,
                timestamp=start_time + timedelta(minutes=idx * rng.randint(5, 20)),
                source_node=source,
                target_node=target if idx % 2 else source,
                user=user,
                tactic=tactic,
                severity=_weighted_choice(rng, HARD_NOISE_SEVERITY),
                is_incident=False,
                incident_id="",
            )
        )
    return rows


def _row(
    rng: random.Random,
    *,
    timestamp: datetime,
    source_node: str,
    target_node: str,
    user: str,
    tactic: str,
    severity: str,
    is_incident: bool,
    incident_id: str,
) -> dict[str, Any]:
    return {
        "alert_id": f"{rng.getrandbits(32):08x}",
        "timestamp": timestamp,
        "source_node": source_node,
        "target_node": target_node,
        "user": user,
        "tactic": tactic,
        "technique": rng.choice(TACTICS[tactic]),
        "severity": severity,
        "process": rng.choice(PROCESS_BY_TACTIC[tactic]),
        "is_incident": is_incident,
        "incident_id": incident_id,
    }


def _severity(rng: random.Random, profile: str, *, incident: bool) -> str:
    if profile == "easy":
        return _weighted_choice(rng, EASY_INCIDENT_SEVERITY if incident else EASY_NOISE_SEVERITY)
    return _weighted_choice(rng, HARD_INCIDENT_SEVERITY if incident else HARD_NOISE_SEVERITY)


def _weighted_choice(rng: random.Random, choices: tuple[tuple[str, float], ...]) -> str:
    labels = [label for label, _ in choices]
    weights = [weight for _, weight in choices]
    return rng.choices(labels, weights=weights, k=1)[0]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic Falcon graph alert CSVs.")
    parser.add_argument("--profile", choices=["easy", "hard"], default="easy")
    parser.add_argument("--output", default="falcon_graph_alerts.csv")
    parser.add_argument("--alerts", type=int, default=DEFAULT_ALERTS)
    parser.add_argument("--incidents", type=int, default=DEFAULT_INCIDENTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rows = generate_falcon_graph_alerts(
        num_alerts=args.alerts,
        num_incidents=args.incidents,
        profile=args.profile,
        seed=args.seed,
    )
    write_alerts_csv(rows, args.output)
    summary = summarize_alerts(rows)
    print(f"Generated {summary['alerts']} {args.profile} Falcon alerts -> {args.output}")
    print(f"Incident alerts: {summary['incident_alerts']} across {summary['incidents']} incidents")
    print(f"Incident severity: {summary['incident_severity']}")
    print(f"Noise severity: {summary['noise_severity']}")
    print(
        "High/Critical rates: "
        f"incident={summary['high_critical_incident_rate']:.3f}, "
        f"noise={summary['high_critical_noise_rate']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
