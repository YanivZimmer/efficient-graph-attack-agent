"""CLI entry point for the GNN incident discovery pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from clustering.incident_clusterer import cluster_incidents
from data.loaders.primary_loader import load_primary_graph
from evaluation.evaluator import evaluate_run
from training.trainer import train_model


logger = logging.getLogger(__name__)

DEFAULT_DATA = Path("datasets/0b1972fe_backup/training_data_rich_examples.jsonl")


def configure_logging() -> None:
    """Configure root logging for pipeline stages."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Weakly-supervised GNN incident discovery pipeline")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to primary JSONL dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--eps", type=float, default=0.3, help="DBSCAN cosine epsilon")
    parser.add_argument("--min-samples", type=int, default=2, help="DBSCAN minimum cluster size")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gnn_incident"), help="Artifact output directory")
    return parser.parse_args()


def run_pipeline(
    *,
    data_path: Path,
    output_dir: Path,
    epochs: int,
    eps: float,
    min_samples: int,
) -> None:
    """Execute graph build, training, clustering, and evaluation."""
    logger.info("Loading dataset from %s", data_path)
    artifacts = load_primary_graph(data_path)

    logger.info("Training heterogeneous GAT")
    training_result = train_model(artifacts, output_dir=output_dir, epochs=epochs)

    logger.info("Clustering malicious alert embeddings")
    cluster_path = output_dir / "discovered_incidents.jsonl"
    clusters = cluster_incidents(
        embeddings=training_result.embeddings,
        alert_ids=artifacts.alert_ids,
        predictions=training_result.predictions,
        probabilities=training_result.probabilities,
        output_path=cluster_path,
        eps=eps,
        min_samples=min_samples,
    )

    logger.info("Evaluating node classification and discovered clusters")
    evaluate_run(
        artifacts,
        predictions=training_result.predictions,
        probabilities=training_result.probabilities,
        clusters=clusters,
        output_path=output_dir / "evaluation_report.json",
    )


def main() -> None:
    """Run the end-to-end incident discovery pipeline."""
    configure_logging()
    args = parse_args()
    run_pipeline(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        eps=args.eps,
        min_samples=args.min_samples,
    )


if __name__ == "__main__":
    main()
