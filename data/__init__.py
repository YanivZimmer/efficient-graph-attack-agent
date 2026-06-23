"""GNN incident discovery data pipeline."""

__all__ = ["AlertGraphArtifacts", "build_graph_from_records"]


def __getattr__(name: str):
    if name in __all__:
        from data.graph_builder import AlertGraphArtifacts, build_graph_from_records

        globals()[name] = locals()[name]
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
