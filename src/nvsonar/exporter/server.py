"""HTTP server entrypoint for the Prometheus exporter."""

from prometheus_client import REGISTRY, start_http_server

from nvsonar.exporter.cache import SnapshotCache
from nvsonar.exporter.collectors import NVSonarCollector


def start_server(
    port: int = 9100,
    gpu_indices: list[int] | None = None,
    interval_s: float = 2.0,
) -> SnapshotCache:
    """Start the background collector and the Prometheus HTTP server.

    Returns the SnapshotCache so the caller can stop it on shutdown.
    """
    from nvsonar.monitor import get_device_count

    if gpu_indices is None:
        gpu_indices = list(range(get_device_count()))

    cache = SnapshotCache(gpu_indices, interval_s=interval_s)
    cache.start()

    collector = NVSonarCollector(cache)
    REGISTRY.register(collector)

    start_http_server(port)
    return cache
