"""Measure CPU time per collect+classify cycle and extrapolate sampling overhead.

Usage:
    python scripts/overhead.py [samples]
"""

import resource
import sys
import time

from nvsonar.analysis import classify
from nvsonar.monitor import MetricsCollector, initialize


def main(samples: int = 200):
    if not initialize():
        print("Error: failed to initialize NVML, no NVIDIA GPU found")
        sys.exit(1)

    collector = MetricsCollector(0)

    for _ in range(5):
        collector.collect()

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cpu_start = time.process_time()
    wall_start = time.monotonic()

    for _ in range(samples):
        metrics = collector.collect()
        classify(metrics)

    cpu = time.process_time() - cpu_start
    wall = time.monotonic() - wall_start
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    ms_per_sample = cpu / samples * 1000
    rss_mb = rss_after / 1024

    print(f"Samples:        {samples}")
    print(f"Wall time:      {wall:.2f}s")
    print(f"CPU time:       {cpu:.3f}s")
    print(f"Per sample:     {ms_per_sample:.2f} ms CPU")
    print(
        f"At 2 Hz:        {ms_per_sample * 2:.1f} ms/s ({ms_per_sample * 2 / 10:.2f}% of one core)"
    )
    print(f"Peak RSS:       {rss_mb:.0f} MB (delta {(rss_after - rss_before) / 1024:.1f} MB)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    main(n)
