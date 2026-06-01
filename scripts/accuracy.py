"""Check that the bottleneck classifier labels known CUDA workloads correctly.

Runs the built-in memory and compute benchmark kernels and samples
the classifier concurrently. Reports the share of samples that match
the expected bottleneck type for each workload.

Usage:
    python scripts/accuracy.py
"""

import sys
import threading
import time

from nvsonar.analysis import classify
from nvsonar.analysis.bottleneck import BottleneckType
from nvsonar.benchmark import run_compute, run_memory
from nvsonar.monitor import MetricsCollector, initialize


def _run_with_sampling(workload, collector, interval_s=0.2) -> list[BottleneckType]:
    stop = threading.Event()
    samples: list[BottleneckType] = []

    def loop():
        while not stop.is_set():
            samples.append(classify(collector.collect()).bottleneck)
            time.sleep(interval_s)

    t = threading.Thread(target=loop)
    t.start()
    try:
        workload()
    finally:
        stop.set()
        t.join()
    return samples


def _report(name: str, expected: BottleneckType, samples: list[BottleneckType]):
    if not samples:
        print(f"{name:8s} no samples collected")
        return
    matches = sum(1 for s in samples if s == expected)
    pct = matches / len(samples) * 100
    print(
        f"{name:8s} expected={expected.value:24s} {matches:>3d}/{len(samples):<3d} ({pct:>3.0f}%)"
    )


def main():
    if not initialize():
        print("Error: failed to initialize NVML, no NVIDIA GPU found")
        sys.exit(1)

    collector = MetricsCollector(0)

    print("Running memory bandwidth workload...")
    samples = _run_with_sampling(run_memory, collector)
    _report("memory", BottleneckType.MEMORY_BANDWIDTH_BOUND, samples)

    time.sleep(1)

    print("Running compute throughput workload...")
    samples = _run_with_sampling(run_compute, collector)
    _report("compute", BottleneckType.COMPUTE_BOUND, samples)


if __name__ == "__main__":
    main()
