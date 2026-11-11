"""Compare the Python and Go exporters: startup, memory, scrape latency and concurrent throughput.

Usage:
    python scripts/exporter_bench.py [scrapes]

Needs an NVIDIA GPU, nvsonar installed, and the Go binary built with `make go-build`.
"""

import statistics
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GO_BINARY = Path(__file__).resolve().parent.parent / "exporter-go" / "bin" / "nvsonar-exporter"


def rss_mb(pid: int) -> float:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def scrape(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.read()


CLIENTS = 16


def run(name: str, cmd: list[str], port: int, scrapes: int) -> dict:
    url = f"http://127.0.0.1:{port}/metrics"
    start = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"{name} exited with code {proc.returncode}")
            try:
                if b"nvsonar_gpu_health_score" in scrape(url):
                    break
            except OSError:
                pass
            time.sleep(0.01)
        startup = time.monotonic() - start

        latencies = []
        for _ in range(scrapes):
            t = time.perf_counter()
            scrape(url)
            latencies.append((time.perf_counter() - t) * 1000)
        latencies.sort()

        t = time.perf_counter()
        with ThreadPoolExecutor(CLIENTS) as pool:
            list(pool.map(scrape, [url] * (4 * scrapes)))
        throughput = 4 * scrapes / (time.perf_counter() - t)
        return {
            "startup_s": startup,
            "rss_mb": rss_mb(proc.pid),
            "p50_ms": statistics.median(latencies),
            "p99_ms": latencies[int(0.99 * (len(latencies) - 1))],
            "scrapes_per_s": throughput,
        }
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def main(scrapes: int = 1000):
    if not GO_BINARY.exists():
        print("Error: Go binary not found, run `make go-build` first")
        sys.exit(1)

    results = {
        "python": run(
            "python", [sys.executable, "-m", "nvsonar", "exporter", "--port", "9301"], 9301, scrapes
        ),
        "go": run("go", [str(GO_BINARY), "--port", "9302"], 9302, scrapes),
    }

    print(f"Scrapes per exporter: {scrapes}")
    print(f"Concurrent clients:   {CLIENTS}")
    print(f"{'':10}{'startup s':>12}{'RSS MB':>10}{'p50 ms':>10}{'p99 ms':>10}{'scrapes/s':>12}")
    for name, r in results.items():
        print(
            f"{name:10}{r['startup_s']:12.2f}{r['rss_mb']:10.1f}{r['p50_ms']:10.2f}{r['p99_ms']:10.2f}"
            f"{r['scrapes_per_s']:12.0f}"
        )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
