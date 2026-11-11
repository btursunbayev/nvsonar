# Go exporter

A standalone build of `nvsonar exporter` as a single binary with no Python runtime. It serves the same metric names, labels and types as the Python exporter, so [../dashboards/nvsonar.json](../dashboards/nvsonar.json) works unchanged.

## Usage

Download `nvsonar-exporter-linux-amd64` or `nvsonar-exporter-linux-arm64` from a GitHub Release, or build it:

```bash
make go-build
./exporter-go/bin/nvsonar-exporter --port 9100
```

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 9100 | HTTP port for `/metrics` |
| `--gpu` | all | GPU index or comma-separated list |
| `--interval` | 2s | background collection interval |
| `--version` | | print version and exit |

The binary links only against libc and loads the NVIDIA driver's NVML library at runtime.

## Layout

```
cmd/nvsonar-exporter/   flags, HTTP server, graceful shutdown
internal/nvmlsource/    NVML reads (go-nvml)
internal/gpu/           metric snapshot and throttle reason decoding
internal/analysis/      bottleneck classification and health score, ported from src/nvsonar/analysis
internal/cache/         background collection loop
internal/collector/     Prometheus collector
```

## Keeping it in sync with Python

`internal/analysis` is a port of the Python rules. Both test suites check the same cases in [../tests/fixtures/classifier_cases.json](../tests/fixtures/classifier_cases.json), so a rule change in one language fails CI until the other matches. After changing the Python rules, regenerate the fixture:

```bash
python tests/test_classifier_parity.py
```
