# Architecture

NVSonar is organized in three layers:
```
monitor/        collect NVML metrics
    ↓
analysis/       classify bottlenecks, detect patterns, find outliers
    ↓
report/         format output (terminal, JSON, CSV)
```

## Monitor

Reads GPU data through NVIDIA's NVML library (the same data source as nvidia-smi). Each NVML call is individually wrapped in error handling so unsupported queries degrade gracefully instead of crashing.

- `metrics.py` collects a full snapshot: utilization, clocks, temperature, power, fan speed, VRAM
- `throttle.py` decodes the clock throttle reason bitmask into human-readable reasons with severity levels
- `hardware.py` reads static GPU info, PCIe link state, ECC error counts, and running processes

## Analysis

Takes raw metrics and produces diagnostic results. No NVML dependency, operates on plain dataclasses.

**Bottleneck classification** (`bottleneck.py`) cross-references multiple metrics to determine what's actually limiting the GPU. Hardware issues (thermal, power) are checked first because they mask the real workload profile.

| Pattern | Classification |
|---|---|
| High GPU util, low memory controller | Compute-bound |
| Low GPU util, high memory controller | Memory-bandwidth-bound |
| VRAM > 95% | Memory-capacity-bound |
| Power near limit + throttle bit | Power-limited |
| HW thermal throttle bit active | Thermal-throttled |
| Low GPU util, high VRAM allocated | Data-starved |

**Temporal analysis** (`temporal.py`) maintains a sliding window of metrics and detects patterns that single snapshots miss: clock oscillation via coefficient of variation, temperature trends via least-squares slope, periodic utilization dips, and steady VRAM growth.

**Outlier detection** (`outlier.py`) compares metrics across GPUs using Z-scores. Any GPU deviating more than 2 standard deviations from the group is flagged.

**Recommendations** (`recommendations.py`) translates all analysis results into prioritized advice with specific commands to run.

## Report

Formats analysis results for different output targets.

- `card.py` renders a Rich terminal panel with health score (0-100, A-F grade), color-coded metrics, and recommendations
- `json.py` produces structured JSON for scripts and LLM agents
- `csv_report.py` produces CSV with one row per GPU

## Health Scoring

Each GPU gets a 0-100 score weighted across thermal (20%), clock stability (20%), ECC errors (20%), power (15%), memory (15%), and PCIe (10%). Multiplicative penalties apply for critical conditions like uncorrectable ECC errors or critical throttle. Idle GPUs skip clock and PCIe penalties since power-saving behavior is normal.

## Benchmarks

CUDA kernels in `benchmark/kernels/` measure actual GPU performance and compare against theoretical specs from `baselines/specs.py`. Covers memory bandwidth (streaming read/write/copy), compute throughput (FP32 FMA saturation), and PCIe bandwidth (pinned transfers). Kernels are compiled on first run using nvcc and cached in `~/.nvsonar/cache/`.

## Session Monitoring

`session.py` provides a Python API for monitoring GPUs during workloads. Calling `nvsonar.start()` spawns a background thread that collects metrics at 0.5s intervals. `nvsonar.stop()` analyzes the full session and reports time distribution (idle, throttled, data-starved), peaks, averages, and detected patterns.

## History

`history.py` saves GPU state after each `nvsonar report` to `~/.nvsonar/history/` as JSONL (one file per GPU per day). Trend analysis compares the first and second half of stored data to detect changes in temperature, ECC errors, throttle frequency, and clock speeds under load.

## CLI

`cli.py` routes commands through typer. Running `nvsonar` with no arguments launches the Textual TUI with tabs for Overview, Report, Benchmark, History, and Peaks. Subcommands (`report`, `benchmark`, `history`) provide non-interactive access to the same functionality with flags for output format and GPU selection.
