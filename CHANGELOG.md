# Changelog

## [2.4.0] - 2026-06-05

### Added
- Prometheus exporter (`nvsonar exporter`) exposing GPU metrics, bottleneck classification, and health score
- Ready-made Grafana dashboard at `dashboards/nvsonar.json` covering bottleneck distribution, throttle reasons, and exporter self-monitoring
- Local test stack (docker-compose) for verifying the dashboard against a running exporter

### Changed
- Health score computation extracted to a shared analysis module for reuse across report and exporter


## [2.3.0] - 2026-06-01

### Added
- Plain text output mode for terminals without color support
- Multi-GPU selection in report (single index, comma-separated list, or all)
- Unit tests covering the bottleneck, temporal, outlier, and recommendation analysis

### Changed
- Project marked as Production/Stable on PyPI
- Lint and test now run on every push and pull request

### Fixed
- Clock reduction warning no longer fires when the GPU is intentionally idle


## [2.2.0] - 2026-03-31

### Changed
- Metrics are now nullable: unavailable metrics show "N/A" instead of misleading zeros
- PCIe section hidden on GPUs without PCIe (NVIDIA GB10 Spark, integrated GPUs)
- Clock reduction warning only fires when throttle data confirms an issue
- Health score treats unavailable metrics as neutral instead of penalizing

### Fixed
- JSON output returns `null` instead of crashing on unavailable metrics
- Temporal analysis skips None values instead of crashing
- History handles nullable fields correctly


## [2.1.2] - 2026-03-30

### Fixed
- CUDA kernel files (.cu) now included in pip package
- TUI fails fast with clear message when no GPU/driver found
- `--json` and `--csv` flags are now mutually exclusive
- `nvsonar benchmark` checks for nvcc once upfront with install link
- Missing kernel files show helpful reinstall message

## [2.1.1] - 2026-03-30

### Fixed
- Handle nvmlDeviceGetMemoryInfo_v2 failure on unsupported hardware (NVIDIA GB10 Spark)
- Handle nvmlSystemGetCudaDriverVersion and nvmlDeviceGetPciInfo failures gracefully
- Corrupted or schema-mismatched history files no longer crash `nvsonar history`
- History save failure no longer crashes `nvsonar report`
- Benchmark failures now show error reason instead of just "failed"
- Session stop() without start() no longer crashes
- Collection errors shown in report output

### Added
- TUI tabs for Report, Benchmark, History, Peaks
- Demo GIF in README
- Color-coded metrics (temperature, VRAM, power, throttle) based on severity
- `errors` field in Metrics for tracking collection failures

### Changed
- Unified design across all commands (white borders, consistent headers, no dim text)
- README restructured with GIF on top

## [2.1.0] - 2026-03-29

### Added
- GPU performance benchmarks: `nvsonar benchmark` (memory bandwidth, compute throughput, PCIe speed)
- Historical tracking: `nvsonar history` with trend analysis
- Session monitoring Python API: `nvsonar.start()`, `nvsonar.stop()`, `nvsonar.monitor()`
- GPU process list in report and JSON output
- CSV process fields
- `--version` flag

### Changed
- CLI report loop refactored (no duplicate GPU iteration for CSV)

## [2.0.0] - 2026-03-27

### Added
- Analysis layer: bottleneck classification (compute, memory bandwidth, memory capacity, power, thermal, data-starved)
- Temporal pattern detection (clock oscillation, temperature trends, utilization dips, memory creep)
- Multi-GPU outlier detection via Z-scores
- Actionable recommendations engine with priorities
- Report command: `nvsonar report` with Rich terminal output
- JSON report output: `nvsonar report --json`
- GPU selection: `nvsonar report --gpu N`
- Health scoring (0-100) with letter grades (A-F)
- Throttle bitmask decoder with severity levels
- PCIe link degradation detection
- ECC error monitoring
- GPU performance benchmarks: memory bandwidth, compute throughput, PCIe speed (`nvsonar benchmark`)
- Historical tracking with trend analysis (`nvsonar history`)
- Session monitoring Python API (`nvsonar.start()`, `nvsonar.stop()`, `nvsonar.monitor()`)
- CSV report output (`nvsonar report --csv`)

### Changed
- Complete architecture rewrite: monitor/ -> analysis/ -> report/ layers
- TUI rewritten to use new analysis layer
- CLI uses typer subcommands (nvsonar for TUI, nvsonar report for diagnostics)

### Removed
- Old core/ module (replaced by analysis/)
- Old utils/ module (replaced by monitor/hardware)
- Old single-threshold bottleneck detection

## [1.1.0] - 2026-02-11

### Added
- Bottleneck types
- Utilization tracking
- Peak value history tab (60 sec window)
- Visuals
- Tabbed interface (Overview/History/Settings)

## [1.0.2] - 2026-01-30

### Fixed
- Personal link

## [1.0.1] - 2026-01-30

### Added
- Multi-GPU support

### Changed
- Simplified to use pynvml directly
- Removed unimplemented feature claims from documentation

## [1.0.0] - 2026-01-28

### Added
- GPU detection and information gathering
- Real-time metrics monitoring (temperature, power, utilization, clocks)
- CLI interface with `list`, `monitor`, and `tui` commands
- Interactive terminal UI with live metrics display
- Graceful handling of unsupported GPU features
- Test suite with GPU-specific markers

### Fixed
- CI workflow configuration for PyPI publishing

## [0.0.1] - 2026-01-27

### Added
- Initial project structure
- PyPI publishing setup with GitHub Actions
- Apache 2.0 license
- Code of conduct and security policy
