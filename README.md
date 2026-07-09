# NVSonar
[![PyPI version](https://img.shields.io/pypi/v/nvsonar.svg)](https://pypi.org/project/nvsonar/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Downloads](https://pepy.tech/badge/nvsonar)](https://pepy.tech/project/nvsonar)
![CI](https://github.com/ikilledkanishk/nvsonar/actions/workflows/ci.yml/badge.svg)

GPU monitoring tools show utilization percentages, but this can be misleading. A GPU reporting 100% utilization may actually be computing useful work, or wastefully stalled waiting on memory transfers, thermal throttling, or power limits. NVSonar analyzes real-time patterns from NVML metrics to identify what's actually limiting your GPU performance.

![nvsonar demo](https://raw.githubusercontent.com/btursunbayev/nvsonar/main/docs/demo.gif)

## Features

- Bottleneck classification (compute-bound, memory-bandwidth-bound, memory-capacity-bound, power-limited, thermal-throttled, data-starved)
- Temporal pattern detection (clock oscillation, temperature trends, utilization dips, memory leaks)
- Multi-GPU outlier detection via Z-scores
- Health scoring with A-F grades (0-100 per GPU)
- Throttle bitmask decoder with severity levels
- PCIe link degradation and ECC error monitoring
- Actionable recommendations with specific commands
- GPU performance benchmarks (memory bandwidth, compute throughput, PCIe speed)
- Historical tracking with trend analysis over time
- Session monitoring Python API for workload profiling
- JSON and CSV output for automation and scripting

## Requirements

- Python 3.10+
- NVIDIA GPU with driver installed
- Linux

## Installation and Usage

```bash
pip install nvsonar
```

```bash
nvsonar                  # interactive TUI
nvsonar report           # one-shot diagnostic
nvsonar report --json    # structured output for scripts/LLMs
nvsonar report --csv     # CSV output for spreadsheets
nvsonar report --gpu 0   # specific GPU
nvsonar benchmark        # GPU performance benchmarks
nvsonar history          # health trends over time
```

## License

Apache License 2.0

## Author

[Bekmukhamed Tursunbayev](https://btursunbayev.com)
