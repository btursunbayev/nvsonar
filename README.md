# NVSonar

[![PyPI version](https://img.shields.io/pypi/v/nvsonar.svg)](https://pypi.org/project/nvsonar/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Downloads](https://pepy.tech/badge/nvsonar)](https://pepy.tech/project/nvsonar)

GPU monitoring tools show utilization percentages, but this can be misleading. A GPU reporting 100% utilization may actually be computing useful work, or wastefully stalled waiting on memory transfers, thermal throttling, or power limits. NVSonar analyzes real-time patterns from NVML metrics to identify what's actually limiting your GPU performance.

![nvsonar demo](https://raw.githubusercontent.com/btursunbayev/nvsonar/main/docs/demo.gif)

## Features

- **Diagnostics:** bottleneck classification (compute, memory, power, thermal, data-starved), temporal pattern detection (clock oscillation, temperature trends, utilization dips, memory leaks)
- **Multi-GPU:** outlier detection via Z-scores, flags the GPU slowing down distributed training
- **Health scoring:** 0-100 per GPU with A-F grades, actionable recommendations with specific commands
- **Benchmarks:** memory bandwidth, compute throughput, PCIe speed vs theoretical specs
- **History:** tracks GPU health over time, detects degradation trends
- **Python API:** session monitoring during training (`nvsonar.start()`, `nvsonar.stop()`)
- **Output:** terminal report, JSON, CSV

## Requirements

- Python 3.10+
- NVIDIA GPU with driver installed
- Linux
- CUDA toolkit (only for `nvsonar benchmark`, not required for other commands)

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

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## License

Apache License 2.0

## Author

[Bekmukhamed Tursunbayev](https://btursunbayev.com)
