# NVSonar

[![PyPI version](https://img.shields.io/pypi/v/nvsonar.svg)](https://pypi.org/project/nvsonar/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Active GPU diagnostic tool that identifies performance bottlenecks using targeted micro-probes.

## Why NVSonar?

Traditional GPU monitoring tools show utilization percentages, but this can be misleading. A GPU reporting 100% utilization may actually be computing useful work, or it may be stalled waiting on memory transfers (memory-bound) or PCIe transfers (PCIe-bound).

## Features

- Runs CUDA micro-probes to stress-test specific GPU subsystems
- Monitor temperature, power, utilization, clocks
- Provides an interactive TUI with multi-GPU support
- Detects performance bottlenecks in memory, compute, or PCIe
- Generates an overall GPU health and performance score

## Installation

```bash
pip install nvsonar
```

## Quick Start

```bash
# Launch interactive TUI with all GPUs and live metrics
nvsonar
```

## Interface

```
┌─ NVSonar ──────────────────────────────────────────────────────┐
│                          Available GPUs                        │  
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━┩
│ Index │ Name                       │ Memory │ Driver    │ CUDA │
│   0   │ NVIDIA GeForce GTX 1650 Ti │ 4.0 GB │ 580.95.05 │ 13.0 │
└───────┴────────────────────────────┴────────┴───────────┴──────┘

╭─────────────────────── GPU 0 Metrics ──────────────────────────╮
│  Temperature         45.0°C                                    │
│  Power               2.8W                                      │
│  GPU Utilization     0%                                        │
│  Memory Utilization  0%                                        │
│  Memory Used         0.4 / 4.0 GB                              │
│  GPU Clock           300 MHz                                   │
│  Memory Clock        405 MHz                                   │
╰────────────────────────────────────────────────────────────────╯
```

- All available GPUs displayed in table at top
- Live metrics for each GPU (updates every 0.5s)
- Press 'q' to quit

## Requirements

- Python 3.10+
- NVIDIA GPU with driver installed
- CUDA toolkit (for active probes)
- Linux (tested on Ubuntu)


## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.


## Author

Maintained by [**Bekmukhamed Tursunbayev**](https://btursunbayev.github.io)  
GitHub: https://github.com/btursunbayev · PyPI: https://pypi.org/user/btursunbayev/

