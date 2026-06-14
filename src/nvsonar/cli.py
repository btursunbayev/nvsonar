"""Command-line interface for NVSonar"""

import sys

import typer

from nvsonar import __version__

app = typer.Typer(add_completion=False, invoke_without_command=True)


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print version and exit"),
):
    """GPU diagnostic tool — run without arguments for TUI"""
    if version:
        typer.echo(f"nvsonar {__version__}")
        sys.exit(0)
    if ctx.invoked_subcommand is not None:
        return

    from nvsonar.monitor import initialize

    if not initialize():
        typer.echo("Error: failed to initialize NVML", err=True)
        typer.echo("Make sure you have an NVIDIA GPU with drivers installed", err=True)
        sys.exit(1)

    try:
        from nvsonar.tui.app import App

        tui_app = App()
        tui_app.run()
    except ImportError as e:
        typer.echo(f"Error: Failed to import TUI: {e}", err=True)
        typer.echo("Install dependencies: pip install nvsonar", err=True)
        sys.exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _parse_gpu_selection(value: str, device_count: int) -> list[int]:
    """Resolve a --gpu argument into a list of device indices.

    Accepts an empty string, "all", "-1", a single index, or a comma-separated list.
    """
    value = value.strip().lower()
    if value in ("", "all", "-1"):
        return list(range(device_count))

    indices = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part)
        except ValueError:
            raise typer.BadParameter(f"invalid GPU index: {part!r}")
        if idx < 0 or idx >= device_count:
            raise typer.BadParameter(f"GPU {idx} not found, {device_count} available")
        if idx not in indices:
            indices.append(idx)

    if not indices:
        raise typer.BadParameter("no GPU indices given")
    return indices


@app.command()
def report(
    json: bool = typer.Option(False, "--json", help="Output as JSON"),
    csv: bool = typer.Option(False, "--csv", help="Output as CSV"),
    plain: bool = typer.Option(False, "--plain", help="Plain text output without colors"),
    gpu: str = typer.Option("", "--gpu", help="GPU index or comma-separated list (default: all)"),
):
    """One-shot GPU diagnostic report"""
    exclusive = [f for f, on in [("--json", json), ("--csv", csv), ("--plain", plain)] if on]
    if len(exclusive) > 1:
        typer.echo(f"Error: {', '.join(exclusive)} are mutually exclusive", err=True)
        sys.exit(1)

    from nvsonar.analysis import classify, detect_outliers, recommend
    from nvsonar.monitor import MetricsCollector, get_device_count, get_gpu_info, initialize
    from nvsonar.report import print_report, print_report_plain, report_to_csv_row, to_csv, to_json

    if not initialize():
        typer.echo("Error: failed to initialize NVML, no NVIDIA GPU found", err=True)
        sys.exit(1)

    device_count = get_device_count()
    if device_count == 0:
        typer.echo("Error: no GPUs detected", err=True)
        sys.exit(1)

    indices = _parse_gpu_selection(gpu, device_count)

    all_metrics = {}
    for i in indices:
        collector = MetricsCollector(i)
        all_metrics[i] = collector.collect()

    outliers = []
    if len(all_metrics) > 1:
        outliers = detect_outliers(all_metrics)

    json_reports = []
    csv_rows = []
    for i in indices:
        info = get_gpu_info(i)
        if not info:
            continue

        metrics = all_metrics[i]
        bottleneck = classify(metrics)

        gpu_outliers = [o for o in outliers if o.gpu_index == i]
        recs = recommend(bottleneck=bottleneck, outliers=gpu_outliers)

        try:
            from nvsonar.history import save_from_metrics

            save_from_metrics(i, info.name, metrics, bottleneck)
        except Exception:
            pass

        if json:
            json_reports.append(to_json(info, metrics, bottleneck, recommendations=recs))
        elif csv:
            csv_rows.append(report_to_csv_row(info, metrics, bottleneck))
        elif plain:
            print_report_plain(info, metrics, bottleneck, recommendations=recs)
        else:
            print_report(info, metrics, bottleneck, recommendations=recs)

    if json:
        if len(json_reports) == 1:
            typer.echo(json_reports[0])
        else:
            typer.echo("[" + ",\n".join(json_reports) + "]")
    elif csv:
        typer.echo(to_csv(csv_rows))


@app.command()
def history(
    gpu: int = typer.Option(-1, "--gpu", help="GPU index, -1 for all"),
    days: int = typer.Option(7, "--days", help="Number of days to show"),
):
    """Show GPU health trends over time"""
    from nvsonar.history import print_history

    gpu_index = gpu if gpu >= 0 else None
    print_history(gpu_index=gpu_index, days=days)


@app.command()
def exporter(
    port: int = typer.Option(9100, "--port", help="HTTP port to serve metrics on"),
    gpu: str = typer.Option("", "--gpu", help="GPU index or comma-separated list (default: all)"),
    interval: float = typer.Option(
        2.0, "--interval", help="Background collection interval in seconds"
    ),
):
    """Run a Prometheus metrics exporter"""
    import time

    from nvsonar.exporter import start_server
    from nvsonar.monitor import get_device_count, initialize

    if not initialize():
        typer.echo("Error: failed to initialize NVML, no NVIDIA GPU found", err=True)
        sys.exit(1)

    device_count = get_device_count()
    if device_count == 0:
        typer.echo("Error: no GPUs detected", err=True)
        sys.exit(1)

    indices = _parse_gpu_selection(gpu, device_count)
    plural = "s" if len(indices) != 1 else ""
    typer.echo(
        f"nvsonar exporter on http://0.0.0.0:{port}/metrics "
        f"({len(indices)} GPU{plural}, interval {interval}s)"
    )

    cache = start_server(port=port, gpu_indices=indices, interval_s=interval)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        cache.stop()


@app.command()
def benchmark(
    memory: bool = typer.Option(False, "--memory", help="Run memory bandwidth only"),
    compute: bool = typer.Option(False, "--compute", help="Run compute throughput only"),
    pcie: bool = typer.Option(False, "--pcie", help="Run PCIe bandwidth only"),
):
    """Run GPU performance benchmarks"""
    import shutil

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from nvsonar.baselines.specs import find_specs
    from nvsonar.monitor import get_gpu_info, initialize

    console = Console()

    if not initialize():
        console.print("[red]Error: failed to initialize NVML, no NVIDIA GPU found[/red]")
        sys.exit(1)

    if not shutil.which("nvcc"):
        console.print("[red]Error: CUDA toolkit not found (nvcc not in PATH)[/red]")
        console.print("Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads")
        sys.exit(1)

    info = get_gpu_info(0)
    if not info:
        console.print("[red]Error: could not read GPU info[/red]")
        sys.exit(1)

    specs = find_specs(info.name)

    run_all = not memory and not compute and not pcie

    table = Table(show_header=True, box=None, padding=(0, 2), show_lines=False)
    table.add_column("Benchmark")
    table.add_column("Measured")
    table.add_column("Spec")
    table.add_column("Score")

    if memory or run_all:
        try:
            from nvsonar.benchmark import run_memory

            result = run_memory()

            spec_str = ""
            score_str = ""
            if specs:
                pct = (result.copy_gbps / specs.memory_bandwidth_gbps) * 100
                spec_str = f"{specs.memory_bandwidth_gbps:.0f} GB/s"
                color = "green" if pct >= 70 else "yellow" if pct >= 50 else "red"
                score_str = f"[{color}]{pct:.0f}%[/{color}]"

            table.add_row("Memory Read", f"{result.read_gbps:.1f} GB/s", "", "")
            table.add_row("Memory Write", f"{result.write_gbps:.1f} GB/s", "", "")
            table.add_row("Memory Copy", f"{result.copy_gbps:.1f} GB/s", spec_str, score_str)
        except RuntimeError as e:
            table.add_row("Memory", f"[red]failed: {e}[/red]", "", "")

    if compute or run_all:
        try:
            from nvsonar.benchmark import run_compute

            result = run_compute()

            spec_str = ""
            score_str = ""
            if specs:
                pct = (result.tflops / specs.fp32_tflops) * 100
                spec_str = f"{specs.fp32_tflops:.1f} TFLOPS"
                color = "green" if pct >= 80 else "yellow" if pct >= 60 else "red"
                score_str = f"[{color}]{pct:.0f}%[/{color}]"

            table.add_row("FP32 Compute", f"{result.tflops:.2f} TFLOPS", spec_str, score_str)
        except RuntimeError as e:
            table.add_row("Compute", f"[red]failed: {e}[/red]", "", "")

    if pcie or run_all:
        try:
            from nvsonar.benchmark import run_pcie

            result = run_pcie()

            spec_str = ""
            score_str = ""
            if specs:
                pct = (result.h2d_gbps / specs.pcie_bandwidth_gbps) * 100
                spec_str = f"{specs.pcie_bandwidth_gbps:.1f} GB/s Gen{specs.pcie_gen}"
                color = "green" if pct >= 70 else "yellow" if pct >= 40 else "red"
                score_str = f"[{color}]{pct:.0f}%[/{color}]"

            table.add_row("PCIe Host->GPU", f"{result.h2d_gbps:.1f} GB/s", "", "")
            table.add_row("PCIe GPU->Host", f"{result.d2h_gbps:.1f} GB/s", spec_str, score_str)
        except RuntimeError as e:
            table.add_row("PCIe", f"[red]failed: {e}[/red]", "", "")

    header = Text()
    header.append(f"GPU 0: {info.name}", style="bold")
    header.append("    Benchmark", style="")
    console.print(Panel(table, title=header, border_style="white"))


if __name__ == "__main__":
    app()
