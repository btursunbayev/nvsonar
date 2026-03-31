"""Main TUI application"""

from collections import deque
from dataclasses import dataclass
from time import time

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App as TextualApp
from textual.app import ComposeResult
from textual.widgets import Footer, Header, Static, TabbedContent, TabPane

from nvsonar.monitor import (
    initialize,
    get_device_count,
    get_gpu_info,
    list_gpus,
    MetricsCollector,
)
from nvsonar.analysis import classify, TemporalAnalyzer, recommend

UPDATE_INTERVAL = 0.5
PEAK_WINDOW = 60.0


def _make_bar(value: float, max_value: float, width: int = 20) -> str:
    """Create a text progress bar"""
    if max_value <= 0:
        ratio = 0.0
    else:
        ratio = min(value / max_value, 1.0)

    filled = int(ratio * width)
    empty = width - filled
    return "█" * filled + "░" * empty


@dataclass
class MetricSnapshot:
    """Single metric snapshot with timestamp"""

    timestamp: float
    temperature: float
    power_usage: float
    gpu_utilization: int
    memory_utilization: int
    memory_used: int
    gpu_clock: int
    memory_clock: int
    bottleneck: str | None = None


class DeviceList(Static):
    """Display available GPUs"""

    def on_mount(self) -> None:
        if not initialize():
            self.update("[red]Failed to initialize NVML[/red]")
            return

        devices = list_gpus()
        if not devices:
            self.update("[yellow]No GPUs found[/yellow]")
            return

        table = Table(title="Available GPUs")
        table.add_column("Index", justify="center")
        table.add_column("Name")
        table.add_column("Memory", justify="right")
        table.add_column("Driver")
        table.add_column("CUDA")

        for device in devices:
            memory_gb = device.memory_total / (1024**3)
            table.add_row(
                str(device.index),
                device.name,
                f"{memory_gb:.1f} GB",
                device.driver_version,
                device.cuda_version,
            )

        self.update(table)


class LiveMetrics(Static):
    """Display live metrics for all GPUs"""

    def __init__(self):
        super().__init__()
        self.collectors = []
        self.temporal = {}
        self.history = {}
        self.device_names = {}

    def on_mount(self) -> None:
        if not initialize():
            self.update("[red]Failed to initialize NVML[/red]")
            return

        device_count = get_device_count()
        if device_count == 0:
            self.update("[yellow]No GPUs found[/yellow]")
            return

        for i in range(device_count):
            try:
                collector = MetricsCollector(i)
                self.collectors.append((i, collector))
                self.temporal[i] = TemporalAnalyzer(window_size=60)
                self.history[i] = deque()

                info = get_gpu_info(i)
                self.device_names[i] = info.name if info else f"GPU {i}"
            except RuntimeError:
                pass

        if self.collectors:
            self.set_interval(UPDATE_INTERVAL, self.update_metrics)

    def update_metrics(self) -> None:
        """Update metrics for all GPUs"""
        if not self.collectors:
            return

        try:
            current_time = time()
            panels = []

            for device_index, collector in self.collectors:
                m = collector.collect()
                bottleneck = classify(m)

                # feed temporal analyzer
                self.temporal[device_index].update(m)
                patterns = self.temporal[device_index].detect()

                # get recommendations
                recs = recommend(bottleneck=bottleneck, patterns=patterns)

                # save to history
                self.history[device_index].append(MetricSnapshot(
                    timestamp=current_time,
                    temperature=m.temperature,
                    power_usage=m.power_usage or 0.0,
                    gpu_utilization=m.gpu_utilization,
                    memory_utilization=m.memory_utilization,
                    memory_used=m.memory_used,
                    gpu_clock=m.gpu_clock,
                    memory_clock=m.memory_clock,
                    bottleneck=bottleneck.bottleneck.value,
                ))
                self._clean_old_snapshots(device_index, current_time)

                # build display
                table = Table(show_header=True, box=None, padding=(0, 1), show_lines=False)
                table.add_column("Metric")
                table.add_column("Value")

                # utilization bars
                if m.gpu_utilization is not None:
                    gpu_bar = _make_bar(m.gpu_utilization, 100)
                    table.add_row("GPU Utilization", f"{gpu_bar} {m.gpu_utilization}%")
                else:
                    table.add_row("GPU Utilization", "N/A")

                if m.memory_utilization is not None:
                    mem_bar = _make_bar(m.memory_utilization, 100)
                    table.add_row("Memory Controller", f"{mem_bar} {m.memory_utilization}%")
                else:
                    table.add_row("Memory Controller", "N/A")

                # VRAM
                if m.memory_total is not None and m.memory_total > 0:
                    vram_bar = _make_bar(m.memory_used or 0, m.memory_total)
                    mem_pct = m.memory_used_pct
                    vram_str = (
                        f"{vram_bar} {(m.memory_used or 0) / (1024**3):.1f} / "
                        f"{m.memory_total / (1024**3):.1f} GB"
                    )
                    if mem_pct is not None:
                        vram_str += f" ({mem_pct:.0f}%)"
                    table.add_row("VRAM", vram_str)
                else:
                    table.add_row("VRAM", "N/A")

                # clocks
                if m.gpu_clock is not None and m.max_gpu_clock is not None:
                    clock_str = f"{m.gpu_clock} / {m.max_gpu_clock} MHz"
                    clock_drop = m.clock_reduction_pct
                    if clock_drop is not None and clock_drop > 1:
                        clock_str += f" ({clock_drop:.0f}% reduced)"
                    table.add_row("Clocks", clock_str)
                else:
                    table.add_row("Clocks", "N/A")

                # temperature
                if m.temperature is not None:
                    temp_color = "red" if m.temperature > 85 else "yellow" if m.temperature > 75 else "green" if m.temperature < 60 else ""
                    temp_str = f"[{temp_color}]{m.temperature}C[/{temp_color}]" if temp_color else f"{m.temperature}C"
                    table.add_row("Temperature", temp_str)
                else:
                    table.add_row("Temperature", "N/A")

                # power
                if m.power_usage is not None:
                    if m.power_limit is not None:
                        power_bar = _make_bar(m.power_usage, m.power_limit)
                        power_str = f"{power_bar} {m.power_usage:.1f}W / {m.power_limit:.1f}W"
                    else:
                        power_str = f"{m.power_usage:.1f}W"
                    table.add_row("Power", power_str)

                # fan
                if m.fan_speed is not None:
                    fan_bar = _make_bar(m.fan_speed, 100)
                    table.add_row("Fan Speed", f"{fan_bar} {m.fan_speed}%")

                # throttle
                if m.throttle.is_throttled:
                    table.add_row("Throttle", f"[red]{m.throttle.summary}[/red]")
                else:
                    table.add_row("Throttle", f"[green]{m.throttle.summary}[/green]")

                # bottleneck status
                table.add_row("", "")
                color = _bottleneck_color(bottleneck.bottleneck.value)
                conf_pct = int(bottleneck.confidence * 100)
                table.add_row(
                    "Status",
                    f"[{color}]{bottleneck.bottleneck.value}[/{color}] ({conf_pct}%)",
                )
                table.add_row("", bottleneck.detail)

                # warnings
                for w in bottleneck.warnings:
                    table.add_row("", f"[yellow]{w}[/yellow]")

                # temporal patterns
                for p in patterns:
                    pcolor = "red" if p.severity == "critical" else "yellow"
                    table.add_row("", f"[{pcolor}]{p.detail}[/{pcolor}]")

                # top recommendation
                if recs and recs[0].priority <= 2:
                    table.add_row("", "")
                    table.add_row(
                        "Action",
                        f"[bold]{recs[0].title}[/bold]",
                    )
                    for action in recs[0].actions[:2]:
                        table.add_row("", f"  - {action}")

                device_name = self.device_names.get(device_index, f"GPU {device_index}")
                panel = Panel(table, title=f"{device_name}", border_style="white")
                panels.append(panel)

            group = Group(*panels)
            self.update(group)
        except Exception as e:
            self.update(f"[red]Error: {e}[/red]")

    def _clean_old_snapshots(self, device_index: int, current_time: float) -> None:
        """Remove snapshots older than PEAK_WINDOW seconds"""
        history = self.history.get(device_index)
        if not history:
            return
        while history and (current_time - history[0].timestamp) > PEAK_WINDOW:
            history.popleft()

    def _get_peaks(self, device_index: int, current_time: float) -> dict:
        """Get peak values from history"""
        self._clean_old_snapshots(device_index, current_time)
        history = self.history.get(device_index, [])

        if not history:
            return {}

        return {
            "temperature": max(s.temperature for s in history),
            "power_usage": max(s.power_usage for s in history),
            "gpu_utilization": max(s.gpu_utilization for s in history),
            "memory_utilization": max(s.memory_utilization for s in history),
            "memory_used": max(s.memory_used for s in history),
            "gpu_clock": max(s.gpu_clock for s in history),
            "memory_clock": max(s.memory_clock for s in history),
            "bottleneck": history[-1].bottleneck,
        }


class PeakMetrics(Static):
    """Display peak metrics from history"""

    def __init__(self, metrics_widget):
        super().__init__()
        self.metrics_widget = metrics_widget

    def on_mount(self) -> None:
        self.set_interval(UPDATE_INTERVAL, self.update_peaks)

    def update_peaks(self) -> None:
        """Update peak metrics display"""
        if not self.metrics_widget.collectors:
            self.update("[yellow]No peak data available[/yellow]")
            return

        try:
            current_time = time()
            panels = []

            for device_index, collector in self.metrics_widget.collectors:
                peaks = self.metrics_widget._get_peaks(device_index, current_time)

                if not peaks:
                    continue

                table = Table(show_header=True, box=None, padding=(0, 1), show_lines=False)
                table.add_column("Metric")
                table.add_column("Peak Value")

                gpu_bar = _make_bar(peaks["gpu_utilization"], 100)
                table.add_row("GPU Utilization", f"{gpu_bar} {peaks['gpu_utilization']}%")

                mem_bar = _make_bar(peaks["memory_utilization"], 100)
                table.add_row("Memory Controller", f"{mem_bar} {peaks['memory_utilization']}%")

                table.add_row("Temperature", f"{peaks['temperature']:.0f}C")

                if peaks["power_usage"] > 0:
                    table.add_row("Power", f"{peaks['power_usage']:.1f}W")

                table.add_row("GPU Clock", f"{peaks['gpu_clock']} MHz")
                table.add_row("Memory Clock", f"{peaks['memory_clock']} MHz")

                if peaks.get("bottleneck"):
                    color = _bottleneck_color(peaks["bottleneck"])
                    table.add_row("", "")
                    table.add_row("Status", f"[{color}]{peaks['bottleneck']}[/{color}]")

                device_name = self.metrics_widget.device_names.get(
                    device_index, f"GPU {device_index}"
                )
                panel = Panel(
                    table,
                    title=f"{device_name} Peak Values (last 60s)",
                    border_style="white",
                )
                panels.append(panel)

            if panels:
                group = Group(*panels)
                self.update(group)
            else:
                self.update("No peak data yet")
        except Exception as e:
            self.update(f"[red]Error: {e}[/red]")


class ReportTab(Static):
    """Display diagnostic report card"""

    def on_mount(self) -> None:
        from nvsonar.report.card import _health_score, _grade

        if not initialize():
            self.update("[red]Failed to initialize NVML[/red]")
            return

        device_count = get_device_count()
        if device_count == 0:
            self.update("[yellow]No GPUs found[/yellow]")
            return

        panels = []
        for i in range(device_count):
            info = get_gpu_info(i)
            if not info:
                continue

            collector = MetricsCollector(i)
            metrics = collector.collect()
            bottleneck = classify(metrics)
            recs = recommend(bottleneck=bottleneck)

            score = _health_score(metrics, bottleneck)
            grade, grade_color = _grade(score)

            table = Table(show_header=True, box=None, padding=(0, 2), show_lines=False)
            table.add_column("Metric", width=20)
            table.add_column("Value")

            table.add_row("GPU utilization", f"{metrics.gpu_utilization}%")
            table.add_row("Memory controller", f"{metrics.memory_utilization}%")
            table.add_row("VRAM", f"{metrics.memory_used // (1024**2)}MB / {metrics.memory_total // (1024**2)}MB ({metrics.memory_used_pct:.0f}%)")
            table.add_row("Clocks", f"{metrics.gpu_clock} / {metrics.max_gpu_clock} MHz")

            temp_color = "red" if metrics.temperature > 85 else "yellow" if metrics.temperature > 75 else "green" if metrics.temperature < 60 else ""
            temp_str = f"[{temp_color}]{metrics.temperature}C[/{temp_color}]" if temp_color else f"{metrics.temperature}C"
            table.add_row("Temperature", temp_str)

            if metrics.power_usage is not None:
                if metrics.power_limit is not None:
                    table.add_row("Power", f"{metrics.power_usage:.0f}W / {metrics.power_limit:.0f}W")
                else:
                    table.add_row("Power", f"{metrics.power_usage:.0f}W")

            if metrics.throttle.is_throttled:
                table.add_row("Throttle", f"[red]{metrics.throttle.summary}[/red]")
            else:
                table.add_row("Throttle", f"[green]{metrics.throttle.summary}[/green]")

            content = Table.grid(padding=(0, 0))
            content.add_row(table)

            content.add_row(Text())
            conf_pct = int(bottleneck.confidence * 100)
            content.add_row(Text(f"  Bottleneck: {bottleneck.bottleneck.value} ({conf_pct}%)", style="bold"))
            content.add_row(Text(f"  {bottleneck.detail}"))

            if bottleneck.warnings:
                content.add_row(Text())
                content.add_row(Text("  Warnings:", style="bold yellow"))
                for w in bottleneck.warnings:
                    content.add_row(Text(f"    {w}", style="yellow"))

            if recs:
                content.add_row(Text())
                content.add_row(Text("  Recommendations:", style="bold"))
                for rec in recs:
                    priority_color = "red" if rec.priority == 1 else "yellow" if rec.priority == 2 else "white"
                    content.add_row(Text(f"    [P{rec.priority}] {rec.title}", style=f"bold {priority_color}"))
                    for action in rec.actions:
                        content.add_row(Text(f"      - {action}"))

            header = Text()
            header.append(f"GPU {i}: {info.name}", style="bold")
            header.append(f"    Health: ", style="")
            header.append(f"{grade} ({score}/100)", style=f"bold {grade_color}")
            panels.append(Panel(content, title=header, border_style="white"))

        self.update(Group(*panels))


class BenchmarkTab(Static):
    """Display GPU benchmark results"""

    def on_mount(self) -> None:
        if not initialize():
            self.update("[red]Failed to initialize NVML[/red]")
            return

        info = get_gpu_info(0)
        if not info:
            self.update("[red]Could not read GPU info[/red]")
            return

        self.update("Running benchmarks...")
        self.call_later(self._run_benchmarks, info)

    def _run_benchmarks(self, info) -> None:
        from nvsonar.baselines.specs import find_specs

        specs = find_specs(info.name)

        table = Table(show_header=True, box=None, padding=(0, 2), show_lines=False)
        table.add_column("Benchmark")
        table.add_column("Measured")
        table.add_column("Spec")
        table.add_column("Score")

        try:
            from nvsonar.benchmark import run_memory
            result = run_memory()
            spec_str, score_str = "", ""
            if specs:
                pct = (result.copy_gbps / specs.memory_bandwidth_gbps) * 100
                spec_str = f"{specs.memory_bandwidth_gbps:.0f} GB/s"
                color = "green" if pct >= 70 else "yellow" if pct >= 50 else "red"
                score_str = f"[{color}]{pct:.0f}%[/{color}]"
            table.add_row("Memory Read", f"{result.read_gbps:.1f} GB/s", "", "")
            table.add_row("Memory Write", f"{result.write_gbps:.1f} GB/s", "", "")
            table.add_row("Memory Copy", f"{result.copy_gbps:.1f} GB/s", spec_str, score_str)
        except RuntimeError:
            table.add_row("Memory", "[red]failed[/red]", "", "")

        try:
            from nvsonar.benchmark import run_compute
            result = run_compute()
            spec_str, score_str = "", ""
            if specs:
                pct = (result.tflops / specs.fp32_tflops) * 100
                spec_str = f"{specs.fp32_tflops:.1f} TFLOPS"
                color = "green" if pct >= 80 else "yellow" if pct >= 60 else "red"
                score_str = f"[{color}]{pct:.0f}%[/{color}]"
            table.add_row("FP32 Compute", f"{result.tflops:.2f} TFLOPS", spec_str, score_str)
        except RuntimeError:
            table.add_row("Compute", "[red]failed[/red]", "", "")

        try:
            from nvsonar.benchmark import run_pcie
            result = run_pcie()
            spec_str, score_str = "", ""
            if specs:
                pct = (result.h2d_gbps / specs.pcie_bandwidth_gbps) * 100
                spec_str = f"{specs.pcie_bandwidth_gbps:.1f} GB/s Gen{specs.pcie_gen}"
                color = "green" if pct >= 70 else "yellow" if pct >= 40 else "red"
                score_str = f"[{color}]{pct:.0f}%[/{color}]"
            table.add_row("PCIe Host->GPU", f"{result.h2d_gbps:.1f} GB/s", "", "")
            table.add_row("PCIe GPU->Host", f"{result.d2h_gbps:.1f} GB/s", spec_str, score_str)
        except RuntimeError:
            table.add_row("PCIe", "[red]failed[/red]", "", "")

        header = Text()
        header.append(f"GPU 0: {info.name}", style="bold")
        header.append("    Benchmark", style="")
        self.update(Panel(table, title=header, border_style="white"))


class HistoryTab(Static):
    """Display historical trends"""

    def on_mount(self) -> None:
        from nvsonar.history import load, analyze_trends

        entries = load()

        if not entries:
            self.update("No history data. Run nvsonar report to start collecting.")
            return

        # group by GPU
        gpus: dict[int, list] = {}
        for e in entries:
            gpus.setdefault(e.gpu_index, []).append(e)

        panels = []
        for idx, gpu_entries in sorted(gpus.items()):
            import time as _time
            name = gpu_entries[0].gpu_name
            total = len(gpu_entries)

            first_time = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(gpu_entries[0].timestamp))
            last_time = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(gpu_entries[-1].timestamp))

            table = Table(show_header=True, box=None, padding=(0, 2), show_lines=False)
            table.add_column("Metric", width=20)
            table.add_column("Value")

            temps = [e.temperature for e in gpu_entries]
            utils = [e.gpu_utilization for e in gpu_entries]

            table.add_row("Samples", f"{total} ({first_time} to {last_time})")
            table.add_row("Temperature", f"{min(temps):.0f}C - {max(temps):.0f}C (avg {sum(temps)/len(temps):.0f}C)")
            table.add_row("Utilization", f"{min(utils)}% - {max(utils)}% (avg {sum(utils)/len(utils):.0f}%)")

            ecc_total = sum(e.ecc_correctable + e.ecc_uncorrectable for e in gpu_entries)
            if ecc_total > 0:
                table.add_row("ECC errors", f"[red]{ecc_total}[/red]")

            throttled = sum(1 for e in gpu_entries if e.throttled)
            if throttled > 0:
                table.add_row("Throttled", f"[yellow]{throttled}/{total} ({throttled/total*100:.0f}%)[/yellow]")

            content = Table.grid(padding=(0, 0))
            content.add_row(table)

            trends = analyze_trends(gpu_entries)
            if trends:
                content.add_row(Text())
                content.add_row(Text("  Trends:", style="bold"))
                for t in trends:
                    color = "red" if t.direction == "rising" and t.metric in ("temperature", "ecc_errors", "throttling") else "yellow" if t.direction == "rising" else "green"
                    content.add_row(Text(f"    [{t.direction}] {t.detail}", style=color))

            header = Text()
            header.append(f"GPU {idx}: {name}", style="bold")
            header.append("    History", style="")
            panels.append(Panel(content, title=header, border_style="white"))

        self.update(Group(*panels))


def _bottleneck_color(bottleneck_type: str) -> str:
    """Get color for bottleneck type"""
    colors = {
        "compute_bound": "blue",
        "memory_bandwidth_bound": "cyan",
        "memory_capacity_bound": "red",
        "thermal_throttled": "red",
        "power_limited": "yellow",
        "data_starved": "magenta",
        "balanced": "green",
        "idle": "white",
        "unknown": "white",
    }
    return colors.get(bottleneck_type, "white")


class App(TextualApp):
    """NVSonar terminal interface"""

    TITLE = "NVSonar"
    SUB_TITLE = "GPU Diagnostic Tool"
    CSS = """
    DeviceList {
        height: auto;
        padding: 1;
        margin: 1;
    }

    LiveMetrics {
        height: auto;
        padding: 0;
        margin: 1;
    }

    ReportTab {
        height: auto;
        padding: 0;
        margin: 1;
    }

    BenchmarkTab {
        height: auto;
        padding: 0;
        margin: 1;
    }

    HistoryTab {
        height: auto;
        padding: 0;
        margin: 1;
    }

    PeakMetrics {
        height: auto;
        padding: 0;
        margin: 1;
    }

    TabbedContent {
        height: auto;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()

        metrics_widget = LiveMetrics()

        with TabbedContent():
            with TabPane("Overview", id="overview"):
                yield DeviceList()
                yield metrics_widget
            with TabPane("Report", id="report"):
                yield ReportTab()
            with TabPane("Benchmark", id="benchmark"):
                yield BenchmarkTab()
            with TabPane("History", id="history_tab"):
                yield HistoryTab()
            with TabPane("Peaks", id="peaks"):
                yield PeakMetrics(metrics_widget)
        yield Footer()

    def action_quit(self) -> None:
        """Quit the application"""
        self.exit()


if __name__ == "__main__":
    app = App()
    app.run()
