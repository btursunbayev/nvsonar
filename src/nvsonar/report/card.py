"""Terminal report card with health score"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nvsonar.analysis.bottleneck import BottleneckResult, BottleneckType
from nvsonar.analysis.health import health_score as _health_score
from nvsonar.analysis.recommendations import Recommendation
from nvsonar.analysis.temporal import Pattern
from nvsonar.monitor import Metrics
from nvsonar.monitor.hardware import GPUInfo


def _grade(score: int) -> tuple[str, str]:
    """Score to letter grade and color"""
    if score >= 90:
        return "A", "dark_green"
    elif score >= 75:
        return "B", "yellow"
    elif score >= 50:
        return "C", "yellow"
    elif score >= 25:
        return "D", "red"
    else:
        return "F", "red"


def _severity_color(severity: str) -> str:
    if severity == "critical":
        return "bright_red"
    elif severity == "warning":
        return "yellow"
    return "white"


def print_report(
    gpu_info: GPUInfo,
    metrics: Metrics,
    bottleneck: BottleneckResult,
    patterns: list[Pattern] | None = None,
    recommendations: list[Recommendation] | None = None,
    console: Console | None = None,
):
    """Print a full diagnostic report card for one GPU"""
    console = console or Console()

    score = _health_score(metrics, bottleneck)
    grade, grade_color = _grade(score)

    # header
    header = Text()
    header.append(f"GPU {gpu_info.index}: {gpu_info.name}", style="bold")
    header.append("    Health: ", style="")
    header.append(f"{grade} ({score}/100)", style=f"bold {grade_color}")

    # metrics table
    table = Table(show_header=True, box=None, padding=(0, 2), show_lines=False)
    table.add_column("Metric", width=20)
    table.add_column("Value")

    # show N/A for unavailable metrics
    if metrics.gpu_utilization is not None:
        table.add_row("GPU utilization", f"{metrics.gpu_utilization}%")
    else:
        table.add_row("GPU utilization", "N/A")

    if metrics.memory_utilization is not None:
        table.add_row("Memory controller", f"{metrics.memory_utilization}%")
    else:
        table.add_row("Memory controller", "N/A")

    mem_pct = metrics.memory_used_pct
    if metrics.memory_total is not None and metrics.memory_total > 0:
        mem_color = (
            "red" if mem_pct and mem_pct > 90 else "yellow" if mem_pct and mem_pct > 75 else ""
        )
        vram_str = (
            f"{metrics.memory_used // (1024**2)}MB / " f"{metrics.memory_total // (1024**2)}MB"
        )
        if mem_pct is not None:
            vram_str += f" ({mem_pct:.0f}%)"
        if mem_color:
            vram_str = f"[{mem_color}]{vram_str}[/{mem_color}]"
        table.add_row("VRAM", vram_str)
    else:
        table.add_row("VRAM", "N/A")

    if metrics.gpu_clock is not None and metrics.max_gpu_clock is not None:
        clock_str = f"{metrics.gpu_clock} / {metrics.max_gpu_clock} MHz"
        clock_drop = metrics.clock_reduction_pct
        if clock_drop is not None and clock_drop > 15:
            clock_str += f" [yellow]({clock_drop:.0f}% reduced)[/yellow]"
        elif clock_drop is not None and clock_drop > 1:
            clock_str += f" ({clock_drop:.0f}% reduced)"
        table.add_row("Clocks", clock_str)
    else:
        table.add_row("Clocks", "N/A")

    if metrics.temperature is not None:
        temp = metrics.temperature
        temp_color = "red" if temp > 85 else "yellow" if temp > 75 else "green" if temp < 60 else ""
        temp_str = f"[{temp_color}]{temp}C[/{temp_color}]" if temp_color else f"{temp}C"
        table.add_row("Temperature", temp_str)
    else:
        table.add_row("Temperature", "N/A")

    table.add_row("Driver", gpu_info.driver_version)
    table.add_row("CUDA", gpu_info.cuda_version)

    if metrics.power_usage is not None:
        if metrics.power_limit is not None:
            pwr_pct = metrics.power_used_pct
            pwr_color = (
                "red" if pwr_pct and pwr_pct > 95 else "yellow" if pwr_pct and pwr_pct > 85 else ""
            )
            pwr_str = f"{metrics.power_usage:.0f}W / {metrics.power_limit:.0f}W"
            if pwr_pct:
                pwr_str += f" ({pwr_pct:.0f}%)"
            if pwr_color:
                pwr_str = f"[{pwr_color}]{pwr_str}[/{pwr_color}]"
        else:
            pwr_str = f"{metrics.power_usage:.0f}W"
        table.add_row("Power", pwr_str)

    pcie = metrics.pcie
    if pcie.max_link_gen > 0:
        pcie_str = f"Gen{pcie.current_link_gen} x{pcie.current_link_width}"
        if pcie.is_degraded:
            pcie_str += f" [yellow](max Gen{pcie.max_link_gen} x{pcie.max_link_width})[/yellow]"
        table.add_row("PCIe", pcie_str)

    if metrics.throttle.is_throttled:
        table.add_row("Throttle", f"[red]{metrics.throttle.summary}[/red]")
    else:
        table.add_row("Throttle", f"[green]{metrics.throttle.summary}[/green]")

    # bottleneck
    conf_pct = int(bottleneck.confidence * 100)
    bottleneck_text = Text()
    bottleneck_text.append(f"{bottleneck.bottleneck.value}", style="bold")
    bottleneck_text.append(f" ({conf_pct}% confidence)", style="")

    # build the panel content
    content = Table.grid(padding=(0, 0))
    content.add_row(table)
    content.add_row(Text())  # blank line
    content.add_row(Text.assemble(("  Bottleneck: ", ""), bottleneck_text))
    content.add_row(Text(f"  {bottleneck.detail}"))

    # warnings
    if bottleneck.warnings:
        content.add_row(Text())
        content.add_row(Text("  Warnings:", style="bold yellow"))
        for w in bottleneck.warnings:
            content.add_row(Text(f"    {w}", style="yellow"))

    # temporal patterns
    if patterns:
        content.add_row(Text())
        content.add_row(Text("  Patterns:", style="bold"))
        for p in patterns:
            color = _severity_color(p.severity)
            content.add_row(Text(f"    [{p.severity}] {p.detail}", style=color))

    # recommendations
    if recommendations:
        content.add_row(Text())
        content.add_row(Text("  Recommendations:", style="bold"))
        for rec in recommendations:
            priority_color = (
                "red" if rec.priority == 1 else "yellow" if rec.priority == 2 else "white"
            )
            content.add_row(
                Text(f"    [P{rec.priority}] {rec.title}", style=f"bold {priority_color}")
            )
            for action in rec.actions:
                content.add_row(Text(f"      - {action}"))

    # processes
    content.add_row(Text())
    content.add_row(Text("  Processes:", style="bold"))
    if metrics.processes:
        for proc in metrics.processes:
            mem_mb = proc.used_memory // (1024**2)
            content.add_row(Text(f"    PID {proc.pid:<8} {proc.name:<24} {mem_mb} MB"))
    else:
        content.add_row(Text("    (none)"))

    # collection errors
    if metrics.errors:
        content.add_row(Text())
        content.add_row(Text("  Errors:", style="bold red"))
        for err in metrics.errors:
            content.add_row(Text(f"    {err}", style="red"))

    console.print(Panel(content, title=header, border_style="white"))


def print_report_plain(
    gpu_info: GPUInfo,
    metrics: Metrics,
    bottleneck: BottleneckResult,
    patterns: list[Pattern] | None = None,
    recommendations: list[Recommendation] | None = None,
):
    """Print the diagnostic report in plain text, suitable for piping or logs"""
    score = _health_score(metrics, bottleneck)
    grade, _ = _grade(score)

    lines: list[str] = []
    lines.append(f"GPU {gpu_info.index}: {gpu_info.name}  Health: {grade} ({score}/100)")

    def kv(key: str, value: str):
        lines.append(f"  {key}: {value}")

    if metrics.gpu_utilization is not None:
        kv("GPU utilization", f"{metrics.gpu_utilization}%")
    else:
        kv("GPU utilization", "N/A")

    if metrics.memory_utilization is not None:
        kv("Memory controller", f"{metrics.memory_utilization}%")
    else:
        kv("Memory controller", "N/A")

    if metrics.memory_total is not None and metrics.memory_total > 0:
        vram_str = f"{metrics.memory_used // (1024**2)}MB / {metrics.memory_total // (1024**2)}MB"
        mem_pct = metrics.memory_used_pct
        if mem_pct is not None:
            vram_str += f" ({mem_pct:.0f}%)"
        kv("VRAM", vram_str)
    else:
        kv("VRAM", "N/A")

    if metrics.gpu_clock is not None and metrics.max_gpu_clock is not None:
        clock_str = f"{metrics.gpu_clock} / {metrics.max_gpu_clock} MHz"
        clock_drop = metrics.clock_reduction_pct
        if clock_drop is not None and clock_drop > 1:
            clock_str += f" ({clock_drop:.0f}% reduced)"
        kv("Clocks", clock_str)
    else:
        kv("Clocks", "N/A")

    if metrics.temperature is not None:
        kv("Temperature", f"{metrics.temperature}C")
    else:
        kv("Temperature", "N/A")

    kv("Driver", gpu_info.driver_version)
    kv("CUDA", gpu_info.cuda_version)

    if metrics.power_usage is not None:
        if metrics.power_limit is not None:
            pwr_str = f"{metrics.power_usage:.0f}W / {metrics.power_limit:.0f}W"
            pwr_pct = metrics.power_used_pct
            if pwr_pct:
                pwr_str += f" ({pwr_pct:.0f}%)"
        else:
            pwr_str = f"{metrics.power_usage:.0f}W"
        kv("Power", pwr_str)

    pcie = metrics.pcie
    if pcie.max_link_gen > 0:
        pcie_str = f"Gen{pcie.current_link_gen} x{pcie.current_link_width}"
        if pcie.is_degraded:
            pcie_str += f" (max Gen{pcie.max_link_gen} x{pcie.max_link_width})"
        kv("PCIe", pcie_str)

    kv("Throttle", metrics.throttle.summary)

    conf_pct = int(bottleneck.confidence * 100)
    lines.append("")
    lines.append(f"  Bottleneck: {bottleneck.bottleneck.value} ({conf_pct}% confidence)")
    lines.append(f"  {bottleneck.detail}")

    if bottleneck.warnings:
        lines.append("")
        lines.append("  Warnings:")
        for w in bottleneck.warnings:
            lines.append(f"    {w}")

    if patterns:
        lines.append("")
        lines.append("  Patterns:")
        for p in patterns:
            lines.append(f"    [{p.severity}] {p.detail}")

    if recommendations:
        lines.append("")
        lines.append("  Recommendations:")
        for rec in recommendations:
            lines.append(f"    [P{rec.priority}] {rec.title}")
            for action in rec.actions:
                lines.append(f"      - {action}")

    lines.append("")
    lines.append("  Processes:")
    if metrics.processes:
        for proc in metrics.processes:
            mem_mb = proc.used_memory // (1024**2)
            lines.append(f"    PID {proc.pid:<8} {proc.name:<24} {mem_mb} MB")
    else:
        lines.append("    (none)")

    if metrics.errors:
        lines.append("")
        lines.append("  Errors:")
        for err in metrics.errors:
            lines.append(f"    {err}")

    print("\n".join(lines))
