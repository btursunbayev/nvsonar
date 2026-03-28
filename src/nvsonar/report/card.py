"""Terminal report card with health score"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nvsonar.monitor import Metrics
from nvsonar.monitor.hardware import GPUInfo
from nvsonar.analysis.bottleneck import BottleneckResult, BottleneckType
from nvsonar.analysis.temporal import Pattern
from nvsonar.analysis.recommendations import Recommendation


def _health_score(metrics: Metrics, bottleneck: BottleneckResult) -> int:
    """Compute a 0-100 health score from metrics and analysis"""
    scores = {}

    # thermal: 100 if <70C, linear decay to 0 at 95C
    temp = metrics.temperature
    if temp < 70:
        scores["thermal"] = 100
    elif temp < 95:
        scores["thermal"] = int(100 * (95 - temp) / 25)
    else:
        scores["thermal"] = 0

    # power: 100 if <80% of limit, linear decay to 50 at 95%
    power_pct = metrics.power_used_pct
    if power_pct is None or power_pct < 80:
        scores["power"] = 100
    elif power_pct < 95:
        scores["power"] = int(100 - (power_pct - 80) * (50 / 15))
    else:
        scores["power"] = 50

    is_idle = bottleneck.bottleneck == BottleneckType.IDLE

    # clocks: 100 minus reduction percentage (skip penalty for idle GPUs)
    scores["clocks"] = 100 if is_idle else max(0, int(100 - metrics.clock_reduction_pct))

    # memory: 100 if <80% used, linear decay to 0 at 100%
    mem_pct = metrics.memory_used_pct
    if mem_pct < 80:
        scores["memory"] = 100
    else:
        scores["memory"] = max(0, int(100 * (100 - mem_pct) / 20))

    # pcie: 100 if at max, 60 if degraded (skip penalty for idle GPUs)
    scores["pcie"] = 100 if (is_idle or not metrics.pcie.is_degraded) else 60

    # ecc: 100 if no errors, 30 if correctable, 0 if uncorrectable
    if metrics.ecc.uncorrectable > 0:
        scores["ecc"] = 0
    elif metrics.ecc.correctable > 0:
        scores["ecc"] = 30
    else:
        scores["ecc"] = 100

    # weighted average
    weights = {
        "thermal": 0.20,
        "clocks": 0.20,
        "ecc": 0.20,
        "power": 0.15,
        "memory": 0.15,
        "pcie": 0.10,
    }

    total = sum(scores[k] * weights[k] for k in weights)

    # multiplicative penalties for critical conditions
    if metrics.ecc.uncorrectable > 0:
        total *= 0.3
    if metrics.throttle.worst_severity == "critical":
        total *= 0.7
    if metrics.pcie.is_degraded and not is_idle:
        total *= 0.85

    return max(0, min(100, int(total)))


def _grade(score: int) -> tuple[str, str]:
    """Score to letter grade and color"""
    if score >= 90:
        return "A", "green"
    elif score >= 75:
        return "B", "yellow"
    elif score >= 50:
        return "C", "bright_yellow"
    elif score >= 25:
        return "D", "red"
    else:
        return "F", "bright_red"


def _severity_color(severity: str) -> str:
    if severity == "critical":
        return "bright_red"
    elif severity == "warning":
        return "yellow"
    return "dim"


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
    header.append("    Health: ", style="dim")
    header.append(f"{grade} ({score}/100)", style=f"bold {grade_color}")

    # metrics table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", width=20)
    table.add_column()

    table.add_row("Driver", gpu_info.driver_version)
    table.add_row("CUDA", gpu_info.cuda_version)
    table.add_row("GPU utilization", f"{metrics.gpu_utilization}%")
    table.add_row("Memory controller", f"{metrics.memory_utilization}%")
    table.add_row(
        "VRAM",
        f"{metrics.memory_used // (1024**2)}MB / "
        f"{metrics.memory_total // (1024**2)}MB "
        f"({metrics.memory_used_pct:.0f}%)",
    )
    table.add_row(
        "Clocks",
        f"{metrics.gpu_clock} / {metrics.max_gpu_clock} MHz"
        + (f" ({metrics.clock_reduction_pct:.0f}% reduced)" if metrics.clock_reduction_pct > 1 else ""),
    )
    table.add_row("Temperature", f"{metrics.temperature}C")

    if metrics.power_usage is not None:
        if metrics.power_limit is not None:
            power_str = (
                f"{metrics.power_usage:.0f}W / {metrics.power_limit:.0f}W"
                + (f" ({metrics.power_used_pct:.0f}%)" if metrics.power_used_pct else "")
            )
        else:
            power_str = f"{metrics.power_usage:.0f}W"
        table.add_row("Power", power_str)

    pcie = metrics.pcie
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
    bottleneck_text.append(f" ({conf_pct}% confidence)", style="dim")

    # build the panel content
    content = Table.grid(padding=(0, 0))
    content.add_row(table)
    content.add_row(Text())  # blank line
    content.add_row(Text.assemble(("  Bottleneck: ", "dim"), bottleneck_text))
    content.add_row(Text(f"  {bottleneck.detail}", style="dim"))

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
            priority_color = "red" if rec.priority == 1 else "yellow" if rec.priority == 2 else "dim"
            content.add_row(Text(f"    [P{rec.priority}] {rec.title}", style=f"bold {priority_color}"))
            for action in rec.actions:
                content.add_row(Text(f"      - {action}", style="dim"))

    console.print(Panel(content, title=header, border_style=grade_color))
