"""Historical data storage and trend analysis"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

HISTORY_DIR = Path.home() / ".nvsonar" / "history"


@dataclass
class HistoryEntry:
    """One saved snapshot of GPU state"""

    timestamp: float
    gpu_index: int
    gpu_name: str

    # metrics
    gpu_utilization: int | None
    memory_utilization: int | None
    memory_used_pct: float | None
    temperature: float | None
    power_usage: float | None
    gpu_clock: int | None
    max_gpu_clock: int | None
    clock_reduction_pct: float | None

    # health
    bottleneck: str
    ecc_correctable: int
    ecc_uncorrectable: int
    pcie_degraded: bool
    throttled: bool


def save(entry: HistoryEntry):
    """Save a history entry to disk"""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    # one file per day per GPU
    day = time.strftime("%Y-%m-%d", time.localtime(entry.timestamp))
    path = HISTORY_DIR / f"gpu{entry.gpu_index}_{day}.jsonl"

    with open(path, "a") as f:
        f.write(json.dumps(asdict(entry)) + "\n")


def save_from_metrics(gpu_index: int, gpu_name: str, metrics, bottleneck):
    """Save a history entry from Metrics and BottleneckResult"""
    entry = HistoryEntry(
        timestamp=time.time(),
        gpu_index=gpu_index,
        gpu_name=gpu_name,
        gpu_utilization=metrics.gpu_utilization,
        memory_utilization=metrics.memory_utilization,
        memory_used_pct=metrics.memory_used_pct,
        temperature=metrics.temperature,
        power_usage=metrics.power_usage,
        gpu_clock=metrics.gpu_clock,
        max_gpu_clock=metrics.max_gpu_clock,
        clock_reduction_pct=metrics.clock_reduction_pct,
        bottleneck=bottleneck.bottleneck.value,
        ecc_correctable=metrics.ecc.correctable,
        ecc_uncorrectable=metrics.ecc.uncorrectable,
        pcie_degraded=metrics.pcie.is_degraded,
        throttled=metrics.throttle.is_throttled,
    )
    save(entry)


def load(gpu_index: int | None = None, days: int = 7) -> list[HistoryEntry]:
    """Load history entries from disk"""
    if not HISTORY_DIR.exists():
        return []

    entries = []
    cutoff = time.time() - (days * 86400)

    for path in sorted(HISTORY_DIR.glob("gpu*.jsonl")):
        # filter by GPU index if specified
        if gpu_index is not None:
            name = path.stem
            if not name.startswith(f"gpu{gpu_index}_"):
                continue

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("timestamp", 0) < cutoff:
                        continue
                    entries.append(HistoryEntry(**data))
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue

    entries.sort(key=lambda e: e.timestamp)
    return entries


@dataclass
class Trend:
    """A detected trend in historical data"""

    metric: str
    direction: str  # rising, falling, stable
    detail: str


def analyze_trends(entries: list[HistoryEntry]) -> list[Trend]:
    """Detect trends in historical data"""
    if len(entries) < 10:
        return []

    trends = []

    # split into first half / second half
    mid = len(entries) // 2
    first = entries[:mid]
    second = entries[mid:]

    # temperature trend
    temps_first = [e.temperature for e in first if e.temperature is not None]
    temps_second = [e.temperature for e in second if e.temperature is not None]

    if temps_first and temps_second:
        avg_temp_first = sum(temps_first) / len(temps_first)
        avg_temp_second = sum(temps_second) / len(temps_second)
        temp_diff = avg_temp_second - avg_temp_first

        if temp_diff > 3:
            trends.append(
                Trend(
                    "temperature",
                    "rising",
                    f"Average temperature increased {temp_diff:.1f}C "
                    f"({avg_temp_first:.0f}C -> {avg_temp_second:.0f}C)",
                )
            )
        elif temp_diff < -3:
            trends.append(
                Trend(
                    "temperature",
                    "falling",
                    f"Average temperature decreased {abs(temp_diff):.1f}C "
                    f"({avg_temp_first:.0f}C -> {avg_temp_second:.0f}C)",
                )
            )

    # ECC error trend
    ecc_first = sum(e.ecc_correctable + e.ecc_uncorrectable for e in first)
    ecc_second = sum(e.ecc_correctable + e.ecc_uncorrectable for e in second)

    if ecc_second > ecc_first and ecc_second > 0:
        trends.append(
            Trend(
                "ecc_errors",
                "rising",
                f"ECC errors increasing ({ecc_first} -> {ecc_second}), "
                f"possible hardware degradation",
            )
        )

    # throttle frequency trend
    throttle_first = sum(1 for e in first if e.throttled) / len(first) * 100
    throttle_second = sum(1 for e in second if e.throttled) / len(second) * 100

    if throttle_second > throttle_first + 10:
        trends.append(
            Trend(
                "throttling",
                "rising",
                f"Throttling more frequent "
                f"({throttle_first:.0f}% -> {throttle_second:.0f}% of samples)",
            )
        )

    # clock degradation
    clocks_first = [
        e.gpu_clock
        for e in first
        if e.gpu_clock is not None and e.gpu_utilization is not None and e.gpu_utilization > 50
    ]
    clocks_second = [
        e.gpu_clock
        for e in second
        if e.gpu_clock is not None and e.gpu_utilization is not None and e.gpu_utilization > 50
    ]

    if clocks_first and clocks_second:
        avg_clock_first = sum(clocks_first) / len(clocks_first)
        avg_clock_second = sum(clocks_second) / len(clocks_second)
        clock_diff = avg_clock_second - avg_clock_first

        if clock_diff < -50:
            trends.append(
                Trend(
                    "clock_speed",
                    "falling",
                    f"Under-load clock speed dropped {abs(clock_diff):.0f} MHz "
                    f"({avg_clock_first:.0f} -> {avg_clock_second:.0f} MHz)",
                )
            )

    # utilization trend (are they using the GPU more or less?)
    util_first = sum(e.gpu_utilization for e in first) / len(first)
    util_second = sum(e.gpu_utilization for e in second) / len(second)

    if util_second > util_first + 15:
        trends.append(
            Trend(
                "utilization",
                "rising",
                f"GPU usage increasing " f"({util_first:.0f}% -> {util_second:.0f}% avg)",
            )
        )

    return trends


def print_history(gpu_index: int | None = None, days: int = 7):
    """Print historical trends to terminal"""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()
    entries = load(gpu_index=gpu_index, days=days)

    if not entries:
        console.print(
            "[dim]No history data found. Run nvsonar report to start collecting data.[/dim]"
        )
        return

    # group by GPU
    gpus: dict[int, list[HistoryEntry]] = {}
    for e in entries:
        gpus.setdefault(e.gpu_index, []).append(e)

    for idx, gpu_entries in sorted(gpus.items()):
        name = gpu_entries[0].gpu_name
        total = len(gpu_entries)

        first_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(gpu_entries[0].timestamp))
        last_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(gpu_entries[-1].timestamp))

        table = Table(show_header=True, box=None, padding=(0, 2), show_lines=False)
        table.add_column("Metric", width=20)
        table.add_column("Value")

        temps = [e.temperature for e in gpu_entries]
        utils = [e.gpu_utilization for e in gpu_entries]

        table.add_row("Samples", f"{total} ({first_time} to {last_time})")
        table.add_row(
            "Temperature",
            f"{min(temps):.0f}C - {max(temps):.0f}C (avg {sum(temps)/len(temps):.0f}C)",
        )
        table.add_row(
            "Utilization", f"{min(utils)}% - {max(utils)}% (avg {sum(utils)/len(utils):.0f}%)"
        )

        ecc_total = sum(e.ecc_correctable + e.ecc_uncorrectable for e in gpu_entries)
        if ecc_total > 0:
            table.add_row("ECC errors", f"[red]{ecc_total}[/red]")

        throttled = sum(1 for e in gpu_entries if e.throttled)
        if throttled > 0:
            table.add_row(
                "Throttled", f"[yellow]{throttled}/{total} ({throttled/total*100:.0f}%)[/yellow]"
            )

        content = Table.grid(padding=(0, 0))
        content.add_row(table)

        # trends
        trends = analyze_trends(gpu_entries)
        if trends:
            content.add_row(Text())
            content.add_row(Text("  Trends:", style="bold"))
            for t in trends:
                color = (
                    "red"
                    if t.direction == "rising"
                    and t.metric in ("temperature", "ecc_errors", "throttling")
                    else "yellow" if t.direction == "rising" else "green"
                )
                content.add_row(Text(f"    [{t.direction}] {t.detail}", style=color))

        header = Text()
        header.append(f"GPU {idx}: {name}", style="bold")
        header.append("    History", style="")
        console.print(Panel(content, title=header, border_style="white"))
