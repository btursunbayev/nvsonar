"""Shared classifier and health-score cases for the Python and Go implementations.

tests/fixtures/classifier_cases.json holds raw GPU readings with the expected bottleneck type
and health score. This test checks the Python implementation against it; the Go exporter
(exporter-go/internal/analysis) checks the same file, so the two cannot drift apart.

Regenerate after an intended rule change:
    python tests/test_classifier_parity.py
"""

import json
import random
from pathlib import Path

import pynvml as nvml
import pytest

from nvsonar.analysis.bottleneck import classify
from nvsonar.analysis.health import health_score
from nvsonar.monitor.hardware import ECCInfo, PCIeInfo
from nvsonar.monitor.metrics import Metrics
from nvsonar.monitor.throttle import REASONS, ThrottleReason, ThrottleStatus

FIXTURE = Path(__file__).parent / "fixtures" / "classifier_cases.json"


def build_metrics(case: dict) -> Metrics:
    bitmask = case["throttle_bitmask"]
    active = [
        ThrottleReason(bit, info["name"], info["severity"], info["explanation"], info["action"])
        for bit, info in REASONS.items()
        if bitmask & bit
    ]
    p = case["pcie"]
    return Metrics(
        gpu_utilization=case["gpu_utilization"],
        memory_utilization=case["memory_utilization"],
        memory_used=case["memory_used"],
        memory_total=case["memory_total"],
        gpu_clock=case["gpu_clock"],
        memory_clock=None,
        max_gpu_clock=case["max_gpu_clock"],
        temperature=case["temperature"],
        power_usage=case["power_usage"],
        power_limit=case["power_limit"],
        fan_speed=None,
        throttle=ThrottleStatus(raw_bitmask=bitmask, active_reasons=active),
        pcie=PCIeInfo(
            p["current_link_gen"],
            p["max_link_gen"],
            p["current_link_width"],
            p["max_link_width"],
            None,
            None,
        ),
        ecc=ECCInfo(case["ecc"]["correctable"], case["ecc"]["uncorrectable"], True),
    )


def _cases() -> list[dict]:
    if not FIXTURE.exists():  # first run of the generator below
        return []
    return json.loads(FIXTURE.read_text())["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["id"])
def test_python_matches_shared_cases(case):
    m = build_metrics(case)
    result = classify(m)
    assert result.bottleneck.value == case["expected"]["bottleneck"]
    assert health_score(m, result) == case["expected"]["health_score"]


def _random_case(rng: random.Random, i: int) -> dict:
    def maybe(value, p_none=0.08):
        return None if rng.random() < p_none else value

    total = rng.choice([6, 16, 24, 80]) * 1024**3
    max_gen, max_width = rng.choice([(0, 0), (3, 16), (4, 16), (5, 16)])
    return {
        "id": f"random_{i:03d}",
        "gpu_utilization": maybe(rng.choice([0, 2, 4, 5, 30, 41, 52, 70, 74, 75, 86, 95, 100])),
        "memory_utilization": maybe(rng.choice([0, 1, 4, 5, 11, 21, 49, 50, 61, 79, 80, 90, 100])),
        "memory_used": maybe(
            int(total * rng.choice([0.05, 0.3, 0.6, 0.85, 0.95, 0.951, 0.97, 1.0]))
        ),
        "memory_total": maybe(total),
        "gpu_clock": maybe(rng.choice([300, 900, 1500, 1800])),
        "max_gpu_clock": maybe(1800),
        "temperature": maybe(float(rng.choice([35, 69, 70, 82, 94, 95, 99]))),
        "power_usage": maybe(rng.choice([10.0, 150.0, 240.0, 270.0, 285.0, 290.0, 300.0])),
        "power_limit": maybe(300.0),
        "throttle_bitmask": rng.choice([0, 0, 0, 0x1, 0x4, 0x8, 0x20, 0x40, 0x80, 0x44, 0x24, 0x5]),
        "pcie": {
            "current_link_gen": max(0, max_gen - rng.choice([0, 0, 1])),
            "max_link_gen": max_gen,
            "current_link_width": max(0, max_width - rng.choice([0, 0, 8])),
            "max_link_width": max_width,
        },
        "ecc": {
            "correctable": rng.choice([0, 0, 0, 3]),
            "uncorrectable": rng.choice([0, 0, 0, 0, 1]),
        },
    }


def _regenerate(n_random: int = 400) -> None:
    # throttle bits in the fixture are NVML's values; the Go table hard-codes the same ones
    assert nvml.nvmlClocksThrottleReasonGpuIdle == 0x1
    assert nvml.nvmlClocksThrottleReasonHwPowerBrakeSlowdown == 0x80
    rng = random.Random(0)
    gib = 1024**3
    base = {
        "gpu_clock": 1500,
        "max_gpu_clock": 1800,
        "temperature": 50.0,
        "power_usage": 100.0,
        "power_limit": 300.0,
        "throttle_bitmask": 0,
        "memory_total": 6 * gib,
        "pcie": {
            "current_link_gen": 3,
            "max_link_gen": 3,
            "current_link_width": 16,
            "max_link_width": 16,
        },
        "ecc": {"correctable": 0, "uncorrectable": 0},
    }
    edge = [
        ("idle", 0, 4, 0.1),
        ("compute", 100, 21, 0.13),
        ("membw_streaming", 100, 100, 0.27),
        ("membw_boundary", 60, 80, 0.2),
        ("capacity_light_compute", 1, 1, 0.97),
        ("starved", 52, 11, 0.65),
        ("starved_low_vram", 41, 9, 0.15),
        ("balanced", 90, 70, 0.4),
        ("light", 80, 55, 0.3),
        ("both_none", None, None, 0.3),
    ]
    cases = [
        {
            **base,
            "id": f"edge_{name}",
            "gpu_utilization": gu,
            "memory_utilization": mu,
            "memory_used": int(frac * 6 * gib),
        }
        for name, gu, mu, frac in edge
    ]
    cases += [_random_case(rng, i) for i in range(n_random)]
    for c in cases:
        m = build_metrics(c)
        r = classify(m)
        c["expected"] = {"bottleneck": r.bottleneck.value, "health_score": health_score(m, r)}
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(
        json.dumps({"generated_by": "tests/test_classifier_parity.py", "cases": cases}, indent=1)
        + "\n"
    )
    print(f"wrote {len(cases)} cases to {FIXTURE}")


if __name__ == "__main__":
    _regenerate()
