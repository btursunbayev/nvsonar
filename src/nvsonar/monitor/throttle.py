"""Clock throttle reason bitmask decoder"""

from dataclasses import dataclass

import pynvml as nvml

# Maps NVML throttle bitmask values to human-readable info
REASONS = {
    nvml.nvmlClocksThrottleReasonGpuIdle: {
        "name": "GPU Idle",
        "severity": "info",
        "explanation": "GPU has no active workload",
        "action": None,
    },
    nvml.nvmlClocksThrottleReasonApplicationsClocksSetting: {
        "name": "Applications Clocks Setting",
        "severity": "info",
        "explanation": "Clocks limited by application clock setting",
        "action": "Check if nvidia-smi -ac was used to limit clocks",
    },
    nvml.nvmlClocksThrottleReasonSwPowerCap: {
        "name": "Software Power Cap",
        "severity": "warning",
        "explanation": "Clocks reduced to stay within software power limit",
        "action": "Increase power limit with nvidia-smi -pl <watts>",
    },
    nvml.nvmlClocksThrottleReasonHwSlowdown: {
        "name": "Hardware Slowdown",
        "severity": "critical",
        "explanation": "Hardware-triggered slowdown due to thermal or power conditions",
        "action": "Check cooling and power delivery, may indicate PSU or thermal issue",
    },
    nvml.nvmlClocksThrottleReasonSyncBoost: {
        "name": "Sync Boost",
        "severity": "info",
        "explanation": "Clocks synced with other GPUs via sync boost",
        "action": None,
    },
    nvml.nvmlClocksThrottleReasonSwThermalSlowdown: {
        "name": "Software Thermal Slowdown",
        "severity": "warning",
        "explanation": "Driver-initiated thermal throttle, GPU approaching thermal limit",
        "action": "Improve case airflow, clean dust from heatsink/fans",
    },
    nvml.nvmlClocksThrottleReasonHwThermalSlowdown: {
        "name": "Hardware Thermal Slowdown",
        "severity": "critical",
        "explanation": "GPU hardware forced clock reduction due to high temperature",
        "action": "Immediate cooling needed, check fans, thermal paste, ambient temperature",
    },
    nvml.nvmlClocksThrottleReasonHwPowerBrakeSlowdown: {
        "name": "Hardware Power Brake",
        "severity": "critical",
        "explanation": "Power brake engaged, PSU cannot deliver enough power",
        "action": "Check PSU wattage and GPU power cable connections",
    },
}


@dataclass
class ThrottleReason:
    """A single active throttle reason"""

    bitmask: int
    name: str
    severity: str
    explanation: str
    action: str | None


@dataclass
class ThrottleStatus:
    """Throttle status for a GPU"""

    raw_bitmask: int
    active_reasons: list[ThrottleReason]

    @property
    def is_throttled(self) -> bool:
        for reason in self.active_reasons:
            if reason.severity in ("warning", "critical"):
                return True
        return False

    @property
    def worst_severity(self) -> str:
        if not self.active_reasons:
            return "ok"

        severities = {"critical": 3, "warning": 2, "info": 1}
        worst = max(
            self.active_reasons,
            key=lambda r: severities.get(r.severity, 0),
        )
        return worst.severity

    @property
    def summary(self) -> str:
        if not self.active_reasons:
            return "No throttling, clocks at maximum"

        if not self.is_throttled:
            names = [r.name for r in self.active_reasons]
            return f"Clock state: {', '.join(names)}"

        for reason in self.active_reasons:
            if reason.severity == "critical":
                return f"THROTTLED: {reason.name} — {reason.explanation}"

        for reason in self.active_reasons:
            if reason.severity == "warning":
                return f"Throttled: {reason.name} — {reason.explanation}"

        return "Unknown throttle state"


def decode_throttle_reasons(handle) -> ThrottleStatus:
    """Read and decode current clock throttle reasons"""
    try:
        bitmask = nvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
        # print(f"throttle bitmask: {hex(bitmask)}")
    except nvml.NVMLError:
        # print("throttle: failed to read bitmask")
        return ThrottleStatus(raw_bitmask=0, active_reasons=[])

    active = []
    for bit, info in REASONS.items():
        if bitmask & bit:
            reason = ThrottleReason(
                bitmask=bit,
                name=info["name"],
                severity=info["severity"],
                explanation=info["explanation"],
                action=info["action"],
            )
            active.append(reason)
            # print(f"  [{reason.severity}] {reason.name}")

    return ThrottleStatus(raw_bitmask=bitmask, active_reasons=active)
