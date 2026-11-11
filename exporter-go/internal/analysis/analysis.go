// Package analysis ports nvsonar.analysis.bottleneck.classify and
// nvsonar.analysis.health.health_score. Rule order and thresholds match the
// Python implementation exactly; tests/fixtures/classifier_cases.json is
// checked by both the Python and the Go test suites.
package analysis

import "github.com/btursunbayev/nvsonar/exporter-go/internal/gpu"

// Bottleneck types, in the order of the Python BottleneckType enum.
const (
	Idle                 = "idle"
	ComputeBound         = "compute_bound"
	MemoryBandwidthBound = "memory_bandwidth_bound"
	MemoryCapacityBound  = "memory_capacity_bound"
	PowerLimited         = "power_limited"
	ThermalThrottled     = "thermal_throttled"
	DataStarved          = "data_starved"
	Balanced             = "balanced"
	Unknown              = "unknown"
)

// BottleneckTypes lists every type, so the exporter can emit 0/1 per type.
var BottleneckTypes = []string{
	Idle, ComputeBound, MemoryBandwidthBound, MemoryCapacityBound,
	PowerLimited, ThermalThrottled, DataStarved, Balanced, Unknown,
}

// Classify returns the bottleneck type for one snapshot.
func Classify(m gpu.Metrics) string {
	if m.GPUUtilization == nil && m.MemoryUtilization == nil {
		return Unknown
	}
	gu, mu := 0, 0
	if m.GPUUtilization != nil {
		gu = *m.GPUUtilization
	}
	if m.MemoryUtilization != nil {
		mu = *m.MemoryUtilization
	}
	memUsedPct := m.MemoryUsedPct()
	powerPct := m.PowerUsedPct()
	t := m.Throttle

	// VRAM nearly full is the bottleneck whatever the compute load
	if memUsedPct != nil && *memUsedPct > 95 {
		return MemoryCapacityBound
	}
	if gu < 5 && mu < 5 {
		return Idle
	}

	// hardware throttle takes priority, it masks the workload profile
	if t.WorstSeverity() == "critical" && (t.Has("Hardware Thermal Slowdown") || t.Has("Hardware Slowdown")) {
		return ThermalThrottled
	}
	if t.Has("Software Thermal Slowdown") {
		return ThermalThrottled
	}

	if t.Has("Software Power Cap") && powerPct != nil && *powerPct > 90 {
		return PowerLimited
	}
	if powerPct != nil && *powerPct > 95 {
		return PowerLimited
	}

	// NVML GPU utilization is the share of time any kernel runs, so a
	// memory-bound kernel also reads ~100%; the memory controller decides it.
	if mu >= 80 {
		return MemoryBandwidthBound
	}
	if gu > 85 && mu < 50 {
		return ComputeBound
	}
	// kernels run part of the time and the memory controller is quiet
	if gu < 75 && mu < 50 {
		return DataStarved
	}
	if gu > 70 && mu > 60 {
		return Balanced
	}
	if gu >= 5 || mu >= 5 {
		return Balanced
	}
	return Unknown
}

// HealthScore is the 0-100 weighted score from health.py.
func HealthScore(m gpu.Metrics, bottleneck string) int {
	isIdle := bottleneck == Idle

	thermal := 100
	if m.Temperature != nil {
		switch temp := *m.Temperature; {
		case temp < 70:
			thermal = 100
		case temp < 95:
			thermal = int(100 * (95 - temp) / 25)
		default:
			thermal = 0
		}
	}

	power := 100
	if p := m.PowerUsedPct(); p != nil && *p >= 80 {
		if *p < 95 {
			power = int(100 - (*p-80)*(50.0/15))
		} else {
			power = 50
		}
	}

	clocks := 100
	if cd := m.ClockReductionPct(); !isIdle && cd != nil {
		clocks = max(0, int(100-*cd))
	}

	memory := 100
	if mp := m.MemoryUsedPct(); mp != nil && *mp >= 80 {
		memory = max(0, int(100*(100-*mp)/20))
	}

	pcie := 100
	if !isIdle && m.PCIe.MaxLinkGen != 0 && m.PCIe.IsDegraded() {
		pcie = 60
	}

	ecc := 100
	if m.ECC.Uncorrectable > 0 {
		ecc = 0
	} else if m.ECC.Correctable > 0 {
		ecc = 30
	}

	total := float64(thermal)*0.20 + float64(clocks)*0.20 + float64(ecc)*0.20 +
		float64(power)*0.15 + float64(memory)*0.15 + float64(pcie)*0.10

	if m.ECC.Uncorrectable > 0 {
		total *= 0.3
	}
	if m.Throttle.WorstSeverity() == "critical" {
		total *= 0.7
	}
	if m.PCIe.IsDegraded() && !isIdle && m.PCIe.MaxLinkGen > 0 {
		total *= 0.85
	}
	return max(0, min(100, int(total)))
}
