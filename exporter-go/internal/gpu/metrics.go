// Package gpu holds one GPU metrics snapshot and the throttle-reason table,
// mirroring nvsonar.monitor (metrics.py, hardware.py, throttle.py).
package gpu

// Metrics is a single snapshot. Pointer fields are nil when NVML could not
// report the value, matching None in the Python implementation.
type Metrics struct {
	GPUUtilization    *int     `json:"gpu_utilization"`
	MemoryUtilization *int     `json:"memory_utilization"`
	MemoryUsed        *uint64  `json:"memory_used"`
	MemoryTotal       *uint64  `json:"memory_total"`
	GPUClock          *int     `json:"gpu_clock"`
	MemoryClock       *int     `json:"memory_clock"`
	MaxGPUClock       *int     `json:"max_gpu_clock"`
	Temperature       *float64 `json:"temperature"`
	PowerUsage        *float64 `json:"power_usage"` // watts
	PowerLimit        *float64 `json:"power_limit"` // watts
	FanSpeed          *int     `json:"fan_speed"`
	Throttle          Throttle `json:"throttle"`
	PCIe              PCIe     `json:"pcie"`
	ECC               ECC      `json:"ecc"`
	ProcessCount      int      `json:"process_count"`
}

// MemoryUsedPct is used/total*100, nil when either is unknown or total is 0.
func (m Metrics) MemoryUsedPct() *float64 {
	if m.MemoryUsed == nil || m.MemoryTotal == nil || *m.MemoryTotal == 0 {
		return nil
	}
	v := float64(*m.MemoryUsed) / float64(*m.MemoryTotal) * 100
	return &v
}

// PowerUsedPct is usage/limit*100, nil when either is unknown or limit is 0.
func (m Metrics) PowerUsedPct() *float64 {
	if m.PowerUsage == nil || m.PowerLimit == nil || *m.PowerLimit == 0 {
		return nil
	}
	v := *m.PowerUsage / *m.PowerLimit * 100
	return &v
}

// ClockReductionPct is how far the SM clock sits below its maximum, in percent, never negative.
func (m Metrics) ClockReductionPct() *float64 {
	if m.GPUClock == nil || m.MaxGPUClock == nil || *m.MaxGPUClock == 0 {
		return nil
	}
	v := (1 - float64(*m.GPUClock)/float64(*m.MaxGPUClock)) * 100
	if v < 0 {
		v = 0
	}
	return &v
}

// PCIe is the link state; generation and width are 0 when unavailable.
type PCIe struct {
	CurrentLinkGen   int `json:"current_link_gen"`
	MaxLinkGen       int `json:"max_link_gen"`
	CurrentLinkWidth int `json:"current_link_width"`
	MaxLinkWidth     int `json:"max_link_width"`
}

// IsDegraded reports a link running below its maximum generation or width.
func (p PCIe) IsDegraded() bool {
	return p.CurrentLinkGen < p.MaxLinkGen || p.CurrentLinkWidth < p.MaxLinkWidth
}

// ECC holds volatile error counts.
type ECC struct {
	Correctable   uint64 `json:"correctable"`
	Uncorrectable uint64 `json:"uncorrectable"`
}
