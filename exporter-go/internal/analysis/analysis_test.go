package analysis

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/btursunbayev/nvsonar/exporter-go/internal/gpu"
)

// fixturePath is the shared case file also checked by tests/test_classifier_parity.py.
const fixturePath = "../../../tests/fixtures/classifier_cases.json"

type sharedCase struct {
	ID                string   `json:"id"`
	GPUUtilization    *int     `json:"gpu_utilization"`
	MemoryUtilization *int     `json:"memory_utilization"`
	MemoryUsed        *uint64  `json:"memory_used"`
	MemoryTotal       *uint64  `json:"memory_total"`
	GPUClock          *int     `json:"gpu_clock"`
	MaxGPUClock       *int     `json:"max_gpu_clock"`
	Temperature       *float64 `json:"temperature"`
	PowerUsage        *float64 `json:"power_usage"`
	PowerLimit        *float64 `json:"power_limit"`
	ThrottleBitmask   uint64   `json:"throttle_bitmask"`
	PCIe              gpu.PCIe `json:"pcie"`
	ECC               gpu.ECC  `json:"ecc"`
	Expected          struct {
		Bottleneck  string `json:"bottleneck"`
		HealthScore int    `json:"health_score"`
	} `json:"expected"`
}

func loadCases(t *testing.T) []sharedCase {
	t.Helper()
	raw, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatalf("read shared fixture: %v", err)
	}
	var f struct {
		Cases []sharedCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatalf("parse shared fixture: %v", err)
	}
	if len(f.Cases) == 0 {
		t.Fatal("shared fixture has no cases")
	}
	return f.Cases
}

func TestMatchesPythonOnSharedCases(t *testing.T) {
	for _, c := range loadCases(t) {
		m := gpu.Metrics{
			GPUUtilization: c.GPUUtilization, MemoryUtilization: c.MemoryUtilization,
			MemoryUsed: c.MemoryUsed, MemoryTotal: c.MemoryTotal,
			GPUClock: c.GPUClock, MaxGPUClock: c.MaxGPUClock, Temperature: c.Temperature,
			PowerUsage: c.PowerUsage, PowerLimit: c.PowerLimit,
			Throttle: gpu.DecodeThrottle(c.ThrottleBitmask), PCIe: c.PCIe, ECC: c.ECC,
		}
		b := Classify(m)
		if b != c.Expected.Bottleneck {
			t.Errorf("%s: bottleneck %q, Python says %q", c.ID, b, c.Expected.Bottleneck)
		}
		if h := HealthScore(m, b); h != c.Expected.HealthScore {
			t.Errorf("%s: health score %d, Python says %d", c.ID, h, c.Expected.HealthScore)
		}
	}
}

func TestMemoryBoundKernelReadsFullUtilization(t *testing.T) {
	gu, mu := 100, 100
	if got := Classify(gpu.Metrics{GPUUtilization: &gu, MemoryUtilization: &mu}); got != MemoryBandwidthBound {
		t.Fatalf("got %q, want %q", got, MemoryBandwidthBound)
	}
}

func TestUnknownWhenUtilizationUnavailable(t *testing.T) {
	if got := Classify(gpu.Metrics{}); got != Unknown {
		t.Fatalf("got %q, want %q", got, Unknown)
	}
}
