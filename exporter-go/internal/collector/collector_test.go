package collector

import (
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/btursunbayev/nvsonar/exporter-go/internal/cache"
	"github.com/btursunbayev/nvsonar/exporter-go/internal/gpu"
)

type fakeSource struct{ m gpu.Metrics }

func (f fakeSource) Name(int) string                  { return "Test GPU" }
func (f fakeSource) Collect(int) (gpu.Metrics, error) { return f.m, nil }

func intp(v int) *int           { return &v }
func floatp(v float64) *float64 { return &v }
func uint64p(v uint64) *uint64  { return &v }

func registry(t *testing.T, m gpu.Metrics) *prometheus.Registry {
	t.Helper()
	c := cache.New(fakeSource{m}, []int{0}, time.Hour)
	c.Refresh()
	reg := prometheus.NewRegistry()
	reg.MustRegister(New(c))
	return reg
}

func fullMetrics() gpu.Metrics {
	return gpu.Metrics{
		GPUUtilization: intp(100), MemoryUtilization: intp(100),
		MemoryUsed: uint64p(1 << 30), MemoryTotal: uint64p(6 << 30),
		GPUClock: intp(1500), MemoryClock: intp(6000), MaxGPUClock: intp(1800),
		Temperature: floatp(60), PowerUsage: floatp(70), PowerLimit: floatp(80), FanSpeed: intp(40),
		Throttle:     gpu.DecodeThrottle(0x4),
		PCIe:         gpu.PCIe{CurrentLinkGen: 3, MaxLinkGen: 3, CurrentLinkWidth: 16, MaxLinkWidth: 16},
		ProcessCount: 2,
	}
}

// The families and types the Python exporter emits (exporter/collectors.py).
var pythonFamilies = map[string]string{
	"nvsonar_exporter_scrape_duration_seconds": "gauge", "nvsonar_exporter_scrape_errors_total": "counter",
	"nvsonar_gpu_bottleneck": "gauge", "nvsonar_gpu_clock_reduction_ratio": "gauge",
	"nvsonar_gpu_ecc_errors_total": "counter", "nvsonar_gpu_fan_speed_ratio": "gauge",
	"nvsonar_gpu_health_score": "gauge", "nvsonar_gpu_memory_clock_mhz": "gauge",
	"nvsonar_gpu_memory_controller_ratio": "gauge", "nvsonar_gpu_memory_total_bytes": "gauge",
	"nvsonar_gpu_memory_used_bytes": "gauge", "nvsonar_gpu_pcie_link_gen": "gauge",
	"nvsonar_gpu_pcie_link_width": "gauge", "nvsonar_gpu_power_limit_watts": "gauge",
	"nvsonar_gpu_power_watts": "gauge", "nvsonar_gpu_process_count": "gauge",
	"nvsonar_gpu_sm_clock_mhz": "gauge", "nvsonar_gpu_temp_celsius": "gauge",
	"nvsonar_gpu_throttle_active": "gauge", "nvsonar_gpu_utilization_ratio": "gauge",
}

func TestSameFamiliesAsPythonExporter(t *testing.T) {
	families, err := registry(t, fullMetrics()).Gather()
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]string{}
	for _, f := range families {
		got[f.GetName()] = strings.ToLower(f.GetType().String())
	}
	var missing, extra []string
	for name, typ := range pythonFamilies {
		if got[name] != typ {
			missing = append(missing, name+":"+typ)
		}
	}
	for name := range got {
		if _, ok := pythonFamilies[name]; !ok {
			extra = append(extra, name)
		}
	}
	sort.Strings(missing)
	sort.Strings(extra)
	if len(missing) > 0 || len(extra) > 0 {
		t.Fatalf("family mismatch: missing %v, extra %v", missing, extra)
	}
}

func TestBottleneckIsOneHot(t *testing.T) {
	reg := registry(t, fullMetrics())
	want := `
# HELP nvsonar_gpu_bottleneck Active bottleneck classification (1 if active, 0 otherwise)
# TYPE nvsonar_gpu_bottleneck gauge
nvsonar_gpu_bottleneck{gpu="0",name="Test GPU",type="balanced"} 0
nvsonar_gpu_bottleneck{gpu="0",name="Test GPU",type="compute_bound"} 0
nvsonar_gpu_bottleneck{gpu="0",name="Test GPU",type="data_starved"} 0
nvsonar_gpu_bottleneck{gpu="0",name="Test GPU",type="idle"} 0
nvsonar_gpu_bottleneck{gpu="0",name="Test GPU",type="memory_bandwidth_bound"} 1
nvsonar_gpu_bottleneck{gpu="0",name="Test GPU",type="memory_capacity_bound"} 0
nvsonar_gpu_bottleneck{gpu="0",name="Test GPU",type="power_limited"} 0
nvsonar_gpu_bottleneck{gpu="0",name="Test GPU",type="thermal_throttled"} 0
nvsonar_gpu_bottleneck{gpu="0",name="Test GPU",type="unknown"} 0
`
	if err := testutil.GatherAndCompare(reg, strings.NewReader(want), "nvsonar_gpu_bottleneck"); err != nil {
		t.Fatal(err)
	}
}

func TestUnavailableReadingsAreOmitted(t *testing.T) {
	m := fullMetrics()
	m.Temperature, m.FanSpeed = nil, nil
	reg := registry(t, m)
	for name, want := range map[string]int{
		"nvsonar_gpu_temp_celsius": 0, "nvsonar_gpu_fan_speed_ratio": 0, "nvsonar_gpu_power_watts": 1,
	} {
		if n, err := testutil.GatherAndCount(reg, name); err != nil || n != want {
			t.Errorf("%s: %d samples (err %v), want %d", name, n, err, want)
		}
	}
}
