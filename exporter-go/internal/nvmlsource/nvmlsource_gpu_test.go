//go:build gpu

// Runs against a real NVIDIA GPU: go test -tags gpu ./...
package nvmlsource

import "testing"

func TestCollectFromRealGPU(t *testing.T) {
	src, err := Init()
	if err != nil {
		t.Skipf("no NVML: %v", err)
	}
	defer src.Shutdown()
	n, err := src.Count()
	if err != nil || n == 0 {
		t.Skip("no GPUs")
	}
	m, err := src.Collect(0)
	if err != nil {
		t.Fatal(err)
	}
	if m.GPUUtilization == nil || m.MemoryTotal == nil || *m.MemoryTotal == 0 {
		t.Fatalf("expected utilization and memory readings, got %+v", m)
	}
	if src.Name(0) == "" {
		t.Fatal("empty device name")
	}
}
