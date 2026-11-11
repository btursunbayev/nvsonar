package gpu

import "testing"

func TestDecodeThrottle(t *testing.T) {
	th := DecodeThrottle(0x44) // software power cap + hardware thermal slowdown
	if len(th.Active) != 2 || !th.Has("Software Power Cap") || !th.Has("Hardware Thermal Slowdown") {
		t.Fatalf("unexpected reasons: %+v", th.Active)
	}
	if th.WorstSeverity() != "critical" {
		t.Fatalf("worst severity %q, want critical", th.WorstSeverity())
	}
	if DecodeThrottle(0).WorstSeverity() != "ok" {
		t.Fatal("no reasons should be ok")
	}
}

func TestDerivedPercentages(t *testing.T) {
	used, total := uint64(3), uint64(4)
	clock, maxClock := 1350, 1800
	usage, limit := 150.0, 300.0
	m := Metrics{MemoryUsed: &used, MemoryTotal: &total, GPUClock: &clock, MaxGPUClock: &maxClock,
		PowerUsage: &usage, PowerLimit: &limit}
	if got := *m.MemoryUsedPct(); got != 75 {
		t.Errorf("memory used %v, want 75", got)
	}
	if got := *m.ClockReductionPct(); got != 25 {
		t.Errorf("clock reduction %v, want 25", got)
	}
	if got := *m.PowerUsedPct(); got != 50 {
		t.Errorf("power used %v, want 50", got)
	}
	if (Metrics{}).MemoryUsedPct() != nil {
		t.Error("unknown memory should give nil")
	}
}
