package gpu

// Reason is one decoded clock throttle reason.
type Reason struct {
	Bit      uint64
	Name     string
	Severity string // "info", "warning" or "critical"
}

// Reasons maps NVML clock-throttle bits to names and severities, as in throttle.py.
var Reasons = []Reason{
	{0x0001, "GPU Idle", "info"},
	{0x0002, "Applications Clocks Setting", "info"},
	{0x0004, "Software Power Cap", "warning"},
	{0x0008, "Hardware Slowdown", "critical"},
	{0x0010, "Sync Boost", "info"},
	{0x0020, "Software Thermal Slowdown", "warning"},
	{0x0040, "Hardware Thermal Slowdown", "critical"},
	{0x0080, "Hardware Power Brake", "critical"},
}

// Throttle is the decoded throttle state of one GPU.
type Throttle struct {
	Bitmask uint64   `json:"bitmask"`
	Active  []Reason `json:"-"`
}

// DecodeThrottle returns the reasons whose bits are set, in table order.
func DecodeThrottle(bitmask uint64) Throttle {
	t := Throttle{Bitmask: bitmask}
	for _, r := range Reasons {
		if bitmask&r.Bit != 0 {
			t.Active = append(t.Active, r)
		}
	}
	return t
}

var severityRank = map[string]int{"critical": 3, "warning": 2, "info": 1}

// WorstSeverity is "ok" with no active reasons, otherwise the most severe one.
func (t Throttle) WorstSeverity() string {
	if len(t.Active) == 0 {
		return "ok"
	}
	worst := t.Active[0]
	for _, r := range t.Active[1:] {
		if severityRank[r.Severity] > severityRank[worst.Severity] {
			worst = r
		}
	}
	return worst.Severity
}

// Has reports whether a reason with the given name is active.
func (t Throttle) Has(name string) bool {
	for _, r := range t.Active {
		if r.Name == name {
			return true
		}
	}
	return false
}
