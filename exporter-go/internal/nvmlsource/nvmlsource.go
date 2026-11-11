// Package nvmlsource reads GPU metrics through NVIDIA's go-nvml bindings,
// mirroring nvsonar.monitor.MetricsCollector.collect. A reading NVML cannot
// provide is left nil, as the Python implementation leaves it None.
package nvmlsource

import (
	"fmt"

	"github.com/NVIDIA/go-nvml/pkg/nvml"

	"github.com/btursunbayev/nvsonar/exporter-go/internal/gpu"
)

// Source reads metrics from NVML. Init must succeed before use.
type Source struct {
	devices map[int]nvml.Device
}

// Init loads the NVIDIA driver library and initializes NVML.
func Init() (*Source, error) {
	if ret := nvml.Init(); ret != nvml.SUCCESS {
		return nil, fmt.Errorf("NVML init failed: %s", nvml.ErrorString(ret))
	}
	return &Source{devices: map[int]nvml.Device{}}, nil
}

// Shutdown releases NVML.
func (s *Source) Shutdown() { nvml.Shutdown() }

// Count returns the number of GPUs NVML can see.
func (s *Source) Count() (int, error) {
	n, ret := nvml.DeviceGetCount()
	if ret != nvml.SUCCESS {
		return 0, fmt.Errorf("device count: %s", nvml.ErrorString(ret))
	}
	return n, nil
}

func (s *Source) device(i int) (nvml.Device, error) {
	if d, ok := s.devices[i]; ok {
		return d, nil
	}
	d, ret := nvml.DeviceGetHandleByIndex(i)
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("GPU %d: %s", i, nvml.ErrorString(ret))
	}
	s.devices[i] = d
	return d, nil
}

// Name returns the device name, or "GPU<i>" when NVML cannot report it.
func (s *Source) Name(i int) string {
	d, err := s.device(i)
	if err != nil {
		return fmt.Sprintf("GPU%d", i)
	}
	name, ret := d.GetName()
	if ret != nvml.SUCCESS {
		return fmt.Sprintf("GPU%d", i)
	}
	return name
}

func intp(v int) *int           { return &v }
func floatp(v float64) *float64 { return &v }
func uint64p(v uint64) *uint64  { return &v }
func ok(ret nvml.Return) bool   { return ret == nvml.SUCCESS }

// Collect reads one snapshot for GPU i.
func (s *Source) Collect(i int) (gpu.Metrics, error) {
	d, err := s.device(i)
	if err != nil {
		return gpu.Metrics{}, err
	}
	var m gpu.Metrics

	if u, ret := d.GetUtilizationRates(); ok(ret) {
		m.GPUUtilization, m.MemoryUtilization = intp(int(u.Gpu)), intp(int(u.Memory))
	}
	if mem, ret := d.GetMemoryInfo(); ok(ret) {
		m.MemoryUsed, m.MemoryTotal = uint64p(mem.Used), uint64p(mem.Total)
	}
	if c, ret := d.GetClockInfo(nvml.CLOCK_GRAPHICS); ok(ret) {
		m.GPUClock = intp(int(c))
	}
	if c, ret := d.GetClockInfo(nvml.CLOCK_MEM); ok(ret) {
		m.MemoryClock = intp(int(c))
	}
	if c, ret := d.GetMaxClockInfo(nvml.CLOCK_GRAPHICS); ok(ret) {
		m.MaxGPUClock = intp(int(c))
	}
	if t, ret := d.GetTemperature(nvml.TEMPERATURE_GPU); ok(ret) {
		m.Temperature = floatp(float64(t))
	}
	if p, ret := d.GetPowerUsage(); ok(ret) {
		m.PowerUsage = floatp(float64(p) / 1000)
	}
	if p, ret := d.GetPowerManagementLimit(); ok(ret) {
		m.PowerLimit = floatp(float64(p) / 1000)
	}
	if f, ret := d.GetFanSpeed(); ok(ret) {
		m.FanSpeed = intp(int(f))
	}
	if b, ret := d.GetCurrentClocksThrottleReasons(); ok(ret) {
		m.Throttle = gpu.DecodeThrottle(b)
	}

	if v, ret := d.GetCurrPcieLinkGeneration(); ok(ret) {
		m.PCIe.CurrentLinkGen = v
	}
	if v, ret := d.GetMaxPcieLinkGeneration(); ok(ret) {
		m.PCIe.MaxLinkGen = v
	}
	if v, ret := d.GetCurrPcieLinkWidth(); ok(ret) {
		m.PCIe.CurrentLinkWidth = v
	}
	if v, ret := d.GetMaxPcieLinkWidth(); ok(ret) {
		m.PCIe.MaxLinkWidth = v
	}

	// as in get_ecc_info: counters are read whenever the ECC mode query succeeds
	if _, _, ret := d.GetEccMode(); ok(ret) {
		if v, ret := d.GetTotalEccErrors(nvml.MEMORY_ERROR_TYPE_CORRECTED, nvml.VOLATILE_ECC); ok(ret) {
			m.ECC.Correctable = v
		}
		if v, ret := d.GetTotalEccErrors(nvml.MEMORY_ERROR_TYPE_UNCORRECTED, nvml.VOLATILE_ECC); ok(ret) {
			m.ECC.Uncorrectable = v
		}
	}

	if procs, ret := d.GetComputeRunningProcesses(); ok(ret) {
		m.ProcessCount = len(procs)
	}
	return m, nil
}
