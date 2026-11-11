// Package collector translates cached snapshots into Prometheus metrics with
// the same names, labels and types as the Python exporter (exporter/collectors.py),
// so dashboards/nvsonar.json works with either exporter.
package collector

import (
	"strconv"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/btursunbayev/nvsonar/exporter-go/internal/analysis"
	"github.com/btursunbayev/nvsonar/exporter-go/internal/cache"
)

const namespace = "nvsonar"

type gauge struct {
	desc    *prometheus.Desc
	extract func(cache.Snapshot) *float64
}

func f(v float64) *float64 { return &v }

func ratio(p *int) *float64 {
	if p == nil {
		return nil
	}
	return f(float64(*p) / 100)
}

var gpuLabels = []string{"gpu", "name"}

func desc(name, help string, labels ...string) *prometheus.Desc {
	return prometheus.NewDesc(namespace+"_"+name, help, labels, nil)
}

// Collector implements prometheus.Collector over a snapshot cache.
type Collector struct {
	cache *cache.Cache

	gauges         []gauge
	bottleneck     *prometheus.Desc
	throttle       *prometheus.Desc
	ecc            *prometheus.Desc
	pcieGen        *prometheus.Desc
	pcieWidth      *prometheus.Desc
	processes      *prometheus.Desc
	scrapeDuration *prometheus.Desc
	scrapeErrors   *prometheus.Desc
}

// New builds the collector.
func New(c *cache.Cache) *Collector {
	return &Collector{
		cache: c,
		gauges: []gauge{
			{desc("gpu_temp_celsius", "GPU temperature in degrees Celsius", gpuLabels...),
				func(s cache.Snapshot) *float64 { return s.Metrics.Temperature }},
			{desc("gpu_utilization_ratio", "GPU compute utilization (0-1)", gpuLabels...),
				func(s cache.Snapshot) *float64 { return ratio(s.Metrics.GPUUtilization) }},
			{desc("gpu_memory_controller_ratio", "GPU memory controller utilization (0-1)", gpuLabels...),
				func(s cache.Snapshot) *float64 { return ratio(s.Metrics.MemoryUtilization) }},
			{desc("gpu_memory_used_bytes", "GPU memory used in bytes", gpuLabels...),
				func(s cache.Snapshot) *float64 {
					if s.Metrics.MemoryUsed == nil {
						return nil
					}
					return f(float64(*s.Metrics.MemoryUsed))
				}},
			{desc("gpu_memory_total_bytes", "GPU memory total in bytes", gpuLabels...),
				func(s cache.Snapshot) *float64 {
					if s.Metrics.MemoryTotal == nil {
						return nil
					}
					return f(float64(*s.Metrics.MemoryTotal))
				}},
			{desc("gpu_power_watts", "GPU power draw in watts", gpuLabels...),
				func(s cache.Snapshot) *float64 { return s.Metrics.PowerUsage }},
			{desc("gpu_power_limit_watts", "GPU power limit in watts", gpuLabels...),
				func(s cache.Snapshot) *float64 { return s.Metrics.PowerLimit }},
			{desc("gpu_sm_clock_mhz", "GPU SM clock in MHz", gpuLabels...),
				func(s cache.Snapshot) *float64 {
					if s.Metrics.GPUClock == nil {
						return nil
					}
					return f(float64(*s.Metrics.GPUClock))
				}},
			{desc("gpu_memory_clock_mhz", "GPU memory clock in MHz", gpuLabels...),
				func(s cache.Snapshot) *float64 {
					if s.Metrics.MemoryClock == nil {
						return nil
					}
					return f(float64(*s.Metrics.MemoryClock))
				}},
			{desc("gpu_fan_speed_ratio", "GPU fan speed (0-1)", gpuLabels...),
				func(s cache.Snapshot) *float64 { return ratio(s.Metrics.FanSpeed) }},
			{desc("gpu_clock_reduction_ratio", "Reduction of current SM clock vs max (0-1)", gpuLabels...),
				func(s cache.Snapshot) *float64 {
					cd := s.Metrics.ClockReductionPct()
					if cd == nil {
						return nil
					}
					return f(*cd / 100)
				}},
			{desc("gpu_health_score", "NVSonar GPU health score (0-100)", gpuLabels...),
				func(s cache.Snapshot) *float64 { return f(float64(s.HealthScore)) }},
		},
		bottleneck: desc("gpu_bottleneck", "Active bottleneck classification (1 if active, 0 otherwise)",
			"gpu", "name", "type"),
		throttle:  desc("gpu_throttle_active", "Active throttle reason (1 if active)", "gpu", "name", "reason", "severity"),
		ecc:       desc("gpu_ecc_errors_total", "Cumulative ECC error counts", "gpu", "name", "severity"),
		pcieGen:   desc("gpu_pcie_link_gen", "Current PCIe link generation", gpuLabels...),
		pcieWidth: desc("gpu_pcie_link_width", "Current PCIe link width (lanes)", gpuLabels...),
		processes: desc("gpu_process_count", "Number of processes using this GPU", gpuLabels...),
		scrapeDuration: desc("exporter_scrape_duration_seconds",
			"Duration of the last background collection cycle"),
		scrapeErrors: desc("exporter_scrape_errors_total", "Total background collection errors since startup"),
	}
}

// Describe sends every metric descriptor.
func (c *Collector) Describe(ch chan<- *prometheus.Desc) {
	for _, g := range c.gauges {
		ch <- g.desc
	}
	for _, d := range []*prometheus.Desc{c.bottleneck, c.throttle, c.ecc, c.pcieGen, c.pcieWidth,
		c.processes, c.scrapeDuration, c.scrapeErrors} {
		ch <- d
	}
}

// Collect emits the current snapshots; values NVML could not read are omitted.
func (c *Collector) Collect(ch chan<- prometheus.Metric) {
	snaps := c.cache.All()
	for _, g := range c.gauges {
		for _, s := range snaps {
			if v := g.extract(s); v != nil {
				ch <- prometheus.MustNewConstMetric(g.desc, prometheus.GaugeValue, *v, strconv.Itoa(s.GPUIndex), s.GPUName)
			}
		}
	}
	for _, s := range snaps {
		idx := strconv.Itoa(s.GPUIndex)
		for _, t := range analysis.BottleneckTypes {
			v := 0.0
			if s.Bottleneck == t {
				v = 1
			}
			ch <- prometheus.MustNewConstMetric(c.bottleneck, prometheus.GaugeValue, v, idx, s.GPUName, t)
		}
		for _, r := range s.Metrics.Throttle.Active {
			ch <- prometheus.MustNewConstMetric(c.throttle, prometheus.GaugeValue, 1, idx, s.GPUName, r.Name, r.Severity)
		}
		ch <- prometheus.MustNewConstMetric(c.ecc, prometheus.CounterValue, float64(s.Metrics.ECC.Correctable),
			idx, s.GPUName, "correctable")
		ch <- prometheus.MustNewConstMetric(c.ecc, prometheus.CounterValue, float64(s.Metrics.ECC.Uncorrectable),
			idx, s.GPUName, "uncorrectable")
		if s.Metrics.PCIe.MaxLinkGen > 0 {
			ch <- prometheus.MustNewConstMetric(c.pcieGen, prometheus.GaugeValue,
				float64(s.Metrics.PCIe.CurrentLinkGen), idx, s.GPUName)
			ch <- prometheus.MustNewConstMetric(c.pcieWidth, prometheus.GaugeValue,
				float64(s.Metrics.PCIe.CurrentLinkWidth), idx, s.GPUName)
		}
		ch <- prometheus.MustNewConstMetric(c.processes, prometheus.GaugeValue, float64(s.Metrics.ProcessCount),
			idx, s.GPUName)
	}
	ch <- prometheus.MustNewConstMetric(c.scrapeDuration, prometheus.GaugeValue, c.cache.LastScrapeDuration().Seconds())
	ch <- prometheus.MustNewConstMetric(c.scrapeErrors, prometheus.CounterValue, float64(c.cache.ScrapeErrors()))
}
