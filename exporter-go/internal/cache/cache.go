// Package cache keeps the latest snapshot per GPU, refreshed by a background
// goroutine so a Prometheus scrape never blocks on NVML (as exporter/cache.py).
package cache

import (
	"sync"
	"sync/atomic"
	"time"

	"github.com/btursunbayev/nvsonar/exporter-go/internal/analysis"
	"github.com/btursunbayev/nvsonar/exporter-go/internal/gpu"
)

// Source provides metrics for GPUs by index.
type Source interface {
	Name(i int) string
	Collect(i int) (gpu.Metrics, error)
}

// Snapshot is one GPU's latest reading plus its classification and health score.
type Snapshot struct {
	GPUIndex    int
	GPUName     string
	Metrics     gpu.Metrics
	Bottleneck  string
	HealthScore int
	Timestamp   time.Time
}

// Cache refreshes snapshots every interval until Stop is called.
type Cache struct {
	src      Source
	indices  []int
	names    map[int]string
	interval time.Duration

	mu        sync.RWMutex
	snapshots map[int]Snapshot

	scrapeErrors       atomic.Uint64
	lastScrapeDuration atomic.Int64 // nanoseconds

	stop chan struct{}
	done chan struct{}
}

// New prepares a cache for the given GPU indices.
func New(src Source, indices []int, interval time.Duration) *Cache {
	c := &Cache{src: src, indices: indices, interval: interval, names: map[int]string{},
		snapshots: map[int]Snapshot{}, stop: make(chan struct{}), done: make(chan struct{})}
	for _, i := range indices {
		c.names[i] = src.Name(i)
	}
	return c
}

// Start runs the refresh loop in a goroutine; the first refresh happens immediately.
func (c *Cache) Start() { go c.loop() }

// Stop ends the refresh loop and waits for it to exit.
func (c *Cache) Stop() {
	close(c.stop)
	<-c.done
}

// Refresh collects every GPU once.
func (c *Cache) Refresh() {
	start := time.Now()
	for _, i := range c.indices {
		m, err := c.src.Collect(i)
		if err != nil {
			c.scrapeErrors.Add(1)
			continue
		}
		b := analysis.Classify(m)
		snap := Snapshot{GPUIndex: i, GPUName: c.names[i], Metrics: m, Bottleneck: b,
			HealthScore: analysis.HealthScore(m, b), Timestamp: time.Now()}
		c.mu.Lock()
		c.snapshots[i] = snap
		c.mu.Unlock()
	}
	c.lastScrapeDuration.Store(int64(time.Since(start)))
}

func (c *Cache) loop() {
	defer close(c.done)
	t := time.NewTicker(c.interval)
	defer t.Stop()
	for {
		c.Refresh()
		select {
		case <-c.stop:
			return
		case <-t.C:
		}
	}
}

// All returns the latest snapshots in GPU index order.
func (c *Cache) All() []Snapshot {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]Snapshot, 0, len(c.snapshots))
	for _, i := range c.indices {
		if s, ok := c.snapshots[i]; ok {
			out = append(out, s)
		}
	}
	return out
}

// ScrapeErrors is the number of failed collections since start.
func (c *Cache) ScrapeErrors() uint64 { return c.scrapeErrors.Load() }

// LastScrapeDuration is how long the last refresh of all GPUs took.
func (c *Cache) LastScrapeDuration() time.Duration {
	return time.Duration(c.lastScrapeDuration.Load())
}
