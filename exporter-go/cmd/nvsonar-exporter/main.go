// Command nvsonar-exporter serves NVSonar GPU metrics, bottleneck classification
// and health score for Prometheus. Flags and metric names match `nvsonar exporter`.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/btursunbayev/nvsonar/exporter-go/internal/cache"
	"github.com/btursunbayev/nvsonar/exporter-go/internal/collector"
	"github.com/btursunbayev/nvsonar/exporter-go/internal/nvmlsource"
)

var version = "dev" // set at build time with -ldflags "-X main.version=..."

// parseGPUs resolves --gpu like the Python CLI: "", "all" or "-1" select every GPU,
// otherwise a comma-separated list of indices.
func parseGPUs(value string, count int) ([]int, error) {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" || value == "all" || value == "-1" {
		out := make([]int, count)
		for i := range out {
			out[i] = i
		}
		return out, nil
	}
	var out []int
	for _, part := range strings.Split(value, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		i, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("invalid GPU index: %q", part)
		}
		if i < 0 || i >= count {
			return nil, fmt.Errorf("GPU index %d out of range (found %d GPUs)", i, count)
		}
		out = append(out, i)
	}
	return out, nil
}

func main() {
	port := flag.Int("port", 9100, "HTTP port to serve metrics on")
	gpus := flag.String("gpu", "", "GPU index or comma-separated list (default: all)")
	interval := flag.Duration("interval", 2*time.Second, "background collection interval")
	showVersion := flag.Bool("version", false, "print version and exit")
	flag.Parse()

	if *showVersion {
		fmt.Println(version)
		return
	}

	src, err := nvmlsource.Init()
	if err != nil {
		log.Fatalf("failed to initialize NVML, no NVIDIA GPU found: %v", err)
	}
	defer src.Shutdown()

	count, err := src.Count()
	if err != nil || count == 0 {
		log.Fatal("no GPUs detected")
	}
	indices, err := parseGPUs(*gpus, count)
	if err != nil {
		log.Fatal(err)
	}

	c := cache.New(src, indices, *interval)
	c.Start()
	defer c.Stop()

	reg := prometheus.NewRegistry()
	reg.MustRegister(collector.New(c), collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}),
		collectors.NewGoCollector())

	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))
	srv := &http.Server{Addr: fmt.Sprintf(":%d", *port), Handler: mux, ReadHeaderTimeout: 5 * time.Second}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdown)
	}()

	log.Printf("nvsonar-exporter %s on http://0.0.0.0:%d/metrics (%d GPU(s), interval %s)",
		version, *port, len(indices), *interval)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}
