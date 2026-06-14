# Dashboard test stack

Runs Prometheus + Grafana locally and points them at a running `nvsonar exporter`. Use this to verify [../nvsonar.json](../nvsonar.json) renders correctly before shipping a release.

## Usage

In one terminal, start the exporter:

```bash
nvsonar exporter --port 9100
```

In another terminal, bring up the stack:

```bash
cd dashboards/test-stack
docker compose up
```

Then:

1. Confirm Prometheus is scraping — open <http://localhost:9090/graph>, query `nvsonar_gpu_health_score`, expect a non-empty time-series.
2. Open Grafana — <http://localhost:3000> (anonymous admin, no login).
3. Import the dashboard — left sidebar → **Dashboards → New → Import → Upload JSON file** → pick [../nvsonar.json](../nvsonar.json) → select **Prometheus** as the datasource → **Import**.
4. Walk every panel:
   - Health Score should plot a 0–100 line per GPU.
   - Bottleneck distribution should show a stacked area with `idle` dominant when no workload runs.
   - Stress the GPU (`nvsonar benchmark`) and confirm the active bottleneck flips to `compute_bound` or `memory_bandwidth_bound`.
   - Temperature, Power, Util, VRAM panels should all populate.
   - Throttle table is empty unless a throttle reason is active.
   - ECC rate is zero unless the GPU is failing.
   - Exporter self-monitoring panels show low scrape duration and zero errors.

## Tear down

```bash
docker compose down
```

## Notes

- Uses `network_mode: host` so the containers reach the host's `localhost:9100` (the exporter). Linux only.
- Grafana runs with anonymous admin access; do not expose this stack on a public network.
