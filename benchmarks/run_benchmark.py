"""Sub-microsecond latency and throughput benchmark for Dynamic Hedge Kalman Filter."""

import gc
import statistics
import time
from typing import List
import numpy as np
from pairs_coint.kalman import DynamicHedgeKalmanFilter


def run_benchmark(num_ticks: int = 1_000_000) -> None:
    rng = np.random.default_rng(42)

    print(f"[*] Generating {num_ticks:,} synthetic asynchronous tick pairs...")
    x = 100.0 + np.cumsum(rng.normal(0.0, 0.1, size=num_ticks))
    y = 1.85 * x + 5.0 + rng.normal(0.0, 0.05, size=num_ticks)

    kf = DynamicHedgeKalmanFilter(delta=1e-4, observation_variance_r=0.01, initial_beta=1.5)
    latencies_ns: List[int] = [0] * num_ticks

    print("[*] Benchmarking online O(1) Kalman updates (Disabling GC)...")
    gc.disable()

    start_wall = time.perf_counter_ns()
    for i in range(num_ticks):
        t0 = time.perf_counter_ns()
        _ = kf.update(float(x[i]), float(y[i]))
        t1 = time.perf_counter_ns()
        latencies_ns[i] = t1 - t0

    end_wall = time.perf_counter_ns()
    gc.enable()

    total_duration_sec = (end_wall - start_wall) / 1e9
    throughput = num_ticks / total_duration_sec

    latencies_ns.sort()
    p50 = latencies_ns[int(num_ticks * 0.50)]
    p90 = latencies_ns[int(num_ticks * 0.90)]
    p99 = latencies_ns[int(num_ticks * 0.99)]
    p99_9 = latencies_ns[int(num_ticks * 0.999)]
    mean_lat = statistics.mean(latencies_ns)

    print("\n" + "=" * 60)
    print("           PAIRS-COINT KALMAN PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"Processed Tick Pairs     : {num_ticks:,}")
    print(f"Final Estimated Beta     : {kf.current_beta:.4f}")
    print(f"Total Wall Duration      : {total_duration_sec:.4f} sec")
    print(f"Kalman Throughput Rate   : {throughput:,.0f} updates/sec")
    print("-" * 60)
    print(f"Mean Latency             : {mean_lat:,.0f} ns ({mean_lat / 1000.0:.2f} µs)")
    print(f"P50 Latency (Median)     : {p50:,.0f} ns ({p50 / 1000.0:.2f} µs)")
    print(f"P90 Latency              : {p90:,.0f} ns ({p90 / 1000.0:.2f} µs)")
    print(f"P99 Latency              : {p99:,.0f} ns ({p99 / 1000.0:.2f} µs)")
    print(f"P99.9 Latency (Tail)     : {p99_9:,.0f} ns ({p99_9 / 1000.0:.2f} µs)")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark(num_ticks=1_000_000)
