import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import aiohttp

CONCURRENCY_LEVEL = [1]
PROMPT_LENGTH = [128, 1024]
FIXED_OUTPUT_TOKENS = 128
NUM_REPEATS = 5
REQUEST_TIMEOUT = 180
STATS_POLL_INTERVAL = 0.25

FILLER_SENTENCE = (
    "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
)


def build_prompt(target_tokens: int) -> str:
    """ Builds a prompt approximately of terget_tokens tokens by repeating text."""
    approx_tokens_per_sentence =16
    repeats = max(1, target_tokens// approx_tokens_per_sentence)
    base = FILLER_SENTENCE*repeats
    return  f"Countinue the following text in a coherent for a while:\n{base}"


@dataclass
class RequestResult:
    ok: bool
    client_start: float
    client_end: float
    client_latency: float
    server_ttft: float | None
    server_total_latency: float | None
    tokens_generated: int | None
    error: str | None = None

    @property
    def network_delay(self) -> float | None:
        """Real per-request network overhead: the full client-observed round
        trip minus the server's own reported processing time. Measured
        directly from each request/response pair -- no separate ping probe,
        no assumptions about symmetric up/down split."""
        if not self.ok or self.server_total_latency is None:
            return None
        return self.client_latency - self.server_total_latency

    @property
    def total_delay(self) -> float | None:
        """network_delay + server_total_latency. Should track client_latency
        almost exactly -- kept as an explicit, separate field so the split is
        visible in the CSV, and so client_latency vs total_delay acts as a
        built-in sanity check (they should never meaningfully diverge)."""
        if self.network_delay is None or self.server_total_latency is None:
            return None
        return self.network_delay + self.server_total_latency

    @property
    def client_tok_s(self) -> float | None:
        if not self.ok or self.tokens_generated is None or self.server_ttft is None:
            return None
        gen_time = self.client_latency - self.server_ttft
        if gen_time <= 0:
            return None
        return self.tokens_generated / gen_time


@dataclass
class RunSummary:
    deployment: str
    prompt_tokens_target: int
    concurrency: int
    repeat_index: int
    n_requests: int
    n_success: int
    p50_latency: float | None
    p95_latency: float | None
    mean_latency: float | None
    stddev_latency: float | None
    p50_ttft: float | None
    p95_ttft: float | None
    mean_ttft: float | None
    stddev_ttft: float | None
    # --- client-side delay (explicit alias of client_latency, named to sit
    # alongside network_delay / server_ttft in the CSV) ---
    p50_client_delay: float | None
    p95_client_delay: float | None
    mean_client_delay: float | None
    stddev_client_delay: float | None
    # --- network delay: real measured overhead, isolated from server compute ---
    p50_network_delay: float | None
    p95_network_delay: float | None
    mean_network_delay: float | None
    stddev_network_delay: float | None
    # --- total delay: network_delay + server_total_latency (sanity check vs client_delay) ---
    mean_total_delay: float | None
    mean_tok_s: float | None
    stddev_tok_s: float | None
    mean_cpu_percent: float | None
    max_cpu_percent: float | None
    mean_ram_percent: float | None
    max_ram_percent: float | None
    gpu_available: bool
    mean_gpu_percent: float | None
    max_gpu_percent: float | None
    mean_vram_percent: float | None
    max_vram_percent: float | None
    n_stats_samples: int


async def send_request(
    session: aiohttp.ClientSession, url: str, prompt: str, max_tokens: int
) -> RequestResult:
    start = time.perf_counter()
    try:
        async with session.post(
            f"{url.rstrip('/')}/generate",
            json={"prompt": prompt, "max_tokens": max_tokens},
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
        ) as resp:
            data = await resp.json()
            end = time.perf_counter()
            return RequestResult(
                ok=resp.status == 200,
                client_start=start,
                client_end=end,
                client_latency=end - start,
                server_ttft=data.get("ttft"),
                server_total_latency=data.get("total_latency"),
                tokens_generated=data.get("tokens_generated", max_tokens),
            )
    except Exception as e:
        end = time.perf_counter()
        return RequestResult(
            ok=False,
            client_start=start,
            client_end=end,
            client_latency=end - start,
            server_ttft=None,
            server_total_latency=None,
            tokens_generated=None,
            error=str(e),
        )


async def poll_stats(
    session: aiohttp.ClientSession, url: str, samples: list, stop_event: asyncio.Event
):
    """
    Polls GET /stats repeatedly until stop_event is set. Runs concurrently
    alongside the request batch. Hits whatever `url` points to -- local
    server or the ngrok-tunneled Colab server -- so it transparently reports
    resource usage of WHICHEVER machine is actually running the model.
    """
    while not stop_event.is_set():
        try:
            async with session.get(
                f"{url.rstrip('/')}/stats",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    samples.append(data)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=STATS_POLL_INTERVAL)
        except asyncio.TimeoutError:
            pass


async def run_batch(
    url: str, prompt: str, max_tokens: int, concurrency: int
) -> tuple[list[RequestResult], list[dict]]:
    """Fires `concurrency` requests simultaneously while polling /stats in parallel."""
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        stats_samples: list[dict] = []
        stop_event = asyncio.Event()

        stats_task = asyncio.create_task(
            poll_stats(session, url, stats_samples, stop_event)
        )

        request_tasks = [
            send_request(session, url, prompt, max_tokens) for _ in range(concurrency)
        ]
        results = await asyncio.gather(*request_tasks)

        stop_event.set()
        await stats_task

        return results, stats_samples


def summarize(
    results: list[RequestResult],
    stats_samples: list[dict],
    deployment: str,
    prompt_tokens_target: int,
    concurrency: int,
    repeat_index: int,
) -> RunSummary:
    ok_results = [r for r in results if r.ok]
    latencies = [r.client_latency for r in ok_results]
    ttfts = [r.server_ttft for r in ok_results if r.server_ttft is not None]
    network_delays = [r.network_delay for r in ok_results if r.network_delay is not None]
    total_delays = [r.total_delay for r in ok_results if r.total_delay is not None]
    tok_s_vals = [r.client_tok_s for r in ok_results if r.client_tok_s is not None]

    def pctl(vals, p):
        if not vals:
            return None
        s = sorted(vals)
        k = (len(s) - 1) * (p / 100)
        f, c = int(k), min(int(k) + 1, len(s) - 1)
        if f == c:
            return s[f]
        return s[f] + (s[c] - s[f]) * (k - f)

    cpu_vals = [s["cpu_percent"] for s in stats_samples if "cpu_percent" in s]
    ram_vals = [s["ram_percent"] for s in stats_samples if "ram_percent" in s]
    gpu_available = any(s.get("gpu_available") for s in stats_samples) if stats_samples else False
    gpu_vals = [s["gpu_percent"] for s in stats_samples if "gpu_percent" in s]
    vram_vals = [s["vram_percent"] for s in stats_samples if "vram_percent" in s]

    return RunSummary(
        deployment=deployment,
        prompt_tokens_target=prompt_tokens_target,
        concurrency=concurrency,
        repeat_index=repeat_index,
        n_requests=len(results),
        n_success=len(ok_results),
        p50_latency=pctl(latencies, 50),
        p95_latency=pctl(latencies, 95),
        mean_latency=statistics.mean(latencies) if latencies else None,
        stddev_latency=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        p50_ttft=pctl(ttfts, 50),
        p95_ttft=pctl(ttfts, 95),
        mean_ttft=statistics.mean(ttfts) if ttfts else None,
        stddev_ttft=statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0,
        p50_client_delay=pctl(latencies, 50),
        p95_client_delay=pctl(latencies, 95),
        mean_client_delay=statistics.mean(latencies) if latencies else None,
        stddev_client_delay=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        p50_network_delay=pctl(network_delays, 50),
        p95_network_delay=pctl(network_delays, 95),
        mean_network_delay=statistics.mean(network_delays) if network_delays else None,
        stddev_network_delay=statistics.stdev(network_delays) if len(network_delays) > 1 else 0.0,
        mean_total_delay=statistics.mean(total_delays) if total_delays else None,
        mean_tok_s=statistics.mean(tok_s_vals) if tok_s_vals else None,
        stddev_tok_s=statistics.stdev(tok_s_vals) if len(tok_s_vals) > 1 else 0.0,
        mean_cpu_percent=statistics.mean(cpu_vals) if cpu_vals else None,
        max_cpu_percent=max(cpu_vals) if cpu_vals else None,
        mean_ram_percent=statistics.mean(ram_vals) if ram_vals else None,
        max_ram_percent=max(ram_vals) if ram_vals else None,
        gpu_available=gpu_available,
        mean_gpu_percent=statistics.mean(gpu_vals) if gpu_vals else None,
        max_gpu_percent=max(gpu_vals) if gpu_vals else None,
        mean_vram_percent=statistics.mean(vram_vals) if vram_vals else None,
        max_vram_percent=max(vram_vals) if vram_vals else None,
        n_stats_samples=len(stats_samples),
    )


async def main(url: str, name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{name}_raw.jsonl"
    stats_raw_path = out_dir / f"{name}_stats_raw.jsonl"
    summary_path = out_dir / f"{name}_summary.csv"

    all_summaries: list[RunSummary] = []

    with open(raw_path, "w") as raw_f, open(stats_raw_path, "w") as stats_f:
        for prompt_tokens in PROMPT_LENGTH:
            prompt = build_prompt(prompt_tokens)
            for concurrency in CONCURRENCY_LEVEL:
                for repeat_index in range(1, NUM_REPEATS + 1):
                    print(
                        f"[{name}] prompt~{prompt_tokens}tok "
                        f"concurrency={concurrency} repeat={repeat_index}/{NUM_REPEATS} ..."
                    )
                    results, stats_samples = await run_batch(
                        url, prompt, FIXED_OUTPUT_TOKENS, concurrency
                    )

                    for r in results:
                        raw_f.write(
                            json.dumps(
                                {
                                    "deployment": name,
                                    "prompt_tokens_target": prompt_tokens,
                                    "concurrency": concurrency,
                                    "repeat_index": repeat_index,
                                    **asdict(r),
                                    "network_delay": r.network_delay,
                                    "total_delay": r.total_delay,
                                }
                            )
                            + "\n"
                        )
                    raw_f.flush()

                    for s in stats_samples:
                        stats_f.write(
                            json.dumps(
                                {
                                    "deployment": name,
                                    "prompt_tokens_target": prompt_tokens,
                                    "concurrency": concurrency,
                                    "repeat_index": repeat_index,
                                    **s,
                                }
                            )
                            + "\n"
                        )
                    stats_f.flush()

                    summary = summarize(
                        results, stats_samples, name, prompt_tokens, concurrency, repeat_index
                    )
                    all_summaries.append(summary)

                    n_fail = summary.n_requests - summary.n_success
                    if summary.p50_latency is not None:
                        print(
                            f"    p50={summary.p50_latency:.3f}s p95={summary.p95_latency:.3f}s "
                            f"ttft_mean={summary.mean_ttft} client_delay_mean={summary.mean_client_delay} "
                            f"network_delay_mean={summary.mean_network_delay} total_delay_mean={summary.mean_total_delay} "
                            f"tok/s_mean={summary.mean_tok_s} "
                            f"cpu%={summary.mean_cpu_percent} "
                            f"gpu%={summary.mean_gpu_percent} vram%={summary.mean_vram_percent} "
                            f"failures={n_fail} stats_samples={summary.n_stats_samples}"
                        )
                    else:
                        print(f"    ALL {summary.n_requests} REQUESTS FAILED")

                    await asyncio.sleep(1.0)

    import csv

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_summaries[0]).keys()))
        writer.writeheader()
        for s in all_summaries:
            writer.writerow(asdict(s))

    print(f"\nDone.\nRaw requests: {raw_path}\nRaw stats: {stats_raw_path}\nSummary CSV: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark an LLM /generate endpoint")
    parser.add_argument("--url", required=True, help="Base URL, e.g. http://localhost:8000")
    parser.add_argument("--name", required=True, help="Deployment label, e.g. local_cpu")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    asyncio.run(main(args.url, args.name, Path(args.out)))