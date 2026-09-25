"""
Plot local CPU vs Colab GPU LLM inference benchmarks.

Matches the schema produced by the confirmed bench.py (the version with
client_delay / network_delay / total_delay added to RunSummary). Once BOTH
deployments have been re-run with that same bench.py, their summary CSVs
share identical columns, so this script reads only the two summary CSVs --
no raw JSONL needed.

Reads:
  - local_cpu_summary.csv
  - google_collab_gpu_summary.csv

Produces (in ./charts/):
  - bar_latency_mean.png        mean client_delay, CPU vs GPU
  - bar_latency_p50.png         p50 (median) client_delay, CPU vs GPU
  - bar_latency_p95.png         p95 (tail) client_delay, CPU vs GPU
  - bar_network_delay_mean.png  mean network_delay, CPU vs GPU
  - bar_network_delay_p50.png   p50 network_delay, CPU vs GPU
  - bar_network_delay_p95.png   p95 network_delay, CPU vs GPU
  - bar_ttft_mean.png           mean server_ttft, CPU vs GPU
  - bar_ttft_p50.png            p50 server_ttft, CPU vs GPU
  - bar_ttft_p95.png            p95 server_ttft, CPU vs GPU
  - bar_tok_s.png                mean_tok_s, CPU vs GPU
  - line_latency_128.png        per-repeat mean client_delay, 128-token prompt
  - line_latency_1024.png       per-repeat mean client_delay, 1024-token prompt
  - line_network_delay.png      per-repeat mean network_delay, both prompt lengths

Every chart has a short caption printed underneath it explaining exactly
what the plotted number means and how it's computed, so the PNG is
self-explanatory without needing to go back to this script or bench.py.

Edit the file paths in main() if needed, then run:
  python plot_benchmarks.py
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------- metric explanations (shown as captions under each chart) ----------

DESCRIPTIONS = {
    "latency": (
        "client_delay (a.k.a. client_latency): full round-trip time measured by\n"
        "the client, from request sent to full response received. This is what a\n"
        "real caller of the API actually experiences -- network + server, combined."
    ),
    "network_delay": (
        "network_delay = client_latency - server_total_latency: the portion of the\n"
        "round trip NOT accounted for by the server's own reported processing time.\n"
        "Covers network transit, tunnel/proxy overhead (e.g. ngrok), and serialization."
    ),
    "ttft": (
        "server_ttft: server-side time-to-first-token, measured on the server's own\n"
        "clock from request received to first generated token. Excludes network time\n"
        "entirely -- this is pure model/prefill latency."
    ),
    "tok_s": (
        "mean_tok_s: tokens generated per second, as reported by bench.py\n"
        "(tokens_generated / (client_latency - server_ttft)). Includes any network\n"
        "delay folded into the denominator, so treat this as client-observed\n"
        "throughput, not pure server compute speed."
    ),
    "percentile_note": (
        "p50 = median across all requests in that run (typical case).\n"
        "p95 = 95th percentile (a slow, tail-end case) -- useful for spotting\n"
        "occasional slow requests that the mean alone would hide."
    ),
}


# ---------- chart builders ----------

def plot_grouped_bar(labels, series_dict, title, ylabel, out_path, caption=None):
    """
    labels: x-axis category labels, e.g. ["128 tokens", "1024 tokens"]
    series_dict: {"Series name": [val_for_label0, val_for_label1, ...]}
    caption: optional text drawn below the chart explaining the metric.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))
    n_series = len(series_dict)
    n_labels = len(labels)
    bar_width = 0.8 / n_series
    x = range(n_labels)

    for i, (name, values) in enumerate(series_dict.items()):
        offsets = [xi + i * bar_width - (0.8 - bar_width) / 2 for xi in x]
        ax.bar(offsets, values, width=bar_width, label=name)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if caption:
        fig.subplots_adjust(bottom=0.30)
        fig.text(0.02, 0.02, caption, ha="left", va="bottom", fontsize=8, wrap=True)
    else:
        fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_lines(x_labels, series_dict, title, ylabel, out_path, log_scale=False, caption=None):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, values in series_dict.items():
        ax.plot(x_labels, values, marker="o", label=name)

    if log_scale:
        ax.set_yscale("log")

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    if caption:
        fig.subplots_adjust(bottom=0.30)
        fig.text(0.02, 0.02, caption, ha="left", va="bottom", fontsize=8, wrap=True)
    else:
        fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


# ---------- main ----------

def main():
    # ---- EDIT THESE PATHS ----
    cpu_summary_path = "./results/local_cpu_summary.csv"
    gpu_summary_path = "./results/google_collab_gpu_summary.csv"
    out_dir = Path("charts")
    # ---------------------------

    out_dir.mkdir(exist_ok=True)

    cpu = pd.read_csv(cpu_summary_path)
    gpu = pd.read_csv(gpu_summary_path)

    required_cols = {
        "mean_latency", "p50_latency", "p95_latency",
        "mean_network_delay", "p50_network_delay", "p95_network_delay",
        "mean_ttft", "p50_ttft", "p95_ttft",
        "mean_tok_s",
    }
    for name, df in [("CPU", cpu), ("GPU", gpu)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise SystemExit(
                f"{name} summary CSV is missing columns {missing} -- "
                "make sure it was generated with the confirmed bench.py "
                "(the version with client_delay/network_delay/total_delay)."
            )

    prompt_lengths = sorted(set(cpu["prompt_tokens_target"]) & set(gpu["prompt_tokens_target"]))
    labels = [f"{p} tokens" for p in prompt_lengths]

    def mean_by_prompt(df, col):
        return [df[df["prompt_tokens_target"] == p][col].mean() for p in prompt_lengths]

    # ---- Latency: mean, p50, p95 ----
    metric_specs = [
        ("mean_latency", "Mean End-to-End Latency (client_delay)", "bar_latency_mean.png", "latency"),
        ("p50_latency", "Median (p50) End-to-End Latency (client_delay)", "bar_latency_p50.png", "latency+pctl"),
        ("p95_latency", "p95 End-to-End Latency (client_delay)", "bar_latency_p95.png", "latency+pctl"),
        ("mean_network_delay", "Mean Network Delay", "bar_network_delay_mean.png", "network_delay"),
        ("p50_network_delay", "Median (p50) Network Delay", "bar_network_delay_p50.png", "network_delay+pctl"),
        ("p95_network_delay", "p95 Network Delay", "bar_network_delay_p95.png", "network_delay+pctl"),
        ("mean_ttft", "Mean Time-to-First-Token (server_ttft)", "bar_ttft_mean.png", "ttft"),
        ("p50_ttft", "Median (p50) Time-to-First-Token", "bar_ttft_p50.png", "ttft+pctl"),
        ("p95_ttft", "p95 Time-to-First-Token", "bar_ttft_p95.png", "ttft+pctl"),
    ]

    for col, title, filename, desc_key in metric_specs:
        base_key = desc_key.split("+")[0]
        caption = DESCRIPTIONS[base_key]
        if desc_key.endswith("+pctl"):
            caption = caption + "\n\n" + DESCRIPTIONS["percentile_note"]

        plot_grouped_bar(
            labels,
            {
                "Local CPU": mean_by_prompt(cpu, col),
                "Colab GPU": mean_by_prompt(gpu, col),
            },
            title,
            "Seconds",
            out_dir / filename,
            caption=caption,
        )

    # ---- tok/s (mean only -- no p50/p95 tok/s column in bench.py) ----
    plot_grouped_bar(
        labels,
        {
            "Local CPU": mean_by_prompt(cpu, "mean_tok_s"),
            "Colab GPU": mean_by_prompt(gpu, "mean_tok_s"),
        },
        "Mean Tokens/sec (mean_tok_s)",
        "Tokens/sec",
        out_dir / "bar_tok_s.png",
        caption=DESCRIPTIONS["tok_s"],
    )

    # ---- Line charts: per-repeat trend, CPU vs GPU, one chart per prompt length ----
    for p in prompt_lengths:
        cpu_sub = cpu[cpu["prompt_tokens_target"] == p].sort_values("repeat_index")
        gpu_sub = gpu[gpu["prompt_tokens_target"] == p].sort_values("repeat_index")
        repeat_labels = [f"Repeat {i}" for i in cpu_sub["repeat_index"]]

        plot_lines(
            repeat_labels,
            {
                "Local CPU": cpu_sub["mean_latency"].tolist(),
                "Colab GPU": gpu_sub["mean_latency"].tolist(),
            },
            f"Latency Across Repeats — {p}-token prompt",
            "Mean latency (s)",
            out_dir / f"line_latency_{p}.png",
            caption=DESCRIPTIONS["latency"],
        )

    # ---- Line chart: network delay across repeats, both prompt lengths, both deployments ----
    series = {}
    x_labels = None
    for p in prompt_lengths:
        cpu_sub = cpu[cpu["prompt_tokens_target"] == p].sort_values("repeat_index")
        gpu_sub = gpu[gpu["prompt_tokens_target"] == p].sort_values("repeat_index")
        if x_labels is None:
            x_labels = [f"Repeat {i}" for i in cpu_sub["repeat_index"]]
        series[f"Local CPU ({p} tok)"] = cpu_sub["mean_network_delay"].tolist()
        series[f"Colab GPU ({p} tok)"] = gpu_sub["mean_network_delay"].tolist()

    plot_lines(
        x_labels,
        series,
        "Network Delay Across Repeats",
        "Seconds",
        out_dir / "line_network_delay.png",
        log_scale=True,
        caption=DESCRIPTIONS["network_delay"],
    )

    print("\nAll charts written to:", out_dir.resolve())


if __name__ == "__main__":
    main()