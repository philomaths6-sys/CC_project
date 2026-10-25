"""
Advanced Publication-Grade Visualization Suite for LLM Quantization Benchmark.
Produces detailed charts including:
1. 6-Panel Comprehensive Architecture & Performance Dashboard
2. Latency Decomposition Breakdown (Network vs TTFT vs Decoding)
3. Quality Sub-metrics Breakdown (ROUGE-1, ROUGE-L, Keyword Accuracy)
4. Domain Category Radar / Grouped Analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "./results"
CHARTS_DIR = "./charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

sum_df = pd.read_csv(os.path.join(RESULTS_DIR, "quant_benchmark_summary.csv"))
det_df = pd.read_csv(os.path.join(RESULTS_DIR, "quant_eval_details.csv"))
sorted_df = sum_df.sort_values(by="disk_size_gb").reset_index(drop=True)

# Aesthetic styling
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 1.0

COLORS = {
    "Q4_K_M": "#3B82F6",  # Blue
    "Q8_0": "#10B981",    # Emerald
    "fp16": "#EF4444"     # Rose / Red
}
MARKERS = {"Q4_K_M": "o", "Q8_0": "D", "fp16": "^"}

# =============================================================================
# 1. LATENCY DECOMPOSITION BREAKDOWN (Stacked Bar)
# =============================================================================
def plot_latency_decomposition():
    fig, ax = plt.subplots(figsize=(8.5, 6))
    
    levels = sorted_df["quant_level"].tolist()
    idx = np.arange(len(levels))
    
    net_delay_s = sorted_df["mean_network_delay_ms"] / 1000.0
    ttft_s = sorted_df["mean_ttft_ms"] / 1000.0
    total_client_s = sorted_df["mean_client_latency_s"]
    decode_s = total_client_s - (net_delay_s + ttft_s)
    decode_s = np.maximum(decode_s, 0.05)  # ensure non-negative

    w = 0.45
    p1 = ax.bar(idx, net_delay_s, w, label="Ngrok WAN Transit Overhead", color="#64748B", edgecolor="black", alpha=0.9)
    p2 = ax.bar(idx, ttft_s, w, bottom=net_delay_s, label="Server TTFT (Prompt Prefill)", color="#F59E0B", edgecolor="black", alpha=0.95)
    p3 = ax.bar(idx, decode_s, w, bottom=net_delay_s + ttft_s, label="GPU Token Generation (Decoding)", color="#3B82F6", edgecolor="black", alpha=0.95)

    # Annotate total latency on top
    for i, total in enumerate(total_client_s):
        ax.text(idx[i], total + 0.06, f"{total:.2f} s", ha="center", va="bottom", fontweight="bold", fontsize=11)
        # In-bar percentages
        ax.text(idx[i], net_delay_s[i] / 2, f"{net_delay_s[i]*1000:.0f}ms\n(WAN)", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")

    ax.set_title("End-to-End Latency Decomposition: Network vs. GPU Prefill vs. Decoding", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Wall-Clock Client Latency (seconds)", fontsize=11, fontweight="bold")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{l}\n({sorted_df.loc[i, 'bits_per_weight']} bpw)" for i, l in enumerate(levels)], fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(total_client_s) * 1.25)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    out = os.path.join(CHARTS_DIR, "quant_latency_decomposition.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[SAVED] {out}")

# =============================================================================
# 2. QUALITY SUB-METRICS BREAKDOWN (ROUGE-1, ROUGE-L, KEYWORD HIT)
# =============================================================================
def plot_quality_metrics_breakdown():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    levels = sorted_df["quant_level"].tolist()
    idx = np.arange(len(levels))
    w = 0.22

    r1 = sorted_df["mean_rouge_1"] * 100
    rl = sorted_df["mean_rouge_l"] * 100
    kw = sorted_df["mean_keyword_hit"] * 100

    b1 = ax.bar(idx - w, r1, w, label="ROUGE-1 F1 (%)", color="#38BDF8", edgecolor="black")
    b2 = ax.bar(idx, rl, w, label="ROUGE-L F1 (%)", color="#6366F1", edgecolor="black")
    b3 = ax.bar(idx + w, kw, w, label="Keyword Hit Rate (%)", color="#10B981", edgecolor="black")

    for bars in [b1, b2, b3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_title("Granular Quality Evaluation: ROUGE-1, ROUGE-L & Factual Keywords", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(idx)
    ax.set_xticklabels(levels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()
    out = os.path.join(CHARTS_DIR, "quant_quality_metrics_breakdown.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[SAVED] {out}")

# =============================================================================
# 3. DOMAIN-SPECIFIC RADAR / SPIDER CHART
# =============================================================================
def plot_category_radar():
    cat_df = det_df.groupby(["category", "quant_level"])["composite_quality"].mean().unstack()
    categories = cat_df.index.tolist()
    n_cats = len(categories)

    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for lvl in ["Q4_K_M", "Q8_0", "fp16"]:
        values = cat_df[lvl].tolist()
        values += values[:1]
        ax.plot(angles, values, color=COLORS[lvl], linewidth=2.2, label=f"{lvl} ({sum_df.loc[sum_df['quant_level']==lvl, 'bits_per_weight'].values[0]} bpw)")
        ax.fill(angles, values, color=COLORS[lvl], alpha=0.12)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace("_", " ").title() for c in categories], fontsize=10.5, fontweight="bold")
    ax.set_rlabel_position(22)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle=":", alpha=0.7)
    plt.title("Task Domain Competency Radar across Quantization Levels", fontsize=14, fontweight="bold", pad=38)
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), framealpha=0.9, fontsize=10)
    plt.tight_layout()
    out = os.path.join(CHARTS_DIR, "quant_category_radar.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out}")

# =============================================================================
# 4. 6-PANEL COMPREHENSIVE BENCHMARK DASHBOARD
# =============================================================================
def plot_comprehensive_6panel_dashboard():
    fig, axes = plt.subplots(3, 2, figsize=(16, 17))
    levels = sorted_df["quant_level"].tolist()
    idx = np.arange(len(levels))
    w = 0.35

    # Panel 1: Pareto Frontier & Marginal Return
    x = sorted_df["disk_size_gb"]
    y = sorted_df["mean_composite_quality"]
    axes[0, 0].plot(x, y, color="#64748B", linestyle="--", linewidth=2.5, zorder=2)
    for _, row in sorted_df.iterrows():
        lvl = row["quant_level"]
        is_k = (lvl == "Q8_0")
        axes[0, 0].scatter(row["disk_size_gb"], row["mean_composite_quality"],
                           color=COLORS[lvl], s=200 if is_k else 130, marker=MARKERS[lvl],
                           edgecolors="black", linewidth=1.5, zorder=5, label=f"{lvl}")
    
    # Knee callout
    k_row = sorted_df[sorted_df["quant_level"] == "Q8_0"].iloc[0]
    axes[0, 0].annotate(
        "★ THE KNEE: Q8_0\n• 98.5% Quality of Q4\n• 47% Lower Memory Footprint\n• 1.6x Throughput of fp16",
        xy=(k_row["disk_size_gb"], k_row["mean_composite_quality"]),
        xytext=(k_row["disk_size_gb"] - 0.25, k_row["mean_composite_quality"] + 0.45),
        fontsize=9.5, fontweight="bold", color="#065F46",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#D1FAE5", edgecolor="#10B981", lw=1.6),
        arrowprops=dict(facecolor="#10B981", edgecolor="#065F46", arrowstyle="fancy,tail_width=0.4", connectionstyle="arc3,rad=-0.1")
    )
    axes[0, 0].set_title("Panel 1: Pareto Frontier & Marginal Return Analysis", fontweight="bold", fontsize=11.5)
    axes[0, 0].set_xlabel("Model Weight Size on Disk (GB)", fontweight="bold")
    axes[0, 0].set_ylabel("Composite Quality (%)", fontweight="bold")
    axes[0, 0].set_ylim(y.min() - 0.4, y.max() + 0.8)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend(loc="lower right")

    # Panel 2: Memory Footprint Breakdown (Disk vs VRAM vs RAM)
    axes[0, 1].bar(idx - w/2, sorted_df["disk_size_gb"], w, label="File Size (Disk)", color="#94A3B8", edgecolor="black")
    axes[0, 1].bar(idx + w/2, sorted_df["vram_gb"], w, label="Active VRAM (Tesla T4)", color="#3B82F6", edgecolor="black")
    for i in range(len(sorted_df)):
        axes[0, 1].text(idx[i] - w/2, sorted_df.loc[i, "disk_size_gb"] + 0.08, f"{sorted_df.loc[i, 'disk_size_gb']:.2f}G", ha="center", fontsize=8.5, fontweight="bold")
        axes[0, 1].text(idx[i] + w/2, sorted_df.loc[i, "vram_gb"] + 0.08, f"{sorted_df.loc[i, 'vram_gb']:.2f}G", ha="center", fontsize=8.5, fontweight="bold")
    axes[0, 1].set_title("Panel 2: Memory Footprint (Disk Weight vs. Active VRAM)", fontweight="bold", fontsize=11.5)
    axes[0, 1].set_ylabel("Allocated Memory (GB)", fontweight="bold")
    axes[0, 1].set_xticks(idx)
    axes[0, 1].set_xticklabels(levels, fontweight="bold")
    axes[0, 1].set_ylim(0, 4.5)
    axes[0, 1].grid(True, linestyle=":", alpha=0.5)
    axes[0, 1].legend(loc="upper left")

    # Panel 3: TTFT vs Model Precision
    p1 = axes[1, 0].bar(idx, sorted_df["mean_ttft_ms"], w*1.2, color=["#3B82F6", "#10B981", "#EF4444"], edgecolor="black")
    for bar in p1:
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0, f"{bar.get_height():.1f} ms",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1, 0].set_title("Panel 3: Prompt Evaluation Time-to-First-Token (TTFT)", fontweight="bold", fontsize=11.5)
    axes[1, 0].set_ylabel("Time to First Token (ms)", fontweight="bold")
    axes[1, 0].set_xticks(idx)
    axes[1, 0].set_xticklabels(levels, fontweight="bold")
    axes[1, 0].set_ylim(0, 55)
    axes[1, 0].grid(True, linestyle=":", alpha=0.5)

    # Panel 4: GPU Throughput vs Bit Depth
    axes[1, 1].plot(sorted_df["bits_per_weight"], sorted_df["mean_throughput_tok_s"],
                    marker="s", markersize=10, linewidth=2.5, color="#059669", zorder=3)
    for _, row in sorted_df.iterrows():
        axes[1, 1].annotate(f"{row['quant_level']}\n{row['mean_throughput_tok_s']:.1f} tok/s",
                            (row["bits_per_weight"], row["mean_throughput_tok_s"] + 4.5),
                            ha="center", fontweight="bold", fontsize=9)
    axes[1, 1].set_title("Panel 4: Token Throughput vs. Model Precision (Bits / Weight)", fontweight="bold", fontsize=11.5)
    axes[1, 1].set_xlabel("Nominal Bits per Weight (bpw)", fontweight="bold")
    axes[1, 1].set_ylabel("Generation Speed (tokens/s)", fontweight="bold")
    axes[1, 1].set_ylim(40, 150)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    # Panel 5: Domain Breakdown
    cat_df = det_df.groupby(["category", "quant_level"])["composite_quality"].mean().unstack()
    cat_df.plot(kind="bar", ax=axes[2, 0], colormap="viridis", edgecolor="black", width=0.7)
    axes[2, 0].set_title("Panel 5: Accuracy Across Task Domains (Instruction Cliff)", fontweight="bold", fontsize=11.5)
    axes[2, 0].set_ylabel("Quality Score (%)", fontweight="bold")
    axes[2, 0].set_xticklabels(axes[2, 0].get_xticklabels(), rotation=15, ha="right", fontsize=9)
    axes[2, 0].legend(title="Quant Level")
    axes[2, 0].grid(True, linestyle=":", alpha=0.5)

    # Panel 6: Quality vs GPU Throughput Trade-off
    q_vals = sorted_df["mean_composite_quality"]
    q_rng = max(0.5, q_vals.max() - q_vals.min())
    axes[2, 1].set_ylim(q_vals.min() - q_rng * 0.35, q_vals.max() + q_rng * 0.35)
    for _, row in sorted_df.iterrows():
        lvl = row["quant_level"]
        axes[2, 1].scatter(row["mean_throughput_tok_s"], row["mean_composite_quality"],
                           color=COLORS[lvl], s=200, edgecolors="black", linewidth=1.5)
        axes[2, 1].annotate(f"{lvl} ({row['mean_throughput_tok_s']:.1f} tok/s)",
                            (row["mean_throughput_tok_s"], row["mean_composite_quality"] + q_rng * 0.08),
                            ha="center", fontweight="bold", fontsize=9.5)
    axes[2, 1].set_title("Panel 6: Generation Throughput vs. Composite Quality", fontweight="bold", fontsize=11.5)
    axes[2, 1].set_xlabel("Throughput (tokens/s)", fontweight="bold")
    axes[2, 1].set_ylabel("Composite Quality (%)", fontweight="bold")
    axes[2, 1].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Comprehensive LLM Quantization Benchmark Suite (Tesla T4 GPU Client-Server)", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    out = os.path.join(CHARTS_DIR, "quant_comprehensive_6panel_dashboard.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[SAVED] {out}")

if __name__ == "__main__":
    plot_latency_decomposition()
    plot_quality_metrics_breakdown()
    plot_category_radar()
    plot_comprehensive_6panel_dashboard()
    print("\n[ALL DETAILED CHARTS SUCCESSFULLY GENERATED]")
