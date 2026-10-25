"""
Deep-Dive Publication-Grade Visualization Suite for LLM Quantization Benchmark.
Generates highly detailed empirical figures directly from actual benchmark records:
1. quant_per_task_breakdown.png: 10-Task Granular Evaluation Breakdown with category groupings & instruction cliff callout.
2. quant_task_performance_heatmap.png: Side-by-side Heatmap Matrix (Task Quality vs GPU Throughput) with exact numeric values.
3. quant_distribution_variance_analysis.png: 4-Panel Empirical Variance & Distribution Analysis (Throughput, Latency, TTFT, Quality).
4. quant_pareto_knee_frontier.png: High-density Pareto Frontier with Trade-off Efficiency Vectors and Diminishing Marginal Returns Table.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

RESULTS_DIR = "./results"
CHARTS_DIR = "./charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

sum_df = pd.read_csv(os.path.join(RESULTS_DIR, "quant_benchmark_summary.csv"))
det_df = pd.read_csv(os.path.join(RESULTS_DIR, "quant_eval_details.csv"))
sorted_df = sum_df.sort_values(by="disk_size_gb").reset_index(drop=True)

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 1.0

COLORS = {
    "Q4_K_M": "#2563EB",  # Deep Blue
    "Q8_0": "#059669",    # Emerald Green
    "fp16": "#DC2626"     # Crimson Red
}
LIGHT_COLORS = {
    "Q4_K_M": "#DBEAFE",
    "Q8_0": "#D1FAE5",
    "fp16": "#FEE2E2"
}

# =============================================================================
# 1. PER-TASK GRANULAR EVALUATION BREAKDOWN (ALL 10 TASKS)
# =============================================================================
def plot_per_task_breakdown():
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Task mapping with descriptive human-readable names
    task_labels = {
        "qa_01": "QA-1: Capital of Aus (Factual)",
        "qa_02": "QA-2: Lorentz Factor (Physics)",
        "math_01": "Math-1: Capacity Plan (Reasoning)",
        "logic_01": "Logic-1: River Crossing (Deduction)",
        "code_01": "Code-1: LRU Cache (Python)",
        "code_02": "Code-2: Binary Search (Bugfix)",
        "summ_01": "Summ-1: MapReduce (Summary)",
        "summ_02": "Summ-2: Cloud Native (Summary)",
        "extract_01": "Extract-1: JSON Entity (Parsing)",
        "instruct_01": "Instruct-1: Strict Schema (Constraint)"
    }
    
    tasks = list(task_labels.keys())
    x = np.arange(len(tasks))
    w = 0.26
    
    piv = det_df.pivot(index="task_id", columns="quant_level", values="composite_quality").reindex(tasks)
    
    b1 = ax.bar(x - w, piv["Q4_K_M"], w, label="Q4_K_M (4.5 bpw, 1.04 GB)", color=COLORS["Q4_K_M"], edgecolor="black", alpha=0.92)
    b2 = ax.bar(x, piv["Q8_0"], w, label="Q8_0 (8.5 bpw, 1.76 GB)", color=COLORS["Q8_0"], edgecolor="black", alpha=0.95)
    b3 = ax.bar(x + w, piv["fp16"], w, label="fp16 (16.0 bpw, 3.32 GB)", color=COLORS["fp16"], edgecolor="black", alpha=0.92)
    
    # Value annotations on top of each bar
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold", rotation=0)
            
    # Highlight the Instruction Following Cliff on instruct_01
    inst_idx = tasks.index("instruct_01")
    ax.annotate(
        "★ INSTRUCTION CLIFF\nQ4 drops to 11.3%\nQ8 matches FP16 at 21.5%\n(+90.5% recovery)",
        xy=(inst_idx, piv.loc["instruct_01", "Q8_0"]),
        xytext=(inst_idx - 2.5, 78),
        fontsize=9.5, fontweight="bold", color="#065F46",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ECFDF5", edgecolor="#10B981", lw=1.6),
        arrowprops=dict(facecolor="#10B981", edgecolor="#065F46", arrowstyle="->", lw=1.8, connectionstyle="arc3,rad=-0.15")
    )

    # Highlight QA parity on qa_01 & extract_01
    ax.annotate(
        "Factual & Extraction Parity:\nHigh score across all tiers",
        xy=(0, 97.2),
        xytext=(0.8, 108),
        fontsize=8.5, fontweight="bold", color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F1F5F9", edgecolor="#64748B", lw=1.2),
        arrowprops=dict(facecolor="#64748B", arrowstyle="->", lw=1.4)
    )

    ax.set_title("Granular Task-by-Task Performance across All 10 Benchmark Prompts", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Composite Quality Score (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([task_labels[t] for t in tasks], rotation=25, ha="right", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 125)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10.5)

    plt.tight_layout()
    out = os.path.join(CHARTS_DIR, "quant_per_task_breakdown.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[SAVED] {out}")


# =============================================================================
# 2. DUAL TASK PERFORMANCE HEATMAP (QUALITY & THROUGHPUT)
# =============================================================================
def plot_task_performance_heatmap():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={'width_ratios': [1, 1]})
    
    task_labels = {
        "qa_01": "QA-1: Capital of Aus (Factual)",
        "qa_02": "QA-2: Lorentz Factor (Physics)",
        "math_01": "Math-1: Capacity Plan (Reasoning)",
        "logic_01": "Logic-1: River Crossing (Deduction)",
        "code_01": "Code-1: LRU Cache (Python)",
        "code_02": "Code-2: Binary Search (Bugfix)",
        "summ_01": "Summ-1: MapReduce (Summary)",
        "summ_02": "Summ-2: Cloud Native (Summary)",
        "extract_01": "Extract-1: JSON Entity (Parsing)",
        "instruct_01": "Instruct-1: Strict Schema (Constraint)"
    }
    tasks = list(task_labels.keys())
    models = ["Q4_K_M", "Q8_0", "fp16"]
    
    piv_q = det_df.pivot(index="task_id", columns="quant_level", values="composite_quality").reindex(tasks)[models]
    piv_t = det_df.pivot(index="task_id", columns="quant_level", values="throughput_tok_s").reindex(tasks)[models]
    
    y_labels = [task_labels[t] for t in tasks]
    col_labels = ["Q4_K_M\n(4.5 bpw)", "Q8_0\n(8.5 bpw)", "fp16\n(16.0 bpw)"]
    
    # Heatmap 1: Quality
    im1 = ax1.imshow(piv_q.values, cmap="YlGnBu", aspect="auto", vmin=10, vmax=100)
    for i in range(len(tasks)):
        for j in range(len(models)):
            val = piv_q.values[i, j]
            text_color = "white" if val > 65 else "black"
            ax1.text(j, i, f"{val:.1f}%", ha="center", va="center", color=text_color, fontweight="bold", fontsize=10.5)
            
    ax1.set_title("(A) Output Quality Score (%)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(col_labels, fontsize=10.5, fontweight="bold")
    ax1.set_yticks(range(len(tasks)))
    ax1.set_yticklabels(y_labels, fontsize=10, fontweight="bold")
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=9)
    
    # Heatmap 2: Throughput
    im2 = ax2.imshow(piv_t.values, cmap="magma", aspect="auto", vmin=50, vmax=140)
    for i in range(len(tasks)):
        for j in range(len(models)):
            val = piv_t.values[i, j]
            text_color = "white" if val < 90 or val > 125 else "black"
            ax2.text(j, i, f"{val:.1f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=10.5)
            
    ax2.set_title("(B) Generation Speed (tokens / sec)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(col_labels, fontsize=10.5, fontweight="bold")
    ax2.set_yticks(range(len(tasks)))
    ax2.set_yticklabels([])  # Hide shared y labels
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=9)
    
    fig.suptitle("Granular Performance Matrix: Quality vs. Generation Throughput by Task", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(CHARTS_DIR, "quant_task_performance_heatmap.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[SAVED] {out}")


# =============================================================================
# 3. EMPIRICAL DISTRIBUTION & VARIANCE ANALYSIS (4-PANEL BOX PLOTS)
# =============================================================================
def plot_distribution_variance_analysis():
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    models = ["Q4_K_M", "Q8_0", "fp16"]
    
    # Panel 1: Throughput Distribution
    t_data = [det_df[det_df["quant_level"] == m]["throughput_tok_s"].values for m in models]
    bp1 = axes[0, 0].boxplot(t_data, patch_artist=True, widths=0.45,
                             boxprops=dict(facecolor="#E0E7FF", color="#3730A3", lw=1.5),
                             medianprops=dict(color="#1E1B4B", lw=2),
                             whiskerprops=dict(color="#3730A3", lw=1.2),
                             capprops=dict(color="#3730A3", lw=1.2))
    for i, m in enumerate(models):
        y_pts = t_data[i]
        x_pts = np.random.normal(i + 1, 0.04, size=len(y_pts))
        axes[0, 0].scatter(x_pts, y_pts, color=COLORS[m], alpha=0.85, s=45, edgecolors="black", zorder=4)
        mean_v = np.mean(y_pts)
        axes[0, 0].text(i + 1, mean_v + 3.5, f"μ={mean_v:.1f}", ha="center", fontweight="bold", fontsize=9.5, color=COLORS[m])
    axes[0, 0].set_title("(A) GPU Token Throughput Distribution (tok/s)", fontweight="bold", fontsize=12)
    axes[0, 0].set_ylabel("Throughput (tokens/s)", fontweight="bold")
    axes[0, 0].set_xticklabels(models, fontweight="bold", fontsize=10.5)
    axes[0, 0].set_ylim(40, 150)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel 2: End-to-End Latency Distribution
    l_data = [det_df[det_df["quant_level"] == m]["client_latency_s"].values for m in models]
    bp2 = axes[0, 1].boxplot(l_data, patch_artist=True, widths=0.45,
                             boxprops=dict(facecolor="#FEF3C7", color="#B45309", lw=1.5),
                             medianprops=dict(color="#78350F", lw=2),
                             whiskerprops=dict(color="#B45309", lw=1.2),
                             capprops=dict(color="#B45309", lw=1.2))
    for i, m in enumerate(models):
        y_pts = l_data[i]
        x_pts = np.random.normal(i + 1, 0.04, size=len(y_pts))
        axes[0, 1].scatter(x_pts, y_pts, color=COLORS[m], alpha=0.85, s=45, edgecolors="black", zorder=4)
        mean_v = np.mean(y_pts)
        axes[0, 1].text(i + 1, mean_v + 0.08, f"μ={mean_v:.2f}s", ha="center", fontweight="bold", fontsize=9.5, color=COLORS[m])
    axes[0, 1].set_title("(B) End-to-End Client Latency Distribution (s)", fontweight="bold", fontsize=12)
    axes[0, 1].set_ylabel("Client Latency (seconds)", fontweight="bold")
    axes[0, 1].set_xticklabels(models, fontweight="bold", fontsize=10.5)
    axes[0, 1].set_ylim(0.5, 3.2)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # Panel 3: TTFT Distribution
    tt_data = [det_df[det_df["quant_level"] == m]["server_ttft_ms"].values for m in models]
    bp3 = axes[1, 0].boxplot(tt_data, patch_artist=True, widths=0.45,
                             boxprops=dict(facecolor="#DCFCE7", color="#15803D", lw=1.5),
                             medianprops=dict(color="#14532D", lw=2),
                             whiskerprops=dict(color="#15803D", lw=1.2),
                             capprops=dict(color="#15803D", lw=1.2))
    for i, m in enumerate(models):
        y_pts = tt_data[i]
        x_pts = np.random.normal(i + 1, 0.04, size=len(y_pts))
        axes[1, 0].scatter(x_pts, y_pts, color=COLORS[m], alpha=0.85, s=45, edgecolors="black", zorder=4)
        mean_v = np.mean(y_pts)
        axes[1, 0].text(i + 1, mean_v + 8.0, f"μ={mean_v:.1f}ms", ha="center", fontweight="bold", fontsize=9.5, color=COLORS[m])
    axes[1, 0].set_title("(C) Server Time-to-First-Token (TTFT) Distribution (ms)", fontweight="bold", fontsize=12)
    axes[1, 0].set_ylabel("Server TTFT (ms)", fontweight="bold")
    axes[1, 0].set_xticklabels(models, fontweight="bold", fontsize=10.5)
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel 4: Quality Distribution
    q_data = [det_df[det_df["quant_level"] == m]["composite_quality"].values for m in models]
    bp4 = axes[1, 1].boxplot(q_data, patch_artist=True, widths=0.45,
                             boxprops=dict(facecolor="#FCE7F3", color="#BE185D", lw=1.5),
                             medianprops=dict(color="#831843", lw=2),
                             whiskerprops=dict(color="#BE185D", lw=1.2),
                             capprops=dict(color="#BE185D", lw=1.2))
    for i, m in enumerate(models):
        y_pts = q_data[i]
        x_pts = np.random.normal(i + 1, 0.04, size=len(y_pts))
        axes[1, 1].scatter(x_pts, y_pts, color=COLORS[m], alpha=0.85, s=45, edgecolors="black", zorder=4)
        mean_v = np.mean(y_pts)
        axes[1, 1].text(i + 1, mean_v + 2.5, f"μ={mean_v:.1f}%", ha="center", fontweight="bold", fontsize=9.5, color=COLORS[m])
    axes[1, 1].set_title("(D) Task Composite Quality Distribution (%)", fontweight="bold", fontsize=12)
    axes[1, 1].set_ylabel("Quality Score (%)", fontweight="bold")
    axes[1, 1].set_xticklabels(models, fontweight="bold", fontsize=10.5)
    axes[1, 1].set_ylim(0, 110)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Empirical Distribution & Variance Across 30 Benchmark Invocations", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(CHARTS_DIR, "quant_distribution_variance_analysis.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[SAVED] {out}")


# =============================================================================
# 4. HIGH-DENSITY PARETO KNEE & MARGINAL RETURN FRONTIER
# =============================================================================
def plot_pareto_knee_frontier():
    fig, ax = plt.subplots(figsize=(12, 7.8))
    
    # Coordinates: X = Active VRAM (GB), Y = Throughput (tok/s)
    x = sorted_df["vram_gb"].values
    y = sorted_df["mean_throughput_tok_s"].values
    quality = sorted_df["mean_composite_quality"].values
    disk = sorted_df["disk_size_gb"].values
    levels = sorted_df["quant_level"].values
    
    # Plot Pareto connection line
    ax.plot(x, y, color="#64748B", linestyle="--", linewidth=2.5, zorder=2, label="Pareto Frontier (Throughput vs. Memory)")
    
    # Bubble plot with size proportional to disk size and color by quality
    for i, lvl in enumerate(levels):
        size = 180 + (disk[i] * 90)
        ax.scatter(x[i], y[i], s=size, color=COLORS[lvl], edgecolors="black", linewidth=2.0, zorder=5, label=f"{lvl} ({disk[i]:.2f} GB disk)")
        
        # Position data badge cleanly
        if lvl == "Q4_K_M":
            bx, by = x[i] + 0.10, y[i] - 12
        elif lvl == "Q8_0":
            bx, by = x[i] + 0.12, y[i] + 8
        else: # fp16
            bx, by = x[i] + 0.12, y[i] - 10
            
        ax.annotate(
            f"{lvl}\nVRAM: {x[i]:.2f} GB\nSpeed: {y[i]:.1f} tok/s\nQuality: {quality[i]:.1f}%",
            xy=(x[i], y[i]),
            xytext=(bx, by),
            fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT_COLORS[lvl], edgecolor=COLORS[lvl], lw=1.4)
        )
        
    # The Knee annotation
    knee_x, knee_y = x[1], y[1]
    ax.annotate(
        "★ THE PARETO KNEE: Q8_0\n"
        "───────────────────────────────\n"
        "• Memory Savings: -1.35 GB VRAM vs FP16 (-35.7%)\n"
        "• Speed Advantage: +36.3 tok/s (+60.6% faster)\n"
        "• Quality Retention: 100.7% vs FP16 (45.37% vs 45.07%)\n"
        "• Prevents Q4's 47.5% drop on strict instruction following",
        xy=(knee_x, knee_y),
        xytext=(knee_x - 0.78, knee_y - 36),
        fontsize=9.5, fontweight="bold", color="#065F46",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#ECFDF5", edgecolor="#059669", lw=2.0),
        arrowprops=dict(facecolor="#059669", edgecolor="#065F46", arrowstyle="fancy,tail_width=0.4", connectionstyle="arc3,rad=-0.12")
    )

    # Inset Marginal Analysis Table
    table_data = [
        ["Transition", "Δ VRAM", "Δ Throughput", "Quality Impact", "Efficiency Verdict"],
        ["Q4_K_M → Q8_0", "+0.63 GB (+35%)", "-31.4 tok/s (-25%)", "+90.5% (Inst)", "★ Knee (Optimal)"],
        ["Q8_0 → fp16", "+1.35 GB (+55%)", "-36.3 tok/s (-38%)", "+0.0% (Flat)", "Diminishing Return"]
    ]
    t = ax.table(cellText=table_data, loc="upper right", cellLoc="center", bbox=[0.33, 0.73, 0.65, 0.23])
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    for (row, col), cell in t.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1E293B")
            cell.set_text_props(color="white", fontweight="bold")
        elif row == 1:
            cell.set_facecolor("#D1FAE5")
            cell.set_text_props(fontweight="bold", color="#065F46")
        else:
            cell.set_facecolor("#FEE2E2")
            cell.set_text_props(fontweight="bold", color="#991B1B")

    ax.set_title("Multi-Objective Pareto Frontier: VRAM Consumption vs. Generation Throughput", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Active GPU VRAM Footprint on Tesla T4 (GB)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Token Generation Throughput (tokens/s)", fontsize=11, fontweight="bold")
    ax.set_xlim(1.4, 4.4)
    ax.set_ylim(35, 150)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=9.5)

    plt.tight_layout()
    out = os.path.join(CHARTS_DIR, "quant_pareto_knee_frontier.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    plot_per_task_breakdown()
    plot_task_performance_heatmap()
    plot_distribution_variance_analysis()
    plot_pareto_knee_frontier()
    print("\n[ALL 4 DEEP-DIVE CHARTS SUCCESSFULLY GENERATED]")
