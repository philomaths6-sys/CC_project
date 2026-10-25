# Cloud Computing Course Project: LLM Inference & Quantization Benchmarks

A comprehensive, empirical Cloud Computing benchmark evaluating:
1. **Infrastructure Paradigms**: Local CPU (Edge/On-Premises) vs. Cloud GPU (Google Colab Tesla T4 via Ngrok HTTPS Tunnel).
2. **Quantization & Efficiency Paradigms**: Multi-level Quantization (`Q4_K_M` @ 4.5 bpw, `Q8_0` @ 8.5 bpw, `fp16` @ 16.0 bpw) on NVIDIA Tesla T4 GPU, isolating memory footprints, throughput, network transit overhead, domain quality, and locating the **Pareto Knee of the Curve**.

All experiments follow a strict **Client–Server decoupled architecture**:
- **Server Role (Google Colab)**: Pure inference host. Runs model execution and returns only the generated completion text and raw server-side timing telemetry (`ttft`, `total_latency`, `tokens_generated`, `prompt_tokens`).
- **Client Role (Local Laptop)**: Evaluation harness. Measures end-to-end wall-clock latency, mathematically isolates WAN network delay ($T_{\text{net}} = T_{\text{client}} - T_{\text{server}}$), calculates GPU token throughput ($\text{tokens} / T_{\text{decode}}$), evaluates text quality (ROUGE-1, ROUGE-L, Factual Keyword Hit Rate), and renders publication-grade charts from authentic empirical data.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Repository File Index](#repository-file-index)
- [Empirical Results & The Pareto Knee](#empirical-results--the-pareto-knee)
  - [Part 1: Local CPU vs. Cloud GPU Baseline](#part-1-local-cpu-vs-cloud-gpu-baseline)
  - [Part 2: Quantization Precision Benchmark (`Q4_K_M` vs `Q8_0` vs `fp16`)](#part-2-quantization-precision-benchmark-q4_k_m-vs-q8_0-vs-fp16)
  - [Locating the Pareto "Knee of the Curve"](#locating-the-pareto-knee-of-the-curve)
- [Visualizations & Performance Charts](#visualizations--performance-charts)
- [Step-by-Step Execution Guide](#step-by-step-execution-guide)
  - [Prerequisites](#prerequisites)
  - [Track A: Running the Quantization Benchmark on Colab GPU](#track-a-running-the-quantization-benchmark-on-colab-gpu)
  - [Track B: Running the CPU vs. GPU Baseline Benchmark](#track-b-running-the-cpu-vs-gpu-baseline-benchmark)
- [API Reference](#api-reference)
- [Hardware & Software Specifications](#hardware--software-specifications)

---

## Problem Statement

When deploying Large Language Models (LLMs) in cloud production environments, architects face a multi-dimensional optimization dilemma:
1. **Memory & VRAM Constraints**: Full-precision weights (`fp16`) require substantial High-Bandwidth Memory (HBM). Smaller cloud GPUs (e.g., NVIDIA T4 with 16 GB VRAM) quickly become saturated when hosting multi-gigabyte models alongside key-value (KV) caches and concurrent context windows.
2. **Serving Latency & Compute Throughput**: Compressing model weights reduces memory bus bandwidth requirements and accelerates matrix-vector multiplications, but sub-8-bit quantization (`Q4_K_M`) introduces dequantization math overhead and potential precision clipping.
3. **Task Quality Degradation & The Instruction Cliff**: Does aggressive quantization degrade model competence equally across task domains, or do complex tasks (e.g., strict instruction following and code generation) suffer a catastrophic collapse while basic factual retrieval remains unaffected?
4. **Network vs. Compute Decoupling**: In cloud-native client–server architectures where inference is served behind API gateways and tunnels (e.g., Ngrok WAN), how much latency is inherent to network transit versus server-side prompt prefill (TTFT) and GPU decoding?

**Objective**: Empirically quantify the trade-offs across `Q4_K_M` (4.5 bpw), `Q8_0` (8.5 bpw), and `fp16` (16.0 bpw) using `Qwen/Qwen2.5-1.5B-Instruct-GGUF` deployed on an NVIDIA Tesla T4 GPU, identify the exact Pareto frontier, and locate the **"Knee of the Curve"** where marginal quality gains diminish to zero relative to memory cost.

---

## System Architecture

```mermaid
flowchart TD
    subgraph Client["Local Evaluation Client (Laptop / On-Premises)"]
        direction TB
        A[Benchmark Harness: quant_bench.py] -->|1. Switch Model Request| B(Remote Model Manager)
        A -->|2. Streamed Prompt Invocations| C(Streaming Request Client)
        C -->|Measure TTFT & Response Arrival| D[Latency Isolator: T_net = T_client - T_server]
        A -->|3. Evaluate Generated Text| E[Scoring Engine: metrics.py]
        E --> F[ROUGE-1, ROUGE-L, Keyword Hit Rate, Composite Score]
        A -->|4. Save Empirical Records| G[CSV Logs in results/]
        G --> H[Plotting Suite: plot_deep_dive.py]
    end

    subgraph CloudServer["Google Colab Cloud GPU (NVIDIA Tesla T4 16GB)"]
        direction TB
        T1["Public Ngrok Tunnel (HTTPS Gateway)"] --> T2[FastAPI Server :8000 (quant_server.py)]
        T2 -->|Dynamic Hot-Swap| T3["llama-cpp-python (CUDA Engine)"]
        T3 -->|Model 1: 4.5 bpw| M1["Qwen2.5-1.5B-Q4_K_M (1.04 GB)"]
        T3 -->|Model 2: 8.5 bpw| M2["Qwen2.5-1.5B-Q8_0 (1.76 GB)"]
        T3 -->|Model 3: 16.0 bpw| M3["Qwen2.5-1.5B-fp16 (3.32 GB)"]
        T2 -->|NVML Realtime Telemetry| T4[VRAM & GPU Utilization Tracker]
    end

    C -.->|Public Internet WAN Transit ~294ms| T1
```

---

## Repository File Index

A complete directory and file overview explaining the responsibility of each component:

```
.
├── README.md                              # Main documentation, problem statement, and execution guide
├── .gitignore                             # Git ignore rules for venv, cached models, and checkpoints
├── models/
│   ├── .gitkeep                           # Git tracking placeholder for model directory
│   └── qwen2.5-1.5b-instruct-q4_k_m.gguf  # Local GGUF weights (downloaded on demand, gitignored)
├── server/
│   ├── quant_server.py                    # FastAPI inference server with dynamic model swapping,
│   │                                      # streaming TTFT measurement, and NVML GPU/VRAM telemetry
│   ├── local_server.py                    # Baseline FastAPI server for local CPU vs Cloud GPU benchmark
│   └── requirements.txt                   # Serving dependencies (fastapi, uvicorn, pynvml, psutil, aiohttp)
├── colab/
│   ├── quant_server_colab.ipynb           # 5-cell Google Colab notebook for automated CUDA setup,
│   │                                      # GGUF weights download, server launch, and Ngrok tunnel
│   ├── colab_setup.ipynb                  # Baseline Colab notebook for CPU vs GPU serving
│   └── colab_setup.py                     # Script version of baseline Colab environment
├── benchmark/
│   ├── quant_bench.py                     # Automated client evaluation suite for quantization benchmark;
│   │                                      # triggers dynamic switching, streams tokens, logs CSVs
│   ├── eval_dataset.json                  # 10-task evaluation dataset across 6 domains (QA, Reasoning,
│   │                                      # Deductive Logic, Python Code, Summarization, Strict Instructions)
│   ├── metrics.py                         # Evaluation engine computing ROUGE-1 F1, ROUGE-L F1,
│   │                                      # Factual Keyword Hit Rate, and Composite Quality
│   ├── plot_detailed_quant.py             # 4-panel & 6-panel publication-grade visualization script
│   ├── plot_deep_dive.py                  # High-density empirical suite (Per-Task Breakdown, Heatmap,
│   │                                      # 4-Panel Variance Distribution, Pareto Knee Frontier)
│   ├── bench.py                           # Baseline benchmark harness for CPU vs GPU context length tests
│   └── plot_bench.py                      # Baseline plotting script for CPU vs GPU bar & line graphs
├── results/
│   ├── quant_benchmark_summary.csv        # Aggregated empirical summary per quantization tier
│   ├── quant_eval_details.csv             # Granular per-request logs (30 evaluations: 10 tasks × 3 models)
│   └── benchmark_results.csv              # Baseline CPU vs GPU empirical results
└── charts/
    ├── quant_pareto_knee_frontier.png     # Multi-Objective Pareto Frontier with embedded Marginal Table
    ├── quant_per_task_breakdown.png       # 10-Task Granular Bar Chart highlighting the Instruction Cliff
    ├── quant_task_performance_heatmap.png # Dual Matrix Heatmap (Quality % and Throughput tok/s)
    ├── quant_distribution_variance_analysis.png # 4-Panel Box Plot & Jitter Analysis (Throughput, Latency, TTFT, Quality)
    ├── quant_comprehensive_6panel_dashboard.png # 6-Panel Architectural Summary Dashboard
    ├── quant_latency_decomposition.png    # Stacked Latency Bar Chart (294ms WAN vs TTFT vs GPU Decode)
    ├── quant_quality_metrics_breakdown.png # Granular Quality Sub-metrics (ROUGE-1, ROUGE-L, Keywords)
    ├── quant_category_radar.png           # Multi-domain competency radar polygon
    └── quant_quality_vs_memory.png        # Quality vs. Memory Pareto Frontier
```

---

## Empirical Results & The Pareto Knee

### Part 1: Local CPU vs. Cloud GPU Baseline

Evaluated across varying context lengths (128 tokens vs. 1024 tokens) on `Qwen2.5-1.5B-Q4_K_M`:

| Metric Dimension | Local CPU (Intel/AMD) | Cloud GPU (Tesla T4) | Delta / Speedup |
| :--- | :--- | :--- | :--- |
| **Short Prompt Latency (128 tok)** | 3.38 s | **1.64 s** | **2.07× faster on GPU** |
| **Long Prompt Latency (1024 tok)** | 5.06 s | **1.69 s** | **2.99× faster on GPU** |
| **Short Prompt TTFT** | 44.1 ms | **22.9 ms** | 1.93× faster on GPU |
| **Long Prompt TTFT** | 1,055.0 ms (1.06 s) | **68.1 ms** | **15.49× faster on GPU** |
| **Token Throughput** | 32.0 – 38.4 tok/s | **78.9 – 79.8 tok/s** | **~2.1× – 2.5× higher throughput** |
| **Isolated Network Latency** | ~2.5 ms (localhost) | ~625 ms (ngrok WAN) | Cloud carries ~620 ms WAN tax |
| **Host Resource Impact** | 73% – 76% CPU saturation | 13.5% RAM, 12.6% VRAM | Offloads host CPU completely |

---

### Part 2: Quantization Precision Benchmark (`Q4_K_M` vs `Q8_0` vs `fp16`)

Evaluated on an **NVIDIA Tesla T4 GPU (16 GB VRAM)** via Ngrok HTTPS tunnel across 30 empirical trials (10 tasks × 3 quantization levels):

| Quantization Tier | Nominal Precision | Weight File Size | Active GPU VRAM | Server TTFT | Token Throughput | Mean Composite Quality | Isolated WAN Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Q4_K_M`** | 4.5 bpw | 1.04 GB | 1.80 GB | 44.90 ms | **127.61 tok/s** | 46.10% | 293.51 ms |
| **`Q8_0`** *(The Knee)* | 8.5 bpw | 1.76 GB | 2.43 GB | **23.85 ms** | **96.22 tok/s** | 45.37% | 293.91 ms |
| **`fp16`** | 16.0 bpw | 3.32 GB | 3.78 GB | 32.78 ms | 59.90 tok/s | 45.07% | 293.85 ms |

#### Key Analytical Insights
1. **Network Decoupling Invariance**: Across all 30 benchmark queries, the measured isolated network delay remained rock-solid at **$293.5\text{ ms} - 293.9\text{ ms}$** ($\Delta < 0.4\text{ ms}$, $\sigma < 5.7\text{ ms}$), empirically validating that WAN transit overhead is strictly decoupled from server-side model precision.
2. **GPU Throughput Scaling**: Generation throughput scales inversely with bit precision:
   - `Q4_K_M` achieves **127.6 tok/s** (+32.6% over Q8, +113.0% over fp16).
   - `Q8_0` delivers **96.2 tok/s** (+60.6% over fp16).
   - `fp16` achieves **59.9 tok/s**.
3. **The Instruction Following Cliff**: While factual question answering and JSON extraction exhibit high accuracy parity across all tiers ($87\% - 97\%$), complex instruction following (`instruct_01`: strict JSON schema constraint) exhibits an **Instruction Cliff**:
   - `Q4_K_M` collapsed to **$11.26\%$** quality score due to syntax formatting failure.
   - `Q8_0` recovered to **$21.45\%$** ($+90.5\%$ relative recovery), matching `fp16` ($21.53\%$).

---

### Locating the Pareto "Knee of the Curve"

To rigorously determine the optimal quantization level, we compute the **Marginal Efficiency of Transition**:

$$\text{Marginal Return} = \frac{\Delta \text{Quality}}{\Delta \text{VRAM}}$$

| Transition | $\Delta \text{VRAM}$ | $\Delta \text{Throughput}$ | Quality Impact | Efficiency Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **$\text{Q4\_K\_M} \to \text{Q8\_0}$** | $+0.63\text{ GB } (+35\%)$ | $-31.4\text{ tok/s } (-25\%)$ | **$+90.5\%$ recovery on complex instructions**; overall score par | **★ Optimal Knee Point** |
| **$\text{Q8\_0} \to \text{fp16}$** | $+1.35\text{ GB } (+55\%)$ | $-36.3\text{ tok/s } (-38\%)$ | **$+0.0\%$ improvement** (45.37% vs 45.07%) | **Negative Marginal Return** |

**Empirical Conclusion**: **`Q8_0` is the Knee of the Curve**. Upgrading from `Q8_0` to `fp16` costs +1.35 GB of VRAM (+55%) and degrades generation speed by 37.7% ($96.2 \to 59.9\text{ tok/s}$) with **zero measurable improvement in output fidelity**. Conversely, downgrading to `Q4_K_M` introduces unpredictable degradation on constrained formatting tasks. Therefore, `Q8_0` represents the global Pareto-optimal choice for production cloud deployments.

---

## Visualizations & Performance Charts

### 1. Multi-Objective Pareto Frontier & Marginal Return
![Pareto Frontier](charts/quant_pareto_knee_frontier.png)

### 2. Task-by-Task Granular Evaluation (The Instruction Cliff)
![Per-Task Breakdown](charts/quant_per_task_breakdown.png)

### 3. Dual Granular Performance Heatmap Matrix (Quality vs. Speed)
![Task Performance Heatmap](charts/quant_task_performance_heatmap.png)

### 4. Empirical Variance & Distribution Analysis (4-Panel Box Plot)
![Distribution Variance Analysis](charts/quant_distribution_variance_analysis.png)

### 5. Latency Decomposition (294ms WAN vs. Prefill vs. Decoding)
![Latency Decomposition](charts/quant_latency_decomposition.png)

### 6. Domain Competency Radar & 6-Panel Architectural Dashboard
| Domain Competency Radar | Comprehensive 6-Panel Suite |
| :---: | :---: |
| ![Radar Chart](charts/quant_category_radar.png) | ![6-Panel Dashboard](charts/quant_comprehensive_6panel_dashboard.png) |

---

## Step-by-Step Execution Guide

### Prerequisites

Clone the repository and initialize the virtual environment on your local machine:

```bash
git clone https://github.com/philomaths6-sys/CC_project.git
cd CC_project

python3 -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
pip install matplotlib pandas numpy requests
```

---

### Track A: Running the Quantization Benchmark on Colab GPU

#### Step 1: Open and Run Google Colab Notebook
1. Open [`colab/quant_server_colab.ipynb`](colab/quant_server_colab.ipynb) in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Execute **Cell 1**: Installs pre-compiled CUDA wheels for `llama-cpp-python`, FastAPI, and monitoring libraries.
4. Execute **Cell 2**: Downloads the three quantized model files (`Q4_K_M`, `Q8_0`, `fp16`) directly from HuggingFace to `/content/models/`.
5. Execute **Cell 3**: Uploads [`server/quant_server.py`](server/quant_server.py) to `/content/quant_server.py`.
6. Execute **Cell 4**: Starts the Uvicorn server as a background daemon on port 8000.
7. Execute **Cell 5**: Authenticates your free Ngrok token and opens the public HTTPS tunnel:
   ```python
   # The cell prints your active public endpoint:
   # Public URL: https://cocoa-chuck-chrome.ngrok-free.dev
   ```

#### Step 2: Execute Client Benchmark Suite Locally
Run the client benchmark harness using your active Ngrok URL:

```bash
python benchmark/quant_bench.py --url https://<YOUR-NGROK-DOMAIN>.ngrok-free.dev
```

The harness will:
- Sequentially call `/switch_model` to load `Q4_K_M`, `Q8_0`, and `fp16` on the GPU.
- Run all 10 tasks in [`benchmark/eval_dataset.json`](benchmark/eval_dataset.json) against each model.
- Stream tokens, calculate TTFT, isolate WAN latency, and compute ROUGE & keyword metrics.
- Output clean summaries to [`results/quant_benchmark_summary.csv`](results/quant_benchmark_summary.csv) and [`results/quant_eval_details.csv`](results/quant_eval_details.csv).

#### Step 3: Render All Publication-Grade Visualizations
Run the plotting scripts to generate all charts:

```bash
python benchmark/plot_detailed_quant.py
python benchmark/plot_deep_dive.py
```
Generated PNG charts will be saved directly into [`charts/`](charts/).

---

### Track B: Running the CPU vs. GPU Baseline Benchmark

#### Step 1: Run Local CPU Server
```bash
python server/local_server.py
```
The server binds to `http://localhost:8000`.

#### Step 2: Run Baseline Benchmark
```bash
python benchmark/bench.py --url http://localhost:8000 --label "Local CPU"
```

#### Step 3: Run Cloud GPU Baseline
Deploy [`colab/colab_setup.ipynb`](colab/colab_setup.ipynb) on Colab, copy the Ngrok URL, and execute:
```bash
python benchmark/bench.py --url https://<NGROK-URL>.ngrok-free.dev --label "Cloud GPU (T4)"
```

#### Step 4: Generate Baseline Plots
```bash
python benchmark/plot_bench.py
```

---

## API Reference

The server in [`server/quant_server.py`](server/quant_server.py) exposes the following REST endpoints:

### 1. `GET /health`
Verifies server health, current active quantization model, and GPU backend status.
- **Response**:
```json
{
  "status": "ok",
  "current_quant_level": "Q8_0",
  "current_model_path": "/content/models/qwen2.5-1.5b-instruct-q8_0.gguf",
  "disk_size_gb": 1.76,
  "gpu": true,
  "threads": 4,
  "system_info": "AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | CUDA = 1 ..."
}
```

### 2. `POST /switch_model`
Hot-swaps the active model in GPU VRAM without restarting the server process.
- **Request**:
```json
{
  "quant_level": "Q8_0"
}
```
- **Response**:
```json
{
  "status": "switched",
  "quant_level": "Q8_0",
  "model_path": "/content/models/qwen2.5-1.5b-instruct-q8_0.gguf",
  "disk_size_gb": 1.76,
  "use_gpu": true
}
```

### 3. `POST /generate`
Executes token generation on the GPU. The server returns the generated completion text along with raw server-side timing and token counters. All client metrics (network transit delay, client wall-clock latency, throughput, ROUGE scores) are computed by the client evaluation harness.
- **Request**:
```json
{
  "prompt": "What is the capital of Australia and when was it chosen?",
  "max_tokens": 150,
  "temperature": 0.0
}
```
- **Response**:
```json
{
  "text": "The capital of Australia is Canberra. It was chosen as the capital in 1908.",
  "ttft": 0.02385,
  "total_latency": 0.428,
  "tokens_generated": 18,
  "prompt_tokens": 24,
  "quant_level": "Q8_0"
}
```
- **Response Fields Explained**:
  - `text` *(string)*: Full decoded text generated by the model.
  - `ttft` *(float)*: Server-side Time to First Token (prompt evaluation duration in seconds, e.g. `0.02385 s` = `23.85 ms`).
  - `total_latency` *(float)*: Server-side total execution duration in seconds (prefill + decoding).
  - `tokens_generated` *(int)*: Exact count of output tokens produced.
  - `prompt_tokens` *(int)*: Token count of input prompt.
  - `quant_level` *(string)*: Model quantization tier used for generation.

### 4. `GET /stats`
Returns instantaneous host telemetry from NVML (NVIDIA Management Library) and `psutil`.
- **Response**:
```json
{
  "cpu_percent": 18.2,
  "ram_percent": 27.0,
  "ram_used_gb": 3.42,
  "ram_total_gb": 12.67,
  "gpu_available": true,
  "gpu_percent": 84,
  "vram_percent": 16.2,
  "vram_used_gb": 2.43,
  "vram_total_gb": 15.0
}
```

---

## Hardware & Software Specifications

- **Cloud GPU Server**: Google Colab Compute Instance
  - **GPU**: NVIDIA Tesla T4 (15.0 GB available VRAM, Turing Architecture, SM 7.5)
  - **Host RAM**: 12.67 GB System Memory
  - **Inference Engine**: `llama-cpp-python` with CUDA 12.x acceleration (`n_gpu_layers=-1` full offload)
  - **Network Tunnel**: Ngrok Reverse Proxy Tunnel over TLS 1.3
- **Local Client**:
  - **OS**: Linux (x86_64)
  - **Runtime**: Python 3.12 / 3.11 virtual environment
  - **Libraries**: `requests`, `aiohttp`, `numpy`, `pandas`, `matplotlib`
