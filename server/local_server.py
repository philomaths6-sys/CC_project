import os
import time

import psutil
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama, llama_cpp

load_dotenv(os.environ.get("ENV_FILE", ".env"))




# Optional GPU monitoring — only available if pynvml is installed AND an
# NVIDIA GPU is present (works locally with a GPU, and inside Colab/Kaggle).
# Falls back gracefully to None if unavailable (e.g. on a CPU-only machine).
try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _GPU_AVAILABLE = True
except Exception:
    _GPU_HANDLE = None
    _GPU_AVAILABLE = False
 




MODEL_PATH = os.environ.get("MODEL_PATH", "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
USE_GPU = os.environ.get("USE_GPU", "0") == "1"   # set USE_GPU=1 env var for GPU runs
N_THREADS = int(os.environ.get("N_THREADS", os.cpu_count() or 8))
N_CTX = 4096


print(f"Loading modle from {MODEL_PATH} | GPU={USE_GPU} | threads={N_THREADS}")

llm = Llama(
    model_path=MODEL_PATH,
    n_threads=N_THREADS,
    n_ctx=N_CTX,
    n_gpu_layers=-1 if USE_GPU else 0,
    verbose=False,
)

SYSTEM_INFO = llama_cpp.llama_print_system_info().decode("utf-8",errors="ignore")
print("llama.cpp system info", SYSTEM_INFO)

app=FastAPI()

class GenRequest(BaseModel):
    prompt:str
    max_tokens:int = 128


@app.get("/health")
def health():
    return {"status":"ok", "system_info":SYSTEM_INFO, "gpu":USE_GPU, "threads":N_THREADS}






@app.get("/stats")
def stats():
    """
    Point-in-time resource snapshot of the machine THIS SERVER is running on.
    Called by bench.py while requests are in flight, to correlate load with
    actual CPU/RAM/GPU/VRAM utilization -- works identically whether this
    server is running locally or inside a Colab notebook (bench.py just hits
    whatever URL it was given, local or via ngrok tunnel).
    """
    ram = psutil.virtual_memory()
    result = {
        "cpu_percent": psutil.cpu_percent(interval=None),          # overall %, all cores
        "cpu_percent_per_core": psutil.cpu_percent(interval=None, percpu=True),
        "ram_percent": ram.percent,
        "ram_used_gb": round(ram.used / (1024**3), 2),
        "ram_total_gb": round(ram.total / (1024**3), 2),
        "gpu_available": _GPU_AVAILABLE,
    }
    if _GPU_AVAILABLE:
        util = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE)
        mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
        result.update({
            "gpu_percent": util.gpu,
            "vram_percent": round(100 * mem.used / mem.total, 2),
            "vram_used_gb": round(mem.used / (1024**3), 2),
            "vram_total_gb": round(mem.total / (1024**3), 2),
        })
    return result





@app.post("/generate")
def generate(req: GenRequest):
    prompt_tokens = llm.tokenize(req.prompt.encode("utf-8"))
    n_prompt_tokens = len(prompt_tokens)

    start= time.perf_counter()
    first_token_time = None
    text_parts =[]
    n_genrated =0


    stream = llm.create_chat_completion(
        messages=[{"role":"user", "content":req.prompt}],
        max_tokens=req.max_tokens,
        stream=True,

    )

    for chunk in stream:
        delta = chunk["choices"][0]["delta"]
        piece = delta.get("content")
        if piece:
            if first_token_time is None:
                first_token_time= time.perf_counter()
            text_parts.append(piece)
            n_genrated += 1 # approximate


    end = time.perf_counter()
    # n_generated above counts stream chunks, not exact tokens (a chunk can be
    # a sub-word piece). For precise tok/s, re-tokenize the output:
    full_text = "".join(text_parts)
    output_tokens = llm.tokenize(full_text.encode("utf-8"), add_bos=False)
    n_output_tokens = len(output_tokens)

    ttft = (first_token_time - start) if first_token_time else None

    return {
        "text": full_text,
        "ttft": ttft,
        "total_latency": end-start,
        "tokens_generated": n_output_tokens,
        "prompt_tokens": n_prompt_tokens,
    }



