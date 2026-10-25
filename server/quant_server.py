import gc
import os
import time
from typing import Optional

import psutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama, llama_cpp

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _GPU_AVAILABLE = True
except Exception:
    _GPU_HANDLE = None
    _GPU_AVAILABLE = False

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODELS_DIR = os.environ.get("MODELS_DIR", "./models")
DEFAULT_MODEL = os.environ.get("MODEL_PATH", os.path.join(MODELS_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf"))
USE_GPU = os.environ.get("USE_GPU", "1") == "1"
N_THREADS = int(os.environ.get("N_THREADS", os.cpu_count() or 4))
N_CTX = int(os.environ.get("N_CTX", "2048"))

QUANT_MAP = {
    "Q4_K_M": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "Q8_0": "qwen2.5-1.5b-instruct-q8_0.gguf",
    "fp16": "qwen2.5-1.5b-instruct-fp16.gguf"
}

current_quant_level = "Q4_K_M"
current_model_path = DEFAULT_MODEL
llm: Optional[Llama] = None

def load_llama_model(model_path: str, gpu: bool = True):
    global llm
    if llm is not None:
        del llm
        llm = None
        gc.collect()

    print(f"[SERVER] Loading model from: {model_path} | GPU={gpu} | threads={N_THREADS}")
    llm = Llama(
        model_path=model_path,
        n_threads=N_THREADS,
        n_ctx=N_CTX,
        n_gpu_layers=-1 if gpu else 0,
        verbose=False
    )
    print(f"[SERVER] Model loaded successfully: {model_path}")

# Load initial model if file exists
if os.path.exists(DEFAULT_MODEL):
    load_llama_model(DEFAULT_MODEL, gpu=USE_GPU)
else:
    print(f"[SERVER] Default model not found at {DEFAULT_MODEL}. Awaiting /switch_model request.")

app = FastAPI(title="LLM Quantization Inference Server")

class GenRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0

class SwitchRequest(BaseModel):
    quant_level: str
    use_gpu: Optional[bool] = None

@app.get("/health")
def health():
    sys_info = llama_cpp.llama_print_system_info().decode("utf-8", errors="ignore")
    disk_size = os.path.getsize(current_model_path) / (1024**3) if os.path.exists(current_model_path) else 0.0
    return {
        "status": "ok" if llm is not None else "no_model_loaded",
        "current_quant_level": current_quant_level,
        "current_model_path": current_model_path,
        "disk_size_gb": round(disk_size, 2),
        "gpu": USE_GPU,
        "threads": N_THREADS,
        "system_info": sys_info
    }

@app.post("/switch_model")
def switch_model(req: SwitchRequest):
    global current_quant_level, current_model_path, USE_GPU
    lvl = req.quant_level
    if lvl not in QUANT_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown quantization level: {lvl}. Choose from {list(QUANT_MAP.keys())}")
    
    fname = QUANT_MAP[lvl]
    target_path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"Model file {target_path} not found on server. Ensure it was downloaded.")

    if req.use_gpu is not None:
        USE_GPU = req.use_gpu

    load_llama_model(target_path, gpu=USE_GPU)
    current_quant_level = lvl
    current_model_path = target_path
    disk_size = os.path.getsize(target_path) / (1024**3)

    return {
        "status": "switched",
        "quant_level": current_quant_level,
        "model_path": current_model_path,
        "disk_size_gb": round(disk_size, 2),
        "use_gpu": USE_GPU
    }

@app.get("/stats")
def stats():
    ram = psutil.virtual_memory()
    res = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_percent": ram.percent,
        "ram_used_gb": round(ram.used / (1024**3), 2),
        "ram_total_gb": round(ram.total / (1024**3), 2),
        "gpu_available": _GPU_AVAILABLE,
    }
    if _GPU_AVAILABLE:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE)
            mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
            res.update({
                "gpu_percent": util.gpu,
                "vram_percent": round(100 * mem.used / mem.total, 2),
                "vram_used_gb": round(mem.used / (1024**3), 2),
                "vram_total_gb": round(mem.total / (1024**3), 2),
            })
        except Exception:
            pass
    return res

@app.post("/generate")
def generate(req: GenRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="No model is currently loaded. Call /switch_model first.")

    prompt_bytes = req.prompt.encode("utf-8")
    prompt_tokens = llm.tokenize(prompt_bytes)
    n_prompt_tokens = len(prompt_tokens)

    start = time.perf_counter()
    first_token_time = None
    text_parts = []

    stream = llm.create_chat_completion(
        messages=[{"role": "user", "content": req.prompt}],
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stream=True
    )

    for chunk in stream:
        delta = chunk["choices"][0].get("delta", {})
        piece = delta.get("content")
        if piece:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            text_parts.append(piece)

    end = time.perf_counter()
    if first_token_time is None:
        first_token_time = end

    full_text = "".join(text_parts).strip()
    out_tokens = llm.tokenize(full_text.encode("utf-8"), add_bos=False)
    n_output_tokens = len(out_tokens)
    ttft = (first_token_time - start)

    return {
        "text": full_text,
        "ttft": ttft,
        "total_latency": end - start,
        "tokens_generated": n_output_tokens,
        "prompt_tokens": n_prompt_tokens,
        "quant_level": current_quant_level
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
