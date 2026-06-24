from __future__ import annotations

import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(r"C:\Users\Administrator\Desktop\cpu_only_inference_benchmark_20260625")
LLAMA_CPP_DIR = Path(r"E:\model\llama.cpp")
LLAMA_SERVER = LLAMA_CPP_DIR / "llama-server.exe"
MODEL = Path(r"E:\model\Qwen3-0.6B-IQ4_NL.gguf")
NANO_REPO = Path(r"C:\Users\Administrator\Desktop\nano_vllm_note - 副本")
PYTHON = Path(r"C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe")

PROMPT = "请用三句话解释CPU-only LLM推理中 batch size 和线程数如何影响吞吐与延迟。"
MAX_TOKENS = 48
THREADS = 4
CTX_SIZE = 512
BATCH_SIZE = 512
UBATCH_SIZE = 512
WARMUP_ROUNDS = 1
SERIAL_ROUNDS = 5
PAIRED_ROUNDS = 5
LLAMA_PORT = 18180
NANO_PORT = 18181
TIMEOUT_S = 180


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


RUN_ID = now_id()
RUN_DIR = ROOT / f"run_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=False)


def json_post(url: str, payload: dict, timeout: int = TIMEOUT_S) -> tuple[dict, float]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    elapsed = time.perf_counter() - started
    return json.loads(body), elapsed


def http_get(url: str, timeout: int = 5) -> tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8", errors="replace")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - rank) + ordered[hi] * (rank - lo)


def summarize(rows: list[dict]) -> dict:
    latencies = [row["latency_s"] for row in rows if row.get("ok")]
    tokens = [row["completion_tokens"] for row in rows if row.get("ok")]
    per_request_tps = [row["completion_tok_s"] for row in rows if row.get("ok")]
    total_completion_tokens = sum(tokens)
    total_latency = sum(latencies)
    return {
        "count": len(rows),
        "ok_count": sum(1 for row in rows if row.get("ok")),
        "error_count": sum(1 for row in rows if not row.get("ok")),
        "latency_avg_s": statistics.mean(latencies) if latencies else 0.0,
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p95_s": percentile(latencies, 0.95),
        "latency_min_s": min(latencies) if latencies else 0.0,
        "latency_max_s": max(latencies) if latencies else 0.0,
        "completion_tokens_total": total_completion_tokens,
        "completion_tokens_avg": statistics.mean(tokens) if tokens else 0.0,
        "aggregate_completion_tok_s": total_completion_tokens / total_latency if total_latency else 0.0,
        "per_request_completion_tok_s_avg": statistics.mean(per_request_tps) if per_request_tps else 0.0,
        "per_request_completion_tok_s_p50": percentile(per_request_tps, 0.50),
        "per_request_completion_tok_s_p95": percentile(per_request_tps, 0.95),
    }


def request_payload() -> dict:
    return {
        "model": "Qwen3-0.6B-IQ4_NL",
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }


def extract_completion_tokens(body: dict) -> int:
    usage = body.get("usage") or {}
    value = usage.get("completion_tokens")
    if isinstance(value, int):
        return value
    choices = body.get("choices") or []
    if choices:
        text = choices[0].get("text") or choices[0].get("message", {}).get("content") or ""
        return max(1, len(text))
    if "token_ids" in body:
        return len(body["token_ids"])
    return 0


def extract_text(body: dict) -> str:
    choices = body.get("choices") or []
    if choices:
        return choices[0].get("text") or choices[0].get("message", {}).get("content") or ""
    return body.get("text", "")


def run_one(system: str, base_url: str, phase: str, round_index: int) -> dict:
    url = f"{base_url}/v1/completions"
    payload = request_payload()
    try:
        body, elapsed = json_post(url, payload)
        completion_tokens = extract_completion_tokens(body)
        return {
            "system": system,
            "phase": phase,
            "round": round_index,
            "ok": True,
            "latency_s": elapsed,
            "completion_tokens": completion_tokens,
            "completion_tok_s": completion_tokens / elapsed if elapsed else 0.0,
            "prompt": PROMPT,
            "text_preview": extract_text(body)[:160],
            "error": "",
        }
    except Exception as exc:
        return {
            "system": system,
            "phase": phase,
            "round": round_index,
            "ok": False,
            "latency_s": 0.0,
            "completion_tokens": 0,
            "completion_tok_s": 0.0,
            "prompt": PROMPT,
            "text_preview": "",
            "error": repr(exc),
        }


def wait_ready(name: str, url: str, process: subprocess.Popen, timeout_s: int = 120) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{name} exited early with code {process.returncode}")
        try:
            status, _ = http_get(url, timeout=5)
            if status == 200:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(1)
    raise TimeoutError(f"{name} was not ready after {timeout_s}s; last_error={last_error!r}")


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def start_processes() -> tuple[subprocess.Popen, subprocess.Popen, dict]:
    llama_log = open(RUN_DIR / "llama_server.log", "w", encoding="utf-8", errors="replace")
    nano_log = open(RUN_DIR / "nano_vllm_server.log", "w", encoding="utf-8", errors="replace")

    llama_cmd = [
        str(LLAMA_SERVER),
        "--model",
        str(MODEL),
        "--host",
        "127.0.0.1",
        "--port",
        str(LLAMA_PORT),
        "--ctx-size",
        str(CTX_SIZE),
        "--batch-size",
        str(BATCH_SIZE),
        "--ubatch-size",
        str(UBATCH_SIZE),
        "--threads",
        str(THREADS),
        "--parallel",
        "1",
        "--no-cache-prompt",
        "--no-webui",
        "--metrics",
    ]
    nano_cmd = [
        str(PYTHON),
        "-m",
        "nanovllm.serve",
        "--model",
        str(MODEL),
        "--host",
        "127.0.0.1",
        "--port",
        str(NANO_PORT),
        "--model-backend",
        "gguf",
        "--max-model-len",
        str(CTX_SIZE),
        "--gguf-n-ctx",
        str(CTX_SIZE),
        "--gguf-n-batch",
        str(BATCH_SIZE),
        "--gguf-n-ubatch",
        str(UBATCH_SIZE),
        "--gguf-n-threads",
        str(THREADS),
    ]
    llama_env = os.environ.copy()
    llama_env["PATH"] = str(LLAMA_CPP_DIR) + os.pathsep + llama_env.get("PATH", "")
    nano_env = os.environ.copy()
    nano_env["CUDA_VISIBLE_DEVICES"] = ""

    started = time.perf_counter()
    llama_proc = subprocess.Popen(
        llama_cmd,
        cwd=str(LLAMA_CPP_DIR),
        env=llama_env,
        stdout=llama_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    nano_proc = subprocess.Popen(
        nano_cmd,
        cwd=str(NANO_REPO),
        env=nano_env,
        stdout=nano_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    commands = {
        "llama_cpp": command_text(llama_cmd),
        "nano_vllm": command_text(nano_cmd),
    }
    wait_ready("llama.cpp", f"http://127.0.0.1:{LLAMA_PORT}/v1/models", llama_proc)
    wait_ready("nano_vllm", f"http://127.0.0.1:{NANO_PORT}/readyz", nano_proc)
    commands["startup_wait_s"] = time.perf_counter() - started
    return llama_proc, nano_proc, commands


def terminate_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def collect_metrics() -> dict:
    metrics = {}
    try:
        status, body = http_get(f"http://127.0.0.1:{LLAMA_PORT}/metrics", timeout=10)
        metrics["llama_cpp_status"] = status
        metrics["llama_cpp_metrics_preview"] = body[:4000]
    except Exception as exc:
        metrics["llama_cpp_metrics_error"] = repr(exc)
    try:
        status, body = http_get(f"http://127.0.0.1:{NANO_PORT}/metrics", timeout=10)
        metrics["nano_vllm_status"] = status
        metrics["nano_vllm_metrics"] = json.loads(body)
    except Exception as exc:
        metrics["nano_vllm_metrics_error"] = repr(exc)
    return metrics


def write_outputs(results: list[dict], metadata: dict, metrics: dict) -> None:
    by_phase_system: dict[str, list[dict]] = {}
    for row in results:
        key = f"{row['phase']}::{row['system']}"
        by_phase_system.setdefault(key, []).append(row)
    summaries = {key: summarize(rows) for key, rows in by_phase_system.items()}
    output = {
        "metadata": metadata,
        "summaries": summaries,
        "results": results,
        "metrics": metrics,
    }
    (RUN_DIR / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(RUN_DIR / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "system",
                "phase",
                "round",
                "ok",
                "latency_s",
                "completion_tokens",
                "completion_tok_s",
                "error",
                "text_preview",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    lines = []
    lines.append("# CPU-only inference benchmark: llama.cpp vs modified Nano vLLM")
    lines.append("")
    lines.append(f"- Run id: `{metadata['run_id']}`")
    lines.append(f"- Date: `{metadata['date']}`")
    lines.append(f"- Model: `{MODEL}`")
    lines.append(f"- Prompt: `{PROMPT}`")
    lines.append(f"- Max tokens: `{MAX_TOKENS}`")
    lines.append(f"- Threads: `{THREADS}`")
    lines.append(f"- Context size: `{CTX_SIZE}`")
    lines.append(f"- Batch size / ubatch size: `{BATCH_SIZE}` / `{UBATCH_SIZE}`")
    lines.append(f"- Warmup rounds: `{WARMUP_ROUNDS}`")
    lines.append(f"- Serial measured rounds per system: `{SERIAL_ROUNDS}`")
    lines.append(f"- Paired concurrent rounds: `{PAIRED_ROUNDS}`")
    lines.append(f"- CPU scope: `CPU-only; CUDA_VISIBLE_DEVICES cleared for Nano; n_gpu_layers=0/default for GGUF backends`")
    lines.append("")
    lines.append("## Commands")
    lines.append("")
    lines.append("```powershell")
    lines.append(metadata["commands"]["llama_cpp"])
    lines.append(metadata["commands"]["nano_vllm"])
    lines.append("```")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Phase | System | OK | Avg latency s | P50 latency s | P95 latency s | Avg completion tok/s | Aggregate tok/s | Avg completion tokens |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(summaries):
        phase, system = key.split("::", 1)
        item = summaries[key]
        lines.append(
            f"| {phase} | {system} | {item['ok_count']}/{item['count']} | "
            f"{item['latency_avg_s']:.4f} | {item['latency_p50_s']:.4f} | {item['latency_p95_s']:.4f} | "
            f"{item['per_request_completion_tok_s_avg']:.3f} | {item['aggregate_completion_tok_s']:.3f} | "
            f"{item['completion_tokens_avg']:.1f} |"
        )
    lines.append("")
    lines.append("## Per-request Results")
    lines.append("")
    lines.append("| Phase | Round | System | Latency s | Completion tokens | Completion tok/s | Text preview |")
    lines.append("|---|---:|---|---:|---:|---:|---|")
    for row in results:
        preview = row["text_preview"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {row['phase']} | {row['round']} | {row['system']} | "
            f"{row['latency_s']:.4f} | {row['completion_tokens']} | {row['completion_tok_s']:.3f} | {preview} |"
        )
    lines.append("")
    lines.append("## Bottleneck Analysis")
    lines.append("")
    serial_llama = summaries.get("serial::llama.cpp", {})
    serial_nano = summaries.get("serial::nano_vllm_gguf", {})
    paired_llama = summaries.get("paired_concurrent::llama.cpp", {})
    paired_nano = summaries.get("paired_concurrent::nano_vllm_gguf", {})
    if serial_llama and serial_nano:
        llama_tps = serial_llama.get("aggregate_completion_tok_s", 0.0)
        nano_tps = serial_nano.get("aggregate_completion_tok_s", 0.0)
        if llama_tps and nano_tps:
            faster = "llama.cpp" if llama_tps > nano_tps else "nano_vllm_gguf"
            ratio = max(llama_tps, nano_tps) / max(min(llama_tps, nano_tps), 1e-9)
            lines.append(f"- Serial CPU throughput winner: `{faster}` by about `{ratio:.2f}x` on aggregate completion tok/s.")
        lines.append("- Both systems are executing GGUF on CPU. The comparison mainly measures wrapper/server overhead plus llama.cpp CPU kernel behavior, not CUDA native nano-vLLM scheduling.")
        lines.append("- `llama.cpp` uses the native `llama-server.exe` path. Nano vLLM uses the new `model_backend=gguf` wrapper around `llama-cpp-python`, with a serial lock in the backend.")
        lines.append("- Prompt caching was disabled for `llama.cpp` with `--no-cache-prompt`; Nano GGUF reports prefix cache unsupported. Repeated prompts therefore should not get prefix-cache acceleration.")
        lines.append("- The paired-concurrent phase intentionally creates CPU contention. Treat it as a co-residency stress check, not as the clean standalone throughput ranking.")
    if paired_llama and paired_nano:
        lines.append(
            f"- Paired concurrent aggregate tok/s: llama.cpp `{paired_llama.get('aggregate_completion_tok_s', 0.0):.3f}`, "
            f"Nano GGUF `{paired_nano.get('aggregate_completion_tok_s', 0.0):.3f}`."
        )
    lines.append("- If either system shows lower throughput with similar output token counts, the likely bottleneck is CPU decode kernel/runtime overhead rather than model size or GPU memory. Larger `--threads` sweeps can identify the local CPU saturation point.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `results.json`: full machine-readable results and metrics.")
    lines.append("- `results.csv`: per-request table.")
    lines.append("- `llama_server.log`: llama.cpp server log.")
    lines.append("- `nano_vllm_server.log`: Nano vLLM server log.")
    (RUN_DIR / "cpu_only_inference_benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    for required in (LLAMA_SERVER, MODEL, PYTHON):
        if not required.exists():
            raise FileNotFoundError(required)
    if not NANO_REPO.exists():
        raise FileNotFoundError(NANO_REPO)

    llama_proc = None
    nano_proc = None
    results: list[dict] = []
    metrics: dict = {}
    commands: dict = {}
    try:
        llama_proc, nano_proc, commands = start_processes()
        systems = {
            "llama.cpp": f"http://127.0.0.1:{LLAMA_PORT}",
            "nano_vllm_gguf": f"http://127.0.0.1:{NANO_PORT}",
        }
        for round_index in range(1, WARMUP_ROUNDS + 1):
            for system, base_url in systems.items():
                results.append(run_one(system, base_url, "warmup", round_index))

        for round_index in range(1, SERIAL_ROUNDS + 1):
            for system, base_url in systems.items():
                results.append(run_one(system, base_url, "serial", round_index))

        for round_index in range(1, PAIRED_ROUNDS + 1):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(run_one, system, base_url, "paired_concurrent", round_index)
                    for system, base_url in systems.items()
                ]
                for future in futures:
                    results.append(future.result())
        metrics = collect_metrics()
    finally:
        terminate_process(nano_proc)
        terminate_process(llama_proc)

    metadata = {
        "run_id": RUN_ID,
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": str(PYTHON),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "model": str(MODEL),
        "model_size_bytes": MODEL.stat().st_size,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "threads": THREADS,
        "ctx_size": CTX_SIZE,
        "batch_size": BATCH_SIZE,
        "ubatch_size": UBATCH_SIZE,
        "warmup_rounds": WARMUP_ROUNDS,
        "serial_rounds": SERIAL_ROUNDS,
        "paired_rounds": PAIRED_ROUNDS,
        "commands": commands,
    }
    write_outputs(results, metadata, metrics)
    print(RUN_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
