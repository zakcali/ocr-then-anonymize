# Medical Document Anonymization Pipeline

This directory contains the core orchestration scripts used to run various Large Language Models (LLMs) locally via vLLM to anonymize Turkish medical records. The goal is to detect and redact 18 categories of HIPAA Protected Health Information (PHI) from OCR'd Markdown documents.

## Directory Structure

```text
anonymize/
├── input/                          # Source Markdown files (from OCR stage)
├── anonym_system.txt               # System prompt defining the AI's role
├── anonym_pre.txt                  # PRE-PROMPT (Labelling Mode): Wraps PHI in XML tags (e.g. <NAME>...</NAME>)
├── anonym_pre-for-real-tag.txt     # PRE-PROMPT (Production Mode): Replaces PHI entirely (e.g. [NAME])
├── vllm-scripts.txt                # vLLM launch commands and GPU power management
└── [model-name].py                 # Python async clients tailored for specific local models
```

## Anonymization Modes

The pipeline supports two distinct modes of operation by swapping the contents of the pre-prompt file (`anonym_pre.txt`):

1. **Evaluation / Labelling Mode (Default):**
   - **Goal:** To measure local LLM performance against a "Gold Standard" human annotation.
   - **Action:** The model *wraps* the identified sensitive text in XML tags (e.g., `<NAME>Prof. Dr. Ahmet ATAK</NAME>`).
   - **Usage:** This format is required for importing predictions into Label Studio for human review and metric calculation.

2. **Production Anonymization Mode:**
   - **Goal:** Real-world anonymization to protect patient privacy.
   - **Action:** The model completely *replaces* the sensitive data with bracketed placeholders (e.g., `[NAME]`), leaving no trace of the original text.
   - **Usage:** Rename or overwrite `anonym_pre.txt` with the contents of `anonym_pre-for-real-tag.txt` before running the Python scripts.

## Supported Local Models

Individual Python scripts exist to orchestrate requests to specific models served via vLLM. Supported models include:
- **Gemma Series:** `gemma-27b-it`, `gemma-4-26B-A4B-it`, `gemma-4-31B-it`, `medgemma-27b-text-it`
- **Qwen Series:** `Qwen3-30B-A3B-Instruct`, `Qwen3-30B-A3B-Thinking`, `Qwen3.5-27B`, `Qwen3.5-35B-A3B`, `Qwen3.6-35B-A3B`
- **GPT-OSS:** `gpt-oss-120b`

### Script Features
- **Asynchronous Execution:** Uses `AsyncOpenAI` with `asyncio.Semaphore` to process multiple documents concurrently (configurable limit, default 40).
- **Page Segmentation:** Automatically splits long documents by `<!-- Page N -->` markers (injected during the OCR phase) to prevent context window overflow.
- **Dynamic GPU Monitoring:** Pings the vLLM metrics API (`/metrics`). If KV Cache usage exceeds 84%, the script automatically pauses and waits for the buffer to clear before sending new requests, preventing Out-Of-Memory (OOM) crashes on large batches.

## Hardware Stability & vLLM Serving

**CRITICAL:** Running large, unquantized or heavily parallelized models on 4x RTX 3090s can cause massive, instantaneous power spikes that will trip the Over-Current Protection (OCP) on ATX power supplies, hard-locking or shutting down the Linux machine.

This system is powered by a **1500W Corsair HX1500i PSU**. To maintain stability across all 4 GPUs (96GB total VRAM), you **must** apply the power limits found in `vllm-scripts.txt` *before* starting the vLLM server:

```bash
# 1. Set Power Limit to 250W for ALL cards
sudo nvidia-smi -pl 250

# 2. Lock Clocks to 1500MHz for ALL cards
sudo nvidia-smi -lgc 1500
```

### Starting the vLLM Server

The `vllm-scripts.txt` file contains the exact `uv run vllm serve` commands for each model. They are optimized for the 96GB VRAM pool using:
- Tensor Parallelism (`--tensor-parallel-size 4`)
- Prefix Caching (`--enable-prefix-caching`)
- Expandable Segments (`PYTORCH_ALLOC_CONF=expandable_segments:True`)

**Example (Qwen 3.5 35B):**
```bash
HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True VLLM_MARLIN_USE_ATOMIC_ADD=1 uv run vllm serve Qwen/Qwen3.5-35B-A3B \
 --served-model-name Qwen/Qwen3.5-35B-A3B \
 --max-model-len 40000 \
 --tensor-parallel-size 4 \
 --gpu-memory-utilization 0.90 \
 --enable-auto-tool-choice \
 --tool-call-parser qwen3_coder\
 --enable-expert-parallel \
 --reasoning-parser qwen3 \
 --async-scheduling \
 --enable-prefix-caching \
 --language-model-only \
 --max-num-seqs 50
```

## Usage

1. Place your OCR'd Markdown files in the `anonymize/input/` directory.
2. Choose your anonymization mode (ensure `anonym_pre.txt` contains the correct prompt).
3. Apply the NVIDIA power limits (250W/1500MHz).
4. Start the vLLM server for your chosen model using the command from `vllm-scripts.txt`.
5. Export the `VLLM_URL` environment variable:
   ```bash
   export VLLM_URL="http://localhost:8000/v1"
   ```
6. Run the corresponding Python script:
   ```bash
   python gemma-4-31B-it.py
   ```
7. Anonymized files will be saved in a new folder (e.g., `output-gemma-4-31B-it/`).
