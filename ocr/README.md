# Medical Document OCR (pdf-ocr-med-multi.py)

This script provides a high-performance, multiprocessing Optical Character Recognition (OCR) pipeline specifically tuned for processing Turkish medical documents. It utilizes a Vision-Language Model (VLM) via an OpenAI-compatible API (like vLLM) to extract text and tables from PDFs and image files into structured Markdown format.

## Features

- **Multiprocessing Support:** Processes multiple files simultaneously (configurable concurrency) to speed up bulk OCR tasks.
- **Multiple File Formats:** Native support for `.pdf`, `.jpg`, `.jpeg`, `.png`, and `.webp` files.
- **VLM-Powered OCR:** Uses `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` (or any configured model) to accurately extract text, preserving medical terminology and numerical data.
- **Markdown Output:** Automatically formats the extracted text, bolding headers, and creating proper Markdown tables (`| | |`) for tabular data.
- **Image Optimization:** Handles image resizing (preventing out-of-memory errors on massive images), removes alpha channels, and standardizes format to JPEG before sending it to the model.

## Prerequisites

- Python 3.x
- Required Python packages: `pdf2image`, `Pillow`, `openai`, `tqdm`
- Poppler (required by `pdf2image` for PDF rendering)
- An active OpenAI-compatible API endpoint (e.g., a local vLLM instance) serving a Vision-Language Model.

## Directory Structure

By default, the script expects the following directory structure:

```text
ocr/
├── pdf-in/       # Place your input PDFs and images here
├── md-out/       # The script will output the Markdown files here
└── pdf-ocr-med-multi.py
```

The script will automatically recreate the folder hierarchy from `pdf-in` into `md-out`.

## Configuration

You can configure the script by editing the variables at the top of `pdf-ocr-med-multi.py`:

- `INPUT_FOLDER`: The directory containing source files (default: `"pdf-in"`).
- `OUTPUT_FOLDER`: The directory where Markdown files will be saved (default: `"md-out"`).
- `API_BASE_URL`: The base URL for the OpenAI-compatible API (default: `"http://localhost:8000/v1"`).
- `API_KEY`: The API key (default: `"EMPTY"` for local vLLM).
- `MODEL_NAME`: The model to use for OCR (default: `"QuantTrio/Qwen3-VL-32B-Instruct-AWQ"`).
- `CONCURRENCY`: The number of files to process simultaneously (default: `8`).
- `DPI`: The rendering DPI for PDF conversion (default: `200`).
- `MAX_IMAGE_DIMENSION`: The maximum pixel dimension for images to prevent VLM crashes (default: `2240`).

## Usage

1. Place your medical documents (PDFs, JPGs, etc.) inside the `pdf-in` directory.
2. Start your local vLLM server. You can use the provided `vllm-script.txt` command:

   ```bash
   HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True VLLM_MARLIN_USE_ATOMIC_ADD=1 \
   uv run vllm serve QuantTrio/Qwen3-VL-32B-Instruct-AWQ \
    --tensor-parallel-size 4 \
    --async-scheduling \
    --trust-remote-code \
    --max-model-len 16384 \
    --enforce-eager \
    --limit-mm-per-prompt '{"video": 0}' \
    --max-num-seqs 10
   ```

3. Ensure the server is accessible at the configured `API_BASE_URL`.
4. Run the script:

   ```bash
   python pdf-ocr-med-multi.py
   ```

5. The processed Markdown files will appear in the `md-out` directory with the exact same folder structure as the input files.

## OCR Prompt Details

The script uses a specialized Turkish prompt for the Vision model:
> "Bu resimdeki Türkçe tıbbi belgeyi Markdown formatına çevir. Kurallar: 1. Metni olduğu gibi, kelimesi kelimesine Türkçe olarak yaz. 2. Tabloları mutlaka Markdown tablosu (| | |) olarak oluştur. 3. Tıbbi terimleri ve sayıları hatasız aktar. 4. Başlıkları kalın yap. 5. Yorum yapma, sadece metni ver."

This ensures the model strictly transcribes the text, perfectly formats tables, pays attention to medical context, and provides clean Markdown without unnecessary conversational text.

## Hardware and Software Environment

This pipeline was developed and tested on the following system configuration:

- **Motherboard:** ASUS WS X299 SAGE
- **RAM:** 128 GB 3200 MHz DDR4
- **GPUs:** 4x NVIDIA RTX 3090 (24 GB VRAM each) — **Total: 96 GB VRAM minimum recommended to run these scripts as configured.**
- **OS:** Linux Mint 21.3
- **NVIDIA Driver:** 580.105.08
- **CUDA Version:** 13.0
