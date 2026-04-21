# Medical Document OCR and LLM Anonymization Pipeline

This repository contains a complete, end-to-end pipeline for processing Turkish medical documents. It performs Optical Character Recognition (OCR) using Vision-Language Models, redacts Protected Health Information (PHI) using various local Large Language Models (LLMs), and mathematically evaluates their anonymization performance against a human-annotated Gold Standard.

## Pipeline Overview

The project is divided into four sequential stages, each contained within its own directory. Please refer to the specific `README.md` in each folder for detailed instructions, configuration options, and scripts.

### 1. OCR (`/ocr`)
Converts raw medical PDFs and images into structured Markdown text.
* **Technology:** Uses Vision-Language Models (e.g., `Qwen3-VL-32B-Instruct-AWQ`) via vLLM.
* **Features:** Multiprocessing, image optimization, strict adherence to Markdown tables, and preservation of medical terminology.
* **Read more:** [OCR Pipeline Documentation](https://github.com/zakcali/ocr-then-anonymize/tree/main/ocr)

### 2. Anonymization (`/anonymize`)
Identifies and redacts 18 categories of HIPAA Safe Harbor identifiers from the OCR'd Markdown.
* **Technology:** Orchestrates various local LLMs (Gemma, Qwen, GPT-OSS) using `vLLM` and asynchronous Python clients.
* **Modes:** Supports both a "Labelling Mode" (wrapping PHI in XML tags for evaluation) and a "Production Mode" (replacing PHI with placeholder tags like `[NAME]`).
* **Read more:** [Anonymization Pipeline Documentation](https://github.com/zakcali/ocr-then-anonymize/tree/main/anonymize)

### 3. Label Studio Integration (`/label-studio`)
Bridges the local LLM predictions with human evaluation tools.
* **Workflow:** Parses XML tags from the anonymization step, aligns them to exact character coordinates in the original text, and converts them into Label Studio compatible JSON.
* **Evaluation Prep:** Merges multiple model predictions and aligns them with a human-annotated "Gold Standard" for side-by-side comparison.
* **Read more:** [Label Studio Integration Documentation](https://github.com/zakcali/ocr-then-anonymize/tree/main/label-studio)

### 4. Statistics & Evaluation (`/statistics`)
Mathematically evaluates the LLMs' performance in detecting PHI.
* **Metrics:** Calculates Precision, Recall, F1-Score, MCC, and Confusion Matrices using span-union NER evaluation.
* **Reports:** Generates Exact Entity Match Reports (did it guess the right category?) and Binary Masking Reports (did it redact the PHI at all, regardless of category?).
* **Read more:** [Statistics & Evaluation Documentation](https://github.com/zakcali/ocr-then-anonymize/tree/main/statistics)

## Hardware Requirements

Running this pipeline—particularly the large unquantized LLMs via vLLM—requires significant computational resources.

* **Developed and Tested On:** 4x NVIDIA RTX 3090 GPUs (Total: 96 GB VRAM) on Linux Mint 21.3.
* **Power Warning:** Running heavy models across 4 GPUs can cause massive power spikes. A high-capacity power supply (e.g., 1500W Corsair HX1500i) and strict GPU power/clock limits (e.g., 250W / 1500MHz via `nvidia-smi`) are strongly recommended to prevent system instability or lockups. See the `anonymize` folder for specific mitigation commands.

## Getting Started

1. **OCR Stage:** Place your raw medical PDFs/images in `ocr/pdf-in/`. Follow the instructions in `ocr/README.md` to generate Markdown files.
2. **Anonymization Stage:** Move the generated Markdown files to `anonymize/input/`. Run your chosen LLMs via vLLM as detailed in `anonymize/README.md`.
3. **Label Studio Stage (Evaluation):** Process the model outputs using the scripts in `label-studio/` to generate prediction coordinates and merge them with your Gold Standard annotations.
4. **Statistics Stage:** Run the evaluation script in `statistics/` to generate performance reports and heatmaps for the models you tested.