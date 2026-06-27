# 🚀 Fine-Tuning & Quantization Pipeline for LLMs/VLMs

An end-to-end, production-grade pipeline for fine-tuning and quantizing open-source Large Language Models (LLMs) and Vision-Language Models (VLMs). Supports QLoRA fine-tuning, multi-format quantization (bitsandbytes, GGUF, AWQ), FastAPI serving, and Docker deployment.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)

---

## 🎯 Overview

This project implements a complete LLMOps pipeline that takes an open-source LLM/VLM from a pretrained base model all the way to a deployable, quantized inference service. It is designed to mirror real-world production workflows used by ML engineering teams working with foundation models.

**Key capabilities:**
- Fine-tune any HuggingFace-compatible LLM or VLM using **QLoRA (4-bit)**
- Quantize the fine-tuned model using **three different backends** (bitsandbytes, GGUF, AWQ), each optimized for a different deployment scenario
- Serve the model via a **FastAPI REST API**
- Provide an interactive **Gradio demo UI** for non-technical users
- Fully **containerized with Docker** for one-command deployment
- Config-driven design — no hardcoded parameters anywhere in the codebase

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Raw Dataset     │────▶│  Dataset Loader   │────▶│  QLoRA Fine-Tuning │
│ (HF Hub: Dolly/  │     │ (format + tokenize)│     │  (LLM / VLM)       │
│  Cauldron)       │     └──────────────────┘     └─────────┬──────────┘
└─────────────────┘                                          │
                                                              ▼
                                              ┌───────────────────────────┐
                                              │   LoRA → Base Model Merge  │
                                              └─────────────┬─────────────┘
                                                             │
                  ┌──────────────────────────────────────────┼──────────────────────────────────────────┐
                  ▼                                          ▼                                          ▼
        ┌──────────────────┐                      ┌──────────────────┐                      ┌──────────────────┐
        │  bitsandbytes      │                      │  GGUF              │                      │  AWQ               │
        │  (4-bit, fast dev) │                      │  (CPU/edge deploy) │                      │  (production speed)│
        └─────────┬─────────┘                      └─────────┬─────────┘                      └─────────┬─────────┘
                  └──────────────────────────────────────────┼──────────────────────────────────────────┘
                                                              ▼
                                              ┌───────────────────────────┐
                                              │   Evaluation Module        │
                                              │ (perplexity + size compare)│
                                              └─────────────┬─────────────┘
                                                             │
                  ┌──────────────────────────────────────────┼──────────────────────────────────────────┐
                  ▼                                                                                       ▼
        ┌──────────────────┐                                                                  ┌──────────────────┐
        │  FastAPI Server    │                                                                  │  Gradio Demo UI    │
        │  (REST endpoints)  │                                                                  │  (interactive test)│
        └──────────────────┘                                                                  └──────────────────┘
                  │                                                                                       │
                  └───────────────────────────────┬───────────────────────────────────────────────────────┘
                                                   ▼
                                        ┌───────────────────────┐
                                        │  Docker Containers      │
                                        │  (one-command deploy)   │
                                        └───────────────────────┘
```


## ✨ Features

| Feature | Description |
|---|---|
| **QLoRA Fine-Tuning** | 4-bit quantized LoRA fine-tuning for both LLMs and VLMs, trains only ~1-2% of total parameters |
| **Multi-Backend Quantization** | Three production-relevant quantization methods, each with documented trade-offs |
| **Config-Driven Design** | All hyperparameters, model names, and paths live in YAML configs — zero hardcoding |
| **Unified Inference Engine** | Single interface abstracts away LLM vs. VLM differences |
| **REST API** | FastAPI server with automatic Swagger documentation at `/docs` |
| **Interactive Demo** | Gradio UI with separate tabs for text and vision-language generation |
| **Containerized Deployment** | Docker + docker-compose for one-command spin-up of API and demo together |
| **Experiment Tracking** | Weights & Biases integration for training run visibility |
| **Automated Evaluation** | Perplexity comparison (base vs. fine-tuned) and quantization size/speed benchmarking |

---

## 📁 Project Structure

```
llm-vlm-finetune-quant-pipeline/
│
├── configs/                      # YAML configs - single source of truth for all settings
│   ├── llm_finetune.yaml
│   ├── vlm_finetune.yaml
│   └── quantization.yaml
│
├── data/
│   └── dataset_loader.py         # Unified LLM/VLM dataset loading & preprocessing
│
├── src/
│   ├── training/
│   │   ├── lora_config.py        # Reusable LoRA/BnB config builders
│   │   ├── train_llm.py          # QLoRA fine-tuning script for LLMs
│   │   └── train_vlm.py          # QLoRA fine-tuning script for VLMs
│   ├── quantization/
│   │   ├── quantize_bnb.py       # bitsandbytes 4-bit quantization
│   │   ├── quantize_gguf.py      # GGUF conversion via llama.cpp
│   │   └── quantize_awq.py       # AWQ activation-aware quantization
│   ├── evaluation/
│   │   └── evaluate.py           # Perplexity + quantization comparison report
│   ├── inference/
│   │   └── inference_engine.py   # Unified LLM/VLM inference engine
│   └── utils/
│       ├── config_loader.py
│       └── logger.py
│
├── api/
│   ├── main.py                   # FastAPI application
│   └── schemas.py                # Pydantic request/response models
│
├── demo/
│   └── app.py                    # Gradio interactive demo
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yaml
│
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/llm-vlm-finetune-quant-pipeline.git
cd llm-vlm-finetune-quant-pipeline

# Install dependencies
pip install -r requirements.txt
```

**Recommended environment:** Python 3.10+, CUDA-enabled GPU (training/quantization tested on Google Colab T4/A100). CPU-only inference is supported for GGUF-quantized models via llama.cpp.

---

## 🚀 Usage

### 1. Fine-Tune an LLM (QLoRA)

```bash
python src/training/train_llm.py --config configs/llm_finetune.yaml
```

### 2. Fine-Tune a VLM (QLoRA)

```bash
python src/training/train_vlm.py --config configs/vlm_finetune.yaml
```

### 3. Quantize the Fine-Tuned Model

```bash
# bitsandbytes (4-bit, fast for development)
python src/quantization/quantize_bnb.py --config configs/quantization.yaml

# GGUF (for CPU / llama.cpp deployment)
python src/quantization/quantize_gguf.py --config configs/quantization.yaml

# AWQ (production-grade serving speed)
python src/quantization/quantize_awq.py --config configs/quantization.yaml
```

### 4. Evaluate & Compare

```bash
python src/evaluation/evaluate.py --config configs/quantization.yaml
```

### 5. Serve via API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Swagger docs available at http://localhost:8000/docs
```

### 6. Launch Interactive Demo

```bash
python demo/app.py
# Opens at http://localhost:7860
```

### 7. Deploy with Docker (API + Demo together)

```bash
docker-compose -f docker/docker-compose.yaml up --build
```

## 📊 Results

> **Note:** The table below is populated automatically by running `src/evaluation/evaluate.py`, which generates `outputs/quantization_report.json`. Replace these placeholder values with your actual run results before sharing this project.

**Fine-Tuning Impact (Perplexity — lower is better):**

| Model | Perplexity |
|---|---|
| Base model (pre-fine-tuning) | _e.g. 12.4_ |
| Fine-tuned model (QLoRA) | _e.g. 7.8_ |
| **Improvement** | _e.g. ~37%_ |

**Quantization Comparison:**

| Method | Size | Best Use Case |
|---|---|---|
| Original (FP16 merged) | _e.g. 3.1 GB_ | Baseline reference |
| bitsandbytes (4-bit) | _e.g. 0.9 GB_ | Fast local development/testing |
| GGUF (Q4_K_M) | _e.g. 0.85 GB_ | CPU / edge / offline deployment |
| AWQ (4-bit) | _e.g. 0.95 GB_ | Production serving (fastest inference) |

---

## 🛠️ Tech Stack

- **Modeling:** PyTorch, HuggingFace Transformers, PEFT
- **Fine-Tuning:** QLoRA, bitsandbytes
- **Quantization:** bitsandbytes, llama.cpp (GGUF), AutoAWQ
- **Experiment Tracking:** Weights & Biases
- **Serving:** FastAPI, Uvicorn
- **Demo UI:** Gradio
- **Deployment:** Docker, docker-compose
- **Data:** HuggingFace Datasets (Dolly-15k, The Cauldron)

---

## 🔮 Future Improvements

- Add vLLM-based high-throughput serving as an alternative inference backend
- Add automated CI/CD pipeline (GitHub Actions) for testing and Docker image builds
- Extend evaluation with task-specific benchmarks (e.g. MMLU, VQA accuracy)
- Add support for full fine-tuning comparison alongside LoRA
- Kubernetes deployment manifests for horizontal scaling

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋 Author

Built as a demonstration of end-to-end LLMOps engineering — covering fine-tuning, quantization, evaluation, and production deployment of open-source LLMs and VLMs.

