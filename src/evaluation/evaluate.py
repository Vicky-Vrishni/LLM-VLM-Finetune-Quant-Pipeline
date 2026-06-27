import sys
import os
import json
import time
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from config_loader import load_config
from logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/quantization.yaml",
        help="Path to quantization config YAML file",
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=50,
        help="Kitne samples perplexity calculation ke liye use karne hain",
    )
    return parser.parse_args()


def calculate_perplexity(model, tokenizer, eval_texts: list, device: str = "cuda") -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for text in eval_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            if not torch.isnan(loss):
                total_loss += loss.item()
                total_samples += 1

    avg_loss = total_loss / max(total_samples, 1)
    perplexity = float(torch.exp(torch.tensor(avg_loss)))

    return perplexity


def compare_base_vs_finetuned(
    base_model_name: str,
    finetuned_model_dir: str,
    eval_dataset_name: str,
    num_samples: int,
) -> dict:
    logger.info("Loading evaluation samples...")
    eval_data = load_dataset(eval_dataset_name, split="train").select(range(num_samples))
    eval_texts = [
        f"### Instruction:\n{ex.get('instruction', '')}\n\n### Response:\n{ex.get('response', '')}"
        for ex in eval_data
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- Base model perplexity ----------
    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16
    ).to(device)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    logger.info("Calculating base model perplexity...")
    base_perplexity = calculate_perplexity(base_model, base_tokenizer, eval_texts, device)
    logger.info(f"Base model perplexity: {base_perplexity:.4f}")

    del base_model
    torch.cuda.empty_cache()

    # ---------- Fine-tuned model perplexity ----------
    logger.info(f"Loading fine-tuned model: {finetuned_model_dir}")
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        finetuned_model_dir, torch_dtype=torch.bfloat16
    ).to(device)
    finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_dir)

    logger.info("Calculating fine-tuned model perplexity...")
    finetuned_perplexity = calculate_perplexity(
        finetuned_model, finetuned_tokenizer, eval_texts, device
    )
    logger.info(f"Fine-tuned model perplexity: {finetuned_perplexity:.4f}")

    del finetuned_model
    torch.cuda.empty_cache()

    improvement_pct = ((base_perplexity - finetuned_perplexity) / base_perplexity) * 100

    return {
        "base_model_perplexity": round(base_perplexity, 4),
        "finetuned_model_perplexity": round(finetuned_perplexity, 4),
        "improvement_percent": round(improvement_pct, 2),
    }


def get_folder_size_mb(folder_path: str) -> float:
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)


def measure_inference_speed(model, tokenizer, prompt: str, device: str, num_runs: int = 5) -> dict:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    latencies = []
    total_tokens_generated = 0

    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            end = time.time()

            latencies.append((end - start) * 1000)  # ms me convert
            total_tokens_generated += output.shape[1] - inputs["input_ids"].shape[1]

    avg_latency_ms = sum(latencies) / len(latencies)
    total_time_sec = sum(latencies) / 1000
    tokens_per_second = total_tokens_generated / total_time_sec if total_time_sec > 0 else 0

    return {
        "avg_latency_ms": round(avg_latency_ms, 2),
        "tokens_per_second": round(tokens_per_second, 2),
    }


def compare_quantization_methods(config: dict) -> dict:
    results = {}

    merged_dir = "outputs/merged_model"
    if os.path.exists(merged_dir):
        results["original_merged_fp16"] = {
            "size_mb": round(get_folder_size_mb(merged_dir), 2)
        }

    bnb_dir = config["bitsandbytes"]["output_dir"]
    if os.path.exists(bnb_dir):
        results["bitsandbytes_4bit"] = {
            "size_mb": round(get_folder_size_mb(bnb_dir), 2)
        }

    gguf_dir = config["gguf"]["output_dir"]
    if os.path.exists(gguf_dir):
        for level in config["gguf"]["quant_levels"]:
            gguf_file = os.path.join(gguf_dir, f"model-{level}.gguf")
            if os.path.exists(gguf_file):
                size_mb = os.path.getsize(gguf_file) / (1024 * 1024)
                results[f"gguf_{level}"] = {"size_mb": round(size_mb, 2)}

    awq_dir = config["awq"]["output_dir"]
    if os.path.exists(awq_dir):
        results["awq_4bit"] = {
            "size_mb": round(get_folder_size_mb(awq_dir), 2)
        }

    return results


def main():
    args = parse_args()
    config = load_config(args.config)

    report = {}

    # ---------- Part 1: Perplexity Comparison ----------
    logger.info("=" * 60)
    logger.info("Running perplexity comparison (base vs fine-tuned)...")
    logger.info("=" * 60)

    try:
        perplexity_results = compare_base_vs_finetuned(
            base_model_name=config["input"]["base_model"],
            finetuned_model_dir="outputs/merged_model",
            eval_dataset_name=config["awq"]["calibration_dataset"],
            num_samples=args.num_eval_samples,
        )
        report["perplexity_comparison"] = perplexity_results
    except Exception as e:
        logger.warning(f"Perplexity comparison failed: {e}")
        report["perplexity_comparison"] = {"error": str(e)}

    # ---------- Part 2: Quantization Size Comparison ----------
    logger.info("=" * 60)
    logger.info("Comparing quantization methods (size)...")
    logger.info("=" * 60)

    quant_comparison = compare_quantization_methods(config)
    report["quantization_comparison"] = quant_comparison

    # ---------- Save Report ----------
    report_path = config["evaluation"]["report_output"]
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report saved to: {report_path}")
    logger.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()