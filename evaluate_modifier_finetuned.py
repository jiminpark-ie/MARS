from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from mr_metrics import all_code_metrics
from config import add_auth_args, apply_auth_args_to_env, resolve_hf_token

ADAPTER_SPECS: Dict[str, Dict] = {
    "requires_scheduler": {"response_template": "### Classification:\n",
                           "max_new_tokens": 1},
    "modification_type": {"response_template": "### Type:\n",
                          "max_new_tokens": 1},
}

STOP_KEYWORDS = ["### User", "###", "User:", "Assistant:", "```"]


def load_test_data(path: str) -> pd.DataFrame:
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))
    else:
        df = pd.read_csv(path)
    if "question" not in df.columns:
        raise ValueError("test data needs a 'question' column")
    return df


def load_model(base_model: str, adapter_dir: str, hf_token: Optional[str]
               ) -> Tuple[torch.nn.Module, AutoTokenizer, List[str]]:
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
        token=hf_token,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    folders = sorted(
        d for d in os.listdir(adapter_dir)
        if os.path.isdir(os.path.join(adapter_dir, d))
        and os.path.exists(os.path.join(adapter_dir, d, "adapter_config.json"))
    )
    if not folders:
        raise FileNotFoundError(f"no adapters found under {adapter_dir}")
    print(f"adapters found: {folders}")
    model = PeftModel.from_pretrained(
        base, os.path.join(adapter_dir, folders[0]),
        adapter_name=folders[0], is_trainable=False)
    for name in folders[1:]:
        model.load_adapter(os.path.join(adapter_dir, name), adapter_name=name)
    model.eval()
    return model, tokenizer, folders


def make_prompt(question: str, adapter_name: str) -> str:
    template = ADAPTER_SPECS[adapter_name]["response_template"]
    return f"### Request:\n{question}\n{template}\n"


def trim_reply(reply: str) -> str:
    cut = len(reply)
    for kw in STOP_KEYWORDS:
        idx = reply.find(kw)
        if idx != -1:
            cut = min(cut, idx)
    return reply[:cut].strip()


@torch.no_grad()
def run_inference(model, tokenizer, prompts: List[str], adapter_name: str,
                  available: List[str]) -> List[str]:
    if adapter_name not in available:
        raise ValueError(f"adapter {adapter_name!r} not in {available}")
    model.set_adapter(adapter_name)
    max_new = ADAPTER_SPECS[adapter_name]["max_new_tokens"]
    out: List[str] = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=2048).to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        reply = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()
        out.append(trim_reply(reply))
    return out


def _first_digit(text: str) -> Optional[int]:
    match = re.search(r"\d", str(text))
    return int(match.group(0)) if match else None


def binary_classification_metrics(predictions: List[str],
                                  labels: List) -> Dict[str, float]:
    pairs = [(1 if "1" in str(p) else 0, int(l))
             for p, l in zip(predictions, labels)
             if str(l) in ("0", "1")]
    if not pairs:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "total": 0}
    preds, gold = zip(*pairs)
    tp = sum(1 for p, l in pairs if p == 1 and l == 1)
    tn = sum(1 for p, l in pairs if p == 0 and l == 0)
    fp = sum(1 for p, l in pairs if p == 1 and l == 0)
    fn = sum(1 for p, l in pairs if p == 0 and l == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"accuracy": (tp + tn) / len(pairs), "precision": precision,
            "recall": recall, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": len(pairs)}


def multiclass_classification_metrics(predictions: List[str],
                                      labels: List) -> Dict[str, float]:
    pairs = [(_first_digit(p), int(l)) for p, l in zip(predictions, labels)
             if str(l).isdigit()]
    if not pairs:
        return {"accuracy": 0.0, "macro_precision": 0.0,
                "macro_recall": 0.0, "macro_f1": 0.0, "total": 0}
    classes = sorted({l for _, l in pairs})
    accuracy = sum(1 for p, l in pairs if p == l) / len(pairs)
    precisions, recalls, f1s = [], [], []
    for c in classes:
        tp = sum(1 for p, l in pairs if p == c and l == c)
        fp = sum(1 for p, l in pairs if p == c and l != c)
        fn = sum(1 for p, l in pairs if p != c and l == c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall)
                   if (precision + recall) else 0.0)
    return {"accuracy": accuracy,
            "macro_precision": float(np.mean(precisions)),
            "macro_recall": float(np.mean(recalls)),
            "macro_f1": float(np.mean(f1s)),
            "total": len(pairs)}


def evaluate_adapter(df: pd.DataFrame, adapter_name: str,
                     with_codebert: bool) -> Dict[str, float]:
    label_col = f"label_{adapter_name}"
    if label_col not in df.columns:
        raise ValueError(f"test data needs column {label_col!r}")
    predictions = df[f"{adapter_name}_pred"].tolist()
    labels = df[label_col].tolist()
    if adapter_name == "requires_scheduler":
        return binary_classification_metrics(predictions, labels)
    if adapter_name == "modification_type":
        return multiclass_classification_metrics(predictions, labels)
    pairs = [(str(p), str(l)) for p, l in zip(predictions, labels)
             if str(l).strip()]
    preds, gold = ([p for p, _ in pairs], [l for _, l in pairs])
    return all_code_metrics(preds, gold, with_codebert=with_codebert)


def main():
    parser = argparse.ArgumentParser()
    add_auth_args(parser)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model", default=os.environ.get(
        "MARS_BASE_MODEL", "meta-llama/Llama-3.1-8B"))
    parser.add_argument("--adapters", nargs="+",
                        default=list(ADAPTER_SPECS.keys()),
                        choices=list(ADAPTER_SPECS.keys()))
    parser.add_argument("--output-dir", default="./results_finetuned")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--no-codebert", action="store_true")
    args = parser.parse_args()
    apply_auth_args_to_env(args)
    hf_token = resolve_hf_token(args.hf_token, required=False)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = load_test_data(args.test_data)
    model, tokenizer, available = load_model(
        args.base_model, args.adapter_dir, hf_token)

    all_metrics = []
    for run_idx in range(1, args.runs + 1):
        run_df = df.copy()
        for adapter_name in args.adapters:
            prompts = [make_prompt(q, adapter_name)
                       for q in run_df["question"]]
            run_df[f"{adapter_name}_pred"] = run_inference(
                model, tokenizer, prompts, adapter_name, available)
            metrics = evaluate_adapter(run_df, adapter_name,
                                       with_codebert=not args.no_codebert)
            metrics.update({"run": run_idx, "adapter": adapter_name})
            all_metrics.append(metrics)
            print(f"[run {run_idx}] {adapter_name}: "
                  + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                              if isinstance(v, float)))
        run_df.to_csv(os.path.join(
            args.output_dir, f"predictions_run{run_idx}_{timestamp}.csv"),
            index=False)

    pd.DataFrame(all_metrics).to_csv(
        os.path.join(args.output_dir, f"metrics_{timestamp}.csv"), index=False)
    print(f"results written to {args.output_dir}")


if __name__ == "__main__":
    main()
