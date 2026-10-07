from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from runtime_config import (
    add_auth_args,
    apply_auth_args_to_env,
    resolve_hf_token,
    resolve_wandb_api_key,
)


@dataclass
class TrainConfig:
    base_model: str = "meta-llama/Llama-3.1-8B"
    data_dir: str = "."
    output_dir: str = "./models/adapters"

    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 1

    max_seq_len_code: int = 1280
    max_seq_len_cls: int = 256
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.15
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optimizer: str = "paged_adamw_32bit"
    val_split_ratio: float = 0.1
    max_val_samples: int = 2000
    seed: int = 42
    logging_steps: int = 100
    eval_steps: int = 100
    save_steps: int = 500


TARGET_MODULES = [
    "q_proj", "v_proj", "k_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "embed_tokens", "lm_head",
]

ADAPTERS: Dict[str, Dict[str, Any]] = {
    "requires_scheduler": {
        "response_template": "### Classification:\n",
        "data_files": "requires_scheduler_train.json",
        "max_len": "cls",
    },
    "modification_type": {
        "response_template": "### Type:\n",
        "data_files": "modification_type_train.json",
        "max_len": "cls",
    },
}


def _find_data_file(adapter_name: str, data_dir: str) -> Path:
    for name in ADAPTERS[adapter_name]["data_files"]:
        path = Path(data_dir) / name
        if path.is_file():
            return path
    names = ", ".join(ADAPTERS[adapter_name]["data_files"])
    raise FileNotFoundError(
        f"No data file for {adapter_name} in {data_dir}. Expected one of: {names}")


def _unwrap_records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        for key in ("data", "train", "records", "samples"):
            if isinstance(payload.get(key), list):
                records = payload[key]
                break
        else:
            raise ValueError("Dataset JSON must contain a list of records.")
    else:
        raise ValueError("Dataset JSON must be a list or contain one.")
    if not all(isinstance(r, dict) for r in records):
        raise ValueError("Every dataset record must be a JSON object.")
    return records


def _record_text(record: Dict[str, Any], adapter_name: str) -> str:
    template = ADAPTERS[adapter_name]["response_template"]
    if "text" in record:
        text = str(record["text"])
        if template not in text:
            raise ValueError(
                f"record does not contain the response template {template!r}")
        return text
    question = record.get("question", "")
    if adapter_name == "requires_scheduler":
        label = record.get("requires_scheduler",
                           record.get("classification", record.get("label")))
    elif adapter_name == "modification_type":
        label = record.get("modification_type",
                           record.get("type", record.get("label")))
    if label is None:
        raise ValueError(f"missing label for {adapter_name}: {record.keys()}")
    return f"### Request:\n{question}\n{template}{label}"


def load_adapter_dataset(adapter_name: str, cfg: TrainConfig,
                         rng: np.random.Generator):
    path = _find_data_file(adapter_name, cfg.data_dir)
    print(f"[{adapter_name}] loading {path}")
    with open(path, encoding="utf-8") as f:
        records = _unwrap_records(json.load(f))
    texts = [_record_text(r, adapter_name) for r in records]
    idx = rng.permutation(len(texts))
    n_val = min(cfg.max_val_samples, max(1, int(len(texts) * cfg.val_split_ratio)))
    val_idx = set(idx[:n_val].tolist())
    train = [texts[i] for i in range(len(texts)) if i not in val_idx]
    val = [texts[i] for i in sorted(val_idx)]
    print(f"[{adapter_name}] train={len(train)} val={len(val)}")
    return Dataset.from_dict({"text": train}), Dataset.from_dict({"text": val})


class CompletionMaskingCollator:
    def __init__(self, tokenizer, response_template: str, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_ids = tokenizer.encode(
            response_template, add_special_tokens=False)
        if not self.template_ids:
            raise ValueError("empty response template after tokenization")

    def _find_template_end(self, ids: List[int]) -> Optional[int]:
        t = self.template_ids
        for start in range(len(ids) - len(t), -1, -1): # last token
            if ids[start:start + len(t)] == t:
                return start + len(t)
        return None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]
        batch = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for b, text in enumerate(texts):
            ids = batch["input_ids"][b].tolist()
            end = self._find_template_end(ids)
            if end is None:
                labels[b, :] = -100
            else:
                labels[b, :end] = -100
        batch["labels"] = labels
        return batch

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_masked_token_accuracy(eval_prediction):
    predictions, labels = eval_prediction
    correct = total = 0
    for pred, label in zip(predictions, labels):
        shifted_pred = pred[:-1]
        shifted_label = label[1:]
        valid = np.flatnonzero(shifted_label != -100)
        for pos in valid:
            total += 1
            correct += int(shifted_pred[pos] == shifted_label[pos])
    return {"accuracy": correct / total if total else 0.0}


def setup_model_and_tokenizer(cfg: TrainConfig, hf_token: Optional[str]):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, trust_remote_code=True, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quant_config,
        device_map="auto",
        token=hf_token,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def train_adapter(model, adapter_name: str, tokenizer, train_ds, val_ds,
                  cfg: TrainConfig, is_first_adapter: bool,
                  log_to_wandb: bool):
    spec = ADAPTERS[adapter_name]
    max_length = (cfg.max_seq_len_code if spec["max_len"] == "code"
                  else cfg.max_seq_len_cls)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    if is_first_adapter and not isinstance(model, PeftModel):
        model = get_peft_model(model, lora_config, adapter_name=adapter_name)
    else:
        model.add_adapter(adapter_name, lora_config)
    model.set_adapter(adapter_name)
    model.print_trainable_parameters()

    collator = CompletionMaskingCollator(
        tokenizer, spec["response_template"], max_length)

    args = TrainingArguments(
        output_dir=os.path.join(cfg.output_dir, f"_runs_{adapter_name}"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        optim=cfg.optimizer,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        fp16=True,
        max_grad_norm=cfg.max_grad_norm,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        group_by_length=False,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=["wandb"] if log_to_wandb else [],
        run_name=f"mars-{adapter_name}",
        seed=cfg.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_masked_token_accuracy,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()

    save_dir = os.path.join(cfg.output_dir, adapter_name)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, selected_adapters=[adapter_name])
    print(f"[{adapter_name}] adapter saved to {save_dir}")
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    add_auth_args(parser)
    parser.add_argument("--base-model", default=TrainConfig.base_model)
    parser.add_argument("--data-dir", default=TrainConfig.data_dir)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--adapters", nargs="+",
                        default=list(ADAPTERS.keys()),
                        choices=list(ADAPTERS.keys()))
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float,
                        default=TrainConfig.learning_rate)
    parser.add_argument("--gradient-accumulation-steps", type=int,
                        default=TrainConfig.gradient_accumulation_steps)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    apply_auth_args_to_env(args)
    hf_token = resolve_hf_token(args.hf_token, required=False)
    wandb_key = resolve_wandb_api_key(getattr(args, "wandb_api_key", None))
    if args.wandb and wandb_key:
        import wandb
        wandb.login(key=wandb_key)

    cfg = TrainConfig(
        base_model=args.base_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    model, tokenizer = setup_model_and_tokenizer(cfg, hf_token)
    for i, adapter_name in enumerate(args.adapters):
        train_ds, val_ds = load_adapter_dataset(adapter_name, cfg, rng)
        model = train_adapter(
            model, adapter_name, tokenizer, train_ds, val_ds, cfg,
            is_first_adapter=(i == 0), log_to_wandb=args.wandb,
        )
    print("done")


if __name__ == "__main__":
    main()
