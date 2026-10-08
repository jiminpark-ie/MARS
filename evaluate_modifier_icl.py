import os
import re
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import time

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from sacrebleu.metrics import CHRF
from mars_adapters import TYPE_TO_LEVEL, Level
from config import add_auth_args, apply_auth_args_to_env, resolve_hf_token

SYSTEM_PROMPT = """You are analyzing job scheduling problem data. The data contains information about jobs, machines, and scheduling constraints.

Environment Parameters:
- n_j: Number of jobs (integer)
- n_m: Number of machines (integer)

All arrays are provided in flattened format but represent the following original structures:

1. ProcessingTime: 2D array [n_m, n_j]
   - data['processing_time'][machine_id][job_id] = processing time for job on machine

2. MachineEligibility: 2D array [n_m, n_j]
   - data['machine_eligibility'][machine_id][job_id] = 1 if machine can process job, 0 otherwise

3. SetupTimes: 3D array [n_m, n_j, n_j]
   - data['setup_time'][machine_id][from_job][to_job] = setup time when switching jobs

4. Precedences: 2D array [n_j, n_j]
   - data['precedence'][job_i][job_j] = 1 if job_i must complete before job_j starts

5. ReadyTimes: 1D array [n_j]
   - data['ready'][job_id] = earliest time when job can start

6. DueDates: 1D array [n_j]
   - data['due'][job_id] = deadline for job completion

7. Priorities: 1D array [n_j]
   - data['priority'][job_id] = priority weight for job

"""

def load_test_data(data_path: str) -> pd.DataFrame:
    with open(data_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    items = data if isinstance(data, list) else data.get("questions", data.get("data", [data]))
    records = [
        {
            "question": item.get("question", ""),
            "label_modification_code": item.get("code", ""),
            "type": item.get("type", item.get("modification_type", "")),
        }
        for item in items
    ]
    return pd.DataFrame(records)

def load_model(hf_token: Optional[str] = None, base_model: Optional[str] = None) -> Tuple:
    token = hf_token or os.environ.get("HF_TOKEN")
    model_name = base_model or os.environ.get("MARS_BASE_MODEL", "meta-llama/Llama-3.1-8B")

    if token:
        login(token=token)
    
    print(f"Loading base model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=token,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        token=token
    )
    
    return tokenizer, model

def generate_prompt(
    run_type: str,
    question: str,
    icl_examples: Optional[List[Dict]] = None,
) -> str:
    
    prompt = ""

    if run_type == 'Vanilla':
        prompt = f"{SYSTEM_PROMPT}\n\n"
    
    elif icl_examples:
        for ex in icl_examples:
            ex_question = ex.get("QUESTION", ex.get("question", ""))
            ex_answer = str(ex["code"])
            prompt += f"### Request:\n{ex_question}\n### Code:\n{ex_answer}\n\n"
    prompt += f"### Request:\n{question}\n### Code:\n"
    return prompt

def run_inference(
    model,
    tokenizer,
    prompts: List[str],
) -> List[str]:
    
    generated = []
    
    for prompt in tqdm(prompts, desc=f"Inference (Modification type)"):
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
        ).to(model.device)
        
        input_length = inputs['input_ids'].shape[1]
        
        gen_kwargs = {
            "max_new_tokens": 1280,
            "do_sample": True,
            "temperature": 0.7,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        generated_tokens = out[0][input_length:]
        reply = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        stop_keywords = ["### User", "###", "User:", "Assistant:", "```"]
        min_idx = len(reply)
        for kw in stop_keywords:
            idx = reply.find(kw)
            if idx != -1:
                min_idx = min(min_idx, idx)
        reply = reply[:min_idx].strip()

        generated.append(reply)
    
    return generated

def compute_chrf(predictions: List[str], labels: List[str]) -> float:
    scores = []
    for pred, label in zip(predictions, labels):
        if not pred.strip() or not label.strip():
            continue
        score = CHRF(word_order=2).sentence_score(pred, [label]).score
        scores.append(score)
    return np.mean(scores) if scores else 0.0


def evaluate_predictions(
    df: pd.DataFrame,
    adapter_name: str,
    pred_column: str,
    with_codebert: bool = True,
) -> Dict[str, float]:

    label_column = f"label_{adapter_name}"

    predictions = df[pred_column].tolist()
    labels = df[label_column].tolist()

    metrics = {}

    valid_pairs = [(p, l) for p, l in zip(predictions, labels) if l]
    if not valid_pairs:
        return {"error": "No valid labels found"}

    valid_preds, valid_labels = zip(*valid_pairs)

    metrics["chrf"] = compute_chrf(list(valid_preds), list(valid_labels))
    if with_codebert:
        from mr_metrics import codebert
        metrics.update(codebert(list(valid_preds), list(valid_labels)))

    return metrics


# Main Pipeline

def run_evaluation_pipeline(
    test_data_path: str,
    output_dir: str = "./results",
    icl_examples_path: Optional[str] = None,
    num_runs: int = 1,
    hf_token: Optional[str] = None,
    base_model: Optional[str] = None,
    icl_job_path: str = "./icl_job.json",
    icl_machine_path: str = "./icl_machine_aug.json",
    icl_schedule_path: str = "./icl_schedule.json",
    with_codebert: bool = True,
):
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load test data
    df = load_test_data(test_data_path)
    
    icl_examples = None
    if icl_examples_path and os.path.exists(icl_examples_path):
        with open(icl_examples_path, "r", encoding='utf-8') as f:
            icl_examples = json.load(f)
        print(f"Loaded {len(icl_examples)} ICL examples")

    with open(icl_job_path, "r", encoding='utf-8') as f:
        icl_job_examples = json.load(f)
    with open(icl_machine_path, "r", encoding='utf-8') as f:
        icl_machine_examples = json.load(f)
    with open(icl_schedule_path, "r", encoding='utf-8') as f:
        icl_schedule_examples = json.load(f)
    pools = {
        Level.JOB: icl_job_examples,
        Level.MACHINE: icl_machine_examples,
        Level.SCHEDULE: icl_schedule_examples,
    }

    tokenizer, model = load_model(hf_token=hf_token, base_model=base_model)
    
    all_results = []
    
    for run_idx in range(1, num_runs + 1):
        print(f"\n{'='*60}")
        print(f"Run {run_idx}/{num_runs}")
        print('='*60)
        
        run_df = df.copy()
        
        prompts = []
        for row in df.itertuples(index=False):
            key = str(row.type).strip().lower().replace("-", "_")
            level = TYPE_TO_LEVEL.get(key)
            examples = pools.get(level)
            if examples is None:
                examples = icl_examples or (
                    icl_job_examples + icl_machine_examples + icl_schedule_examples
                )
            prompts.append(generate_prompt('ICL', row.question, examples))

        run_df['modification_code_prompt'] = prompts
        
        start = time.time()
        predictions = run_inference(model, tokenizer, prompts)

        inference_time = time.time() - start

        run_df['modification_code_pred'] = predictions
        
        metrics = evaluate_predictions(run_df, 'modification_code', 'modification_code_pred',
                                       with_codebert=with_codebert)
        
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value}")
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                all_results.append({
                    "run": run_idx,
                    "metric": metric_name,
                    "value": value,
                })

        run_df.to_csv(
            os.path.join(
                output_dir,
                f"ood_predictions_run{run_idx}_{timestamp}_{inference_time:.2f}s.csv",
            ),
            index=False
        )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        os.path.join(output_dir, f"metrics_{timestamp}.csv"),
        index=False
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if not results_df.empty:
        summary = results_df.groupby(['metric'])['value'].agg(['mean', 'std'])
        print(summary)
    
    return results_df

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Modification Code Generator Evaluation")
    add_auth_args(parser)
    parser.add_argument("--test_data", type=str, default="./test.json")
    parser.add_argument("--icl_examples", type=str, default=None)
    parser.add_argument("--icl-job", type=str, default="./icl_job.json")
    parser.add_argument("--icl-machine", type=str, default="./icl_machine_aug.json")
    parser.add_argument("--icl-schedule", type=str, default="./icl_schedule.json")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--no-codebert", action="store_true",
                        help="skip CodeBERTScore (no model download)")

    args = parser.parse_args()
    apply_auth_args_to_env(args)
    hf_token = resolve_hf_token(args.hf_token, required=False)

    run_evaluation_pipeline(
        args.test_data,
        args.output_dir,
        args.icl_examples,
        args.num_runs,
        hf_token=hf_token,
        base_model=args.base_model,
        icl_job_path=args.icl_job,
        icl_machine_path=args.icl_machine,
        icl_schedule_path=args.icl_schedule,
        with_codebert=not args.no_codebert,
    )

    print("\nDone!")
