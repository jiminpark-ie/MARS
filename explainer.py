from sim.repair_fixed import *

import os
import re
import random
import numpy as np
import json
from typing import Any, Dict, List, Optional, Set

from llm_utils import (
    EXPLAINER_MODEL,
    chat_completion,
    generate_paraphrased_question,
)


def _inject_precedence_dag(precedence, n_j, n_edges):
    if n_edges <= 0:
        return 0
    order = np.random.permutation(n_j)
    pos = np.empty(n_j, dtype=int)
    pos[order] = np.arange(n_j)
    added = attempts = 0
    while added < n_edges and attempts < n_edges * 40:
        attempts += 1
        a, b = int(np.random.randint(n_j)), int(np.random.randint(n_j))
        if a == b:
            continue
        i, j = (a, b) if pos[a] < pos[b] else (b, a)
        if precedence[i, j]:
            continue
        precedence[i, j] = 1
        added += 1
    return added


def generate_random_instance(n_m=10, n_j=100, n_precedence_edges=0):
    env_params = {
        'n_m': n_m,
        'n_j': n_j,
    }

    data = {
        'env_params': env_params,
        'processing_time': np.random.randint(2, 10, size=(env_params['n_m'], env_params['n_j'])),
        'due': np.random.randint(5, 30, size=env_params['n_j']),
        'ready': np.zeros(env_params['n_j']),
        'setup_time': np.random.randint(0, 5, size=(env_params['n_m'], env_params['n_j'], env_params['n_j'])),
        'precedence': np.zeros((env_params['n_j'], env_params['n_j'])),
        'machine_eligibility': np.ones((env_params['n_m'], env_params['n_j'])),
        'priority': np.zeros(env_params['n_j']),
    }

    for i in range(env_params['n_m']):
        for j in range(env_params['n_j']):
            data['setup_time'][i][j][j] = 0

    _inject_precedence_dag(data['precedence'], env_params['n_j'], n_precedence_edges)

    return data


def run_scheduler(data, rule='spt'):
    return run_scheduler_on_data(data, rule=rule)


EXPLAINER_SYSTEM_PROMPT = """You are a scheduling expert who explains schedule modifications to shop-floor operators.

Given:
- The user's modification request
- A causal chain dictionary showing how the modification propagated
- Schedule metrics before and after the modification

Generate a clear, structured explanation with:
1. SUMMARY: One sentence summarizing what was requested and done.
2. DIRECT CHANGES: What happened to the directly affected jobs (machine, position, timing changes).
3. CASCADING EFFECTS: How other jobs were affected, following the causal chain. 
   For each affected job, state: which job, what changed, and why (name the upstream job and the machine it is on when the cause is on another machine).
4. IMPACT: How key metrics (makespan, tardiness) changed.

Rules:
- Use concrete job IDs and machine IDs.
- State actual time values (start, end) when available.
- Keep the language accessible to non-programmers.
- Do not hallucinate information not present in the input.
- Be concise: aim for 100-200 words total."""


def call_gpt_explainer(modification_info, causal_chain_dict, metrics_before, metrics_after,
                       model: str = EXPLAINER_MODEL, temperature: float = 0.7):
    user_prompt = f"""User request: {modification_info['user_query']}
Modification type: {modification_info['modification_type']}
Directly affected jobs: {modification_info['direct_job_ids']}

Causal chain: {json.dumps(causal_chain_dict, default=str)}

Metrics before: {json.dumps(metrics_before, default=str)}
Metrics after: {json.dumps(metrics_after, default=str)}"""

    return chat_completion(
        EXPLAINER_SYSTEM_PROMPT,
        user_prompt,
        model=model,
        temperature=temperature,
        max_tokens=1280,
    )


MODIFICATION_TYPES = [
    'precedence', 'processing-time-change', 'setup-time-change', 'due-date-change',
    'ready-time-change', 'priority-change', 'fixed-machine', 'machine-restricted',
    'job-swap-same-machine', 'job-swap-different-machines',
    'job-insert-same-machine', 'job-insert-different-machines',
]


def _maybe_paraphrase(question: str, paraphrase: bool) -> str:
    return generate_paraphrased_question(question) if paraphrase else question


def generate_random_modification(data, schedule, modification_type, paraphrase=True):
    env_params = data['env_params']
    modification_info = {"modification_type": modification_type}

    if modification_type == 'precedence':
        machine_of = {int(j): m for m, seq in schedule.items() for j in seq}
        i = random.choice(range(env_params['n_j']))
        cross = [j for j in range(env_params['n_j'])
                 if j != i and machine_of.get(j) is not None
                 and machine_of.get(j) != machine_of.get(i)]
        j = random.choice(cross) if cross else random.choice(
            [k for k in range(env_params['n_j']) if k != i])
        modification_info["user_query"] = _maybe_paraphrase(
            f"What if job {i} must be completed before job {j} starts?", paraphrase)
        modification_info["direct_job_ids"] = [i, j]

    elif modification_type == 'processing-time-change':
        m = random.choice(range(env_params['n_m']))
        i = random.choice(schedule[m])
        t = random.randrange(1, 100)
        modification_info["user_query"] = _maybe_paraphrase(
            f"What if the processing time of job {i} on machine {m} is changed to {t}?", paraphrase)
        modification_info["direct_job_ids"] = [i]
        modification_info["machine_id"] = m
        modification_info["new_value"] = t

    elif modification_type == 'setup-time-change':
        m = random.choice(range(env_params['n_m']))
        i, j = random.sample(list(schedule[m]), 2)
        t = random.randrange(1, 100)
        modification_info["user_query"] = _maybe_paraphrase(
            f"What if the setup time on machine {m} from job {i} to job {j} is changed to {t}?", paraphrase)
        modification_info["direct_job_ids"] = [i, j]
        modification_info["machine_id"] = m
        modification_info["from_job"] = i
        modification_info["to_job"] = j
        modification_info["new_value"] = t

    elif modification_type == 'due-date-change':
        i = random.choice(range(env_params['n_j']))
        t = random.randrange(1, 100)
        modification_info["user_query"] = _maybe_paraphrase(
            f"What if the due date of job {i} is changed to {t}?", paraphrase)
        modification_info["direct_job_ids"] = [i]
        modification_info["new_value"] = t

    elif modification_type == 'ready-time-change':
        i = random.choice(range(env_params['n_j']))
        t = random.randrange(1, 100)
        query = _maybe_paraphrase(
            f"What if the ready time of job {i} is changed to {t}?", paraphrase)
        attempts = 0
        while 'start time' in query and attempts < 3:
            query = _maybe_paraphrase(
                f"What if the ready time of job {i} is changed to {t}?", paraphrase)
            attempts += 1
        modification_info["user_query"] = query
        modification_info["direct_job_ids"] = [i]
        modification_info["new_value"] = t

    elif modification_type == 'priority-change':
        i = random.choice(range(env_params['n_j']))
        modification_info["user_query"] = _maybe_paraphrase(
            f"What if job {i} should be processed with the highest priority?", paraphrase)
        modification_info["direct_job_ids"] = [i]

    elif modification_type == 'fixed-machine':
        i = random.choice(range(env_params['n_j']))
        m = random.choice(range(env_params['n_m']))
        modification_info["user_query"] = _maybe_paraphrase(
            f"What if job {i} can be processed only on machine {m}?", paraphrase)
        modification_info["direct_job_ids"] = [i]
        modification_info["target_machine"] = m

    elif modification_type == 'machine-restricted':
        m = random.choice(range(env_params['n_m']))
        modification_info["user_query"] = _maybe_paraphrase(
            f"What if machine {m} breaks down and becomes unusable?", paraphrase)
        modification_info["direct_job_ids"] = list(schedule[m])
        modification_info["machine_id"] = m

    elif modification_type == 'job-swap-same-machine':
        m = random.choice(range(env_params['n_m']))
        i, j = random.sample(list(schedule[m]), 2)
        modification_info["user_query"] = _maybe_paraphrase(
            f"What if the positions of job {i} and job {j} on machine {m} are swapped?", paraphrase)
        modification_info["direct_job_ids"] = [i, j]
        modification_info["machine_id"] = m

    elif modification_type == 'job-swap-different-machines':
        m1, m2 = random.sample(range(env_params['n_m']), 2)
        i = random.choice(schedule[m1])
        j = random.choice(schedule[m2])
        modification_info["user_query"] = _maybe_paraphrase(
            f"What if job {i} on machine {m1} is swapped with job {j} on machine {m2}?", paraphrase)
        modification_info["direct_job_ids"] = [i, j]
        modification_info["source_machine"] = m1
        modification_info["dest_machine"] = m2

    elif modification_type == 'job-insert-same-machine':
        m = random.choice(range(env_params['n_m']))
        i, j = random.sample(list(schedule[m]), 2)
        place_before = random.random() < 0.5
        pos_word = "before" if place_before else "after"
        modification_info["user_query"] = _maybe_paraphrase(
            f"What if job {i} on machine {m} is moved to immediately {pos_word} job {j} on the same machine?",
            paraphrase)
        modification_info["direct_job_ids"] = [i, j]
        modification_info["machine_id"] = m
        modification_info["place_before"] = place_before

    elif modification_type == 'job-insert-different-machines':
        m1, m2 = random.sample(range(env_params['n_m']), 2)
        i = random.choice(schedule[m1])
        j = random.choice(schedule[m2])
        place_before = random.random() < 0.5
        pos_word = "before" if place_before else "after"
        modification_info["user_query"] = _maybe_paraphrase(
            f"What if job {i} on machine {m1} is moved to just {pos_word} job {j} on machine {m2}?",
            paraphrase)
        modification_info["direct_job_ids"] = [i, j]
        modification_info["source_machine"] = m1
        modification_info["dest_machine"] = m2
        modification_info["place_before"] = place_before

    return modification_info


AUDITABLE_TYPES = [t for t in MODIFICATION_TYPES
                   if t not in ('fixed-machine', 'machine-restricted')]


def greedy_schedule(data):
    n_m, n_j = data['processing_time'].shape
    load = [0.0] * n_m
    sched = {m: [] for m in range(n_m)}
    for j in range(n_j):
        elig = [m for m in range(n_m) if data['machine_eligibility'][m, j]] or list(range(n_m))
        m = min(elig, key=lambda mm: load[mm] + float(data['processing_time'][mm, j]))
        sched[m].append(j)
        load[m] += float(data['processing_time'][m, j])
    return sched


def build_case(modification_type, n_precedence_edges=0, scheduler=greedy_schedule,
               n_m=10, n_j=100, paraphrase=False):
    data0 = generate_random_instance(n_m=n_m, n_j=n_j, n_precedence_edges=n_precedence_edges)
    s0 = scheduler(data0)
    data0 = {**data0, 'schedule': s0}
    mod_info = generate_random_modification(data0, s0, modification_type,
                                            paraphrase=paraphrase)
    data_after, s_prime = apply_modification(data0, s0, mod_info)
    return {
        "data0": data0, "s0": s0, "s_prime": s_prime, "data_after": data_after,
        "type": modification_type,
        "rerun": 1 if modification_type in ('fixed-machine', 'machine-restricted') else 0,
        "nl": mod_info["user_query"], "modification_info": mod_info,
        "direct_job_ids": mod_info["direct_job_ids"],
    }


def generate_audit_cases(n_per_type=10, n_precedence_edges=0, types=None,
                         scheduler=greedy_schedule, seed=0, paraphrase=False):
    random.seed(seed)
    np.random.seed(seed)
    for t in (types or AUDITABLE_TYPES):
        for _ in range(n_per_type):
            try:
                yield build_case(t, n_precedence_edges=n_precedence_edges,
                                 scheduler=scheduler, paraphrase=paraphrase)
            except Exception as e:
                print(f"  [build_case:{t}] skipped: {e}")


def generate_explainer_examples(n_examples_per_type=5, verbose=True,
                                n_precedence_edges=40, max_affected_jobs=40, max_depth=10,
                                paraphrase=True):
    examples = []
    failed = 0
    cascades = 0

    for mod_type in MODIFICATION_TYPES:
        for i in range(n_examples_per_type):
            try:
                data = generate_random_instance(n_m=10, n_j=100,
                                                n_precedence_edges=n_precedence_edges)
                schedule = run_scheduler(data)
                modification_info = generate_random_modification(
                    data, schedule, mod_type, paraphrase=paraphrase)

                new_schedule, causal_chain, metrics_before, metrics_after = \
                    repair_schedule_with_trace(data, schedule, modification_info)

                causal_chain_dict = causal_chain.to_dict()
                compressed_chain = compress_causal_chain(
                    causal_chain_dict, max_affected_jobs=max_affected_jobs, max_depth=max_depth
                )

                explanation = call_gpt_explainer(
                    modification_info, compressed_chain, metrics_before, metrics_after
                )

                examples.append({
                    "user_query": modification_info["user_query"],
                    "modification_type": modification_info["modification_type"],
                    "directly_affected_jobs": modification_info["direct_job_ids"],
                    "causal_chain": compressed_chain,
                    "metrics_before": {k: v for k, v in metrics_before.items()
                                       if k != "machine_utilization"},
                    "metrics_after": {k: v for k, v in metrics_after.items()
                                      if k != "machine_utilization"},
                    "explanation": explanation,
                    "causal_depth": causal_chain.max_depth,
                })

                if causal_chain.total_affected_jobs > len(modification_info["direct_job_ids"]):
                    cascades += 1

                if verbose:
                    print(f"  [{mod_type}] Example {i+1}/{n_examples_per_type} "
                          f"(affected: {causal_chain.total_affected_jobs}, "
                          f"depth: {causal_chain.max_depth})")

            except Exception as e:
                failed += 1
                if verbose:
                    print(f"  [{mod_type}] Example {i+1} FAILED: {e}")

    if verbose:
        print(f"\nGenerated {len(examples)} examples, {failed} failed, "
              f"{cascades} with cascading (affected > directly-modified).")

    return examples



def format_explainer_input(modification_info, causal_chain, metrics_before, metrics_after):
    m_before = {k: v for k, v in metrics_before.items() if k != "machine_utilization"}
    m_after = {k: v for k, v in metrics_after.items() if k != "machine_utilization"}

    return (
        f"User request: {modification_info.get('user_query', '')}\n"
        f"Modification type: {modification_info.get('modification_type', '')}\n"
        f"Directly affected jobs: {modification_info.get('direct_job_ids', [])}\n"
        f"Causal chain: {json.dumps(causal_chain, default=str)}\n"
        f"Metrics before: {json.dumps(m_before, default=str)}\n"
        f"Metrics after: {json.dumps(m_after, default=str)}"
    )


def select_examples(mod_type, example_pool, n_shots=1):
    matched = [ex for ex in example_pool if ex['modification_type'] == mod_type]
    if len(matched) < n_shots:
        matched = example_pool
    return random.sample(matched, min(n_shots, len(matched)))


def build_explainer_prompt(test_input, few_shot_examples, n_shots=1):
    system = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a scheduling expert. Given a user's modification request and the resulting causal chain of schedule changes, explain the cascading effects clearly and concisely.<|eot_id|>"
    )

    shots = ""
    for ex in few_shot_examples[:n_shots]:
        ex_input = format_explainer_input(
            {"user_query": ex["user_query"],
             "modification_type": ex["modification_type"],
             "direct_job_ids": ex["directly_affected_jobs"]},
            ex["causal_chain"],
            ex["metrics_before"],
            ex["metrics_after"],
        )
        shots += (
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{ex_input}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
            f"{ex['explanation']}<|eot_id|>"
        )

    query = (
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{test_input}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

    return system + shots + query


def generate_test_instances(examples, n_instances=1, n_shots=1,
                            n_precedence_edges=40, max_affected_jobs=40, max_depth=10,
                            paraphrase=True):
    prompts = []
    references = []

    for i in range(n_instances):
        mod_type = random.choice(MODIFICATION_TYPES)
        data = generate_random_instance(n_m=10, n_j=100,
                                        n_precedence_edges=n_precedence_edges)
        schedule = run_scheduler(data)
        modification_info = generate_random_modification(
            data, schedule, mod_type, paraphrase=paraphrase)

        new_schedule, causal_chain, metrics_before, metrics_after = \
            repair_schedule_with_trace(data, schedule, modification_info)

        causal_chain_dict = causal_chain.to_dict()
        compressed_chain = compress_causal_chain(
            causal_chain_dict, max_affected_jobs=max_affected_jobs, max_depth=max_depth
        )

        test_input = format_explainer_input(
            modification_info, compressed_chain, metrics_before, metrics_after
        )

        selected = select_examples(mod_type, examples, n_shots)
        prompt = build_explainer_prompt(test_input, selected, n_shots)

        prompts.append(prompt)
        references.append({
            "modification_info": modification_info,
            "causal_chain": compressed_chain,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
        })

    return prompts, references


def _placement(schedule: Dict[int, List[int]]) -> Dict[int, Dict[str, int]]:
    out: Dict[int, Dict[str, int]] = {}
    for m, seq in schedule.items():
        for pos, j in enumerate(seq):
            out[int(j)] = {"machine": int(m), "position": int(pos)}
    return out


def _starts(data: Dict[str, Any], schedule: Dict[int, List[int]]) -> Dict[int, float]:
    seq = normalize_schedule_to_sequence(schedule)
    timed = calculate_timing_from_sequence(data, seq)
    return {int(j): float(s) for m, jobs in timed.items() for (j, s, _e) in jobs}


def change_set(
    data0: Dict[str, Any],
    s0: Dict[int, List[int]],
    s_prime: Dict[int, List[int]],
    data_after: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pa, pb = _placement(s0), _placement(s_prime)
    sa = _starts(data0, s0)
    sb = _starts(data_after or data0, s_prime)
    jobs = set(pa) | set(pb)
    affected: List[int] = []
    details: Dict[int, Dict[str, Any]] = {}
    for j in sorted(jobs):
        before = {"machine": pa.get(j, {}).get("machine"),
                  "position": pa.get(j, {}).get("position"),
                  "start": sa.get(j)}
        after = {"machine": pb.get(j, {}).get("machine"),
                 "position": pb.get(j, {}).get("position"),
                 "start": sb.get(j)}
        kinds = []
        if before["machine"] != after["machine"]:
            kinds.append("machine")
        if before["position"] != after["position"]:
            kinds.append("position")
        if before["start"] is not None and after["start"] is not None \
                and abs(before["start"] - after["start"]) > 1e-6:
            kinds.append("timing")
        if kinds:
            affected.append(j)
            details[j] = {"from": before, "to": after, "kind": kinds}
    return {"affected_jobs": affected, "details": details}


def build_causal_chain(query: str, cs: Dict[str, Any],
                       modification_type: str = "", rerun: int = 0) -> Dict[str, Any]:
    direct = []
    for j in cs["affected_jobs"]:
        d = cs["details"][j]
        direct.append({"job": j, "change": d["kind"],
                       "from_machine": d["from"]["machine"], "to_machine": d["to"]["machine"],
                       "from_start": d["from"]["start"], "to_start": d["to"]["start"]})
    return {
        "root_cause": query,
        "modification_type": modification_type,
        "modification_method": "rerun" if rerun else "repair",
        "total_affected_jobs": len(cs["affected_jobs"]),
        "direct_changes": direct,
    }


_JOB_RE = re.compile(r"\bjobs?\s+#?(\d+)", re.IGNORECASE)


def jobs_mentioned(text: str) -> Set[int]:
    return {int(m) for m in _JOB_RE.findall(text or "")}


def score_faithfulness(explanation: str, cs: Dict[str, Any]) -> Dict[str, float]:
    mentioned = jobs_mentioned(explanation)
    affected = set(cs["affected_jobs"])
    inter = mentioned & affected
    precision = (len(inter) / len(mentioned)) if mentioned else float("nan")
    recall = (len(inter) / len(affected)) if affected else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (mentioned and affected and (precision + recall) > 0) else float("nan"))
    hallucinated = sorted(mentioned - affected)
    return {"precision": precision, "recall": recall, "f1": f1,
            "n_mentioned": len(mentioned), "n_affected": len(affected),
            "n_hallucinated": len(hallucinated)}


def aggregate_faithfulness(per_case: List[Dict[str, float]]) -> Dict[str, Any]:
    inter = sum(max(0, c["n_mentioned"] - c["n_hallucinated"]) for c in per_case)
    mentioned = sum(c["n_mentioned"] for c in per_case)
    affected = sum(c["n_affected"] for c in per_case)
    micro_p = inter / mentioned if mentioned else float("nan")
    micro_r = inter / affected if affected else float("nan")
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if mentioned and affected and (micro_p + micro_r) > 0 else float("nan"))

    def _nanmean(key):
        vals = [c[key] for c in per_case
                if not (isinstance(c[key], float) and np.isnan(c[key]))]
        return (sum(vals) / len(vals) if vals else float("nan")), len(vals)

    macro_p, n_p = _nanmean("precision")
    macro_r, n_r = _nanmean("recall")
    macro_f1, n_f1 = _nanmean("f1")
    return {
        "n_cases": len(per_case),
        "micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1,
        "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1,
        "n_cases_precision_defined": n_p,
        "n_cases_recall_defined": n_r,
        "n_cases_f1_defined": n_f1,
        "total_hallucinated": sum(c["n_hallucinated"] for c in per_case),
    }


def _render_schedule(schedule: Dict[int, List[int]], limit: int = 40) -> str:
    rows = []
    for m in sorted(schedule, key=lambda x: int(x)):
        seq = list(schedule[m])
        s = ", ".join(map(str, seq[:limit])) + (" ..." if len(seq) > limit else "")
        rows.append(f"Machine {m}: [{s}]")
    return "\n".join(rows)


class OpenAIExplainer:
    def __init__(self, model: str = EXPLAINER_MODEL, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def explain(self, query, s0, s_prime, causal_chain=None, data_after=None,
                data0=None):
        prompt = (
            "You are assisting a production engineer. Given a schedule modification request and the original/modified schedules, describe in clear natural language how the schedule changed. Reference jobs by their id. Only describe changes that actually occurred; do not invent changes.\n\n"
            f"Request:\n{query}\n\n"
            f"Original schedule (job sequence per machine):\n{_render_schedule(s0)}\n\n"
            f"Modified schedule:\n{_render_schedule(s_prime)}\n\n"
        )
        if causal_chain:
            prompt += ("Verified changes (use these as ground truth):\n"
                       f"{causal_chain.get('direct_changes')}\n\n")
        prompt += "Explanation:"
        from llm_utils import get_openai_client
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=self.model, temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()


def main():
    import argparse
    from config import add_auth_args, apply_auth_args_to_env, resolve_hf_token

    parser = argparse.ArgumentParser()
    add_auth_args(parser)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n-examples-per-type", type=int, default=5)
    parser.add_argument("--n-instances", type=int, default=5)
    parser.add_argument("--n-shots", type=int, default=1)
    args = parser.parse_args()
    apply_auth_args_to_env(args)
    hf_token = resolve_hf_token(args.hf_token, required=False)

    print("=== Generating explainer examples ===")
    examples = generate_explainer_examples(n_examples_per_type=args.n_examples_per_type)

    print(f"\n=== Loading model ===")
    from huggingface_hub import login
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if hf_token:
        login(token=hf_token)

    model_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=hf_token,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map='auto', token=hf_token,
    )

    print(f"\n=== Generating test prompts ===")
    prompts, references = generate_test_instances(
        examples,
        n_instances=args.n_instances,
        n_shots=args.n_shots,
    )

    print(f"\n=== Running inference ===")
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(model.device)
        input_length = inputs['input_ids'].shape[1]

        out = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated_tokens = out[0][input_length:]
        reply = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        mod_type = references[i]["modification_info"]["modification_type"]
        print(f"\n--- Test {i+1} ({mod_type}) ---")
        print(f"Query: {references[i]['modification_info']['user_query']}")
        print(f"Affected jobs: {references[i]['causal_chain']['total_affected_jobs']}")
        print(f"Reply:\n{reply}")
        print()


if __name__ == "__main__":
    main()
