import hashlib
import random
import json
import argparse
import os
from typing import Dict, List

from llm_utils import generate_paraphrased_question
from config import add_auth_args, apply_auth_args_to_env

import concurrent.futures
from tqdm import tqdm


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

def generate_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def format_for_classification(question: str, requires_scheduler: int) -> Dict[str, str]:
    text = f"""### Request:
{question}

### Classification:
{requires_scheduler}"""
    
    return {"text": text}

def format_for_modification_code(question: str, code: str) -> Dict[str, str]:
    text = f"""### Request:
{question}

### Code:
{code}"""
    
    return {"text": text}

def format_for_modification_type(question: str, modification_type: str) -> Dict[str, str]:
    text = f"""### Request:
{question}

### Type:
{modification_type}"""
    
    return {"text": text}



def generate_precedence_relationship(env_params: Dict) -> Dict:
    i, j = random.sample(range(env_params['n_j']), 2)
    
    question = f"What if job {i} must be completed before job {j} starts?"
    output_code = f"data['precedence'][{i}][{j}] = 1"
    
    return {
        "question": generate_paraphrased_question(question),
        "code": output_code,
        "type": 0,
        "requires_scheduler": 1,
        "raw_question_hash": generate_hash(question)
    }


def change_machine_processing_time(env_params: Dict) -> Dict:
    i = random.choice(range(env_params['n_j']))
    m = random.choice(range(env_params['n_m']))
    t = random.randrange(1, 100)
    
    question = f"What if the processing time of job {i} on machine {m} is changed to {t}?"
    output_code = f"data['processing_time'][{m}, {i}] = {t}"
    
    return {
        "question": generate_paraphrased_question(question),
        "code": output_code,
        "type": 0,
        "requires_scheduler": 1,
        "raw_question_hash": generate_hash(question)
    }


def change_setup_time(env_params: Dict) -> Dict:
    i, j = random.sample(range(env_params['n_j']), 2)
    m = random.choice(range(env_params['n_m']))
    t = random.randrange(1, 100)
    
    question = f"What if the setup time on machine {m} from job {i} to job {j} is changed to {t}?"
    output_code = f"data['setup_time'][{m}, {i}, {j}] = {t}"
    
    return {
        "question": generate_paraphrased_question(question),
        "code": output_code,
        "type": 0,
        "requires_scheduler": 1,
        "raw_question_hash": generate_hash(question)
    }


def change_due_date(env_params: Dict) -> Dict:
    i = random.choice(range(env_params['n_j']))
    t = random.randrange(1, 100)
    
    question = f"What if the due date of job {i} is changed to {t}?"
    output_code = f"data['due'][{i}] = {t}"
    
    return {
        "question": generate_paraphrased_question(question),
        "code": output_code,
        "type": 0,
        "requires_scheduler": 1,
        "raw_question_hash": generate_hash(question)
    }


def change_ready_time(env_params: Dict) -> Dict:
    i = random.choice(range(env_params['n_j']))
    t = random.randrange(1, 100)
    
    question = f"What if the ready time of job {i} is changed to {t}?"
    output_code = f"data['ready'][{i}] = {t}"
    
    paraphrased_question = generate_paraphrased_question(question)
    while 'start time' in paraphrased_question:
        paraphrased_question = generate_paraphrased_question(question)
    return {
        "question": paraphrased_question,
        "code": output_code,
        "type": 0,
        "requires_scheduler": 1,
        "raw_question_hash": generate_hash(question)
    }


def change_priority(env_params: Dict) -> Dict:
    i = random.choice(range(env_params['n_j']))
    
    question = f"What if job {i} should be processed with the highest priority?"
    output_code = f"data['priority'][{i}] = float('inf')"
    
    return {
        "question": generate_paraphrased_question(question),
        "code": output_code,
        "type": 0,
        "requires_scheduler": 1,
        "raw_question_hash": generate_hash(question)
    }


def generate_fixed_machine(env_params: Dict) -> Dict:
    i = random.randint(0, env_params['n_j'] - 1)
    m = random.randint(0, env_params['n_m'] - 1)
    
    question = f"What if job {i} can be processed only on machine {m}?"
    output_code = f"data['machine_eligibility'][:, {i}] = 0\ndata['machine_eligibility'][{m}, {i}] = 1"
    
    return {
        "question": generate_paraphrased_question(question),
        "code": output_code,
        "type": 1,
        "requires_scheduler": 1,
        "raw_question_hash": generate_hash(question)
    }


def machine_restricted(env_params: Dict) -> Dict:
    m = random.randint(0, env_params['n_m'] - 1)
    
    question = f"What if machine {m} breaks down and becomes unusable?"
    output_code = f"data['machine_eligibility'][{m}, :] = 0"
    
    return {
        "question": generate_paraphrased_question(question),
        "code": output_code,
        "type": 1,
        "requires_scheduler": 1,
        "raw_question_hash": generate_hash(question)
    }


def job_swap_same_machine(env_params: Dict) -> Dict:
    i, j = random.sample(range(env_params['n_j']), 2)
    m = random.choice(range(env_params['n_m']))

    question = f"What if the positions of job {i} and job {j} on machine {m} are swapped?"
    output_code = f"""schedule_m = data['schedule'][{m}]\n
i_idx = schedule_m.index({i})\n
j_idx = schedule_m.index({j})\n
schedule_m[i_idx], schedule_m[j_idx] = schedule_m[j_idx], schedule_m[i_idx]"""
    
    return {
        "question": generate_paraphrased_question(question),
        "code": output_code,
        "type": 2,
        "requires_scheduler": 0,
        "raw_question_hash": generate_hash(question)
    }


def job_swap_different_machines(env_params: Dict) -> Dict:
    i, j = random.sample(range(env_params['n_j']), 2)
    m1, m2 = random.sample(range(env_params['n_m']), 2)
    
    question = f"What if job {i} on machine {m1} is swapped with job {j} on machine {m2}?"
    output_code = f"""schedule_m1 = data['schedule'][{m1}]\n
schedule_m2 = data['schedule'][{m2}]\n
i_idx = schedule_m1.index({i})\n
j_idx = schedule_m2.index({j})\n
schedule_m1[i_idx], schedule_m2[j_idx] = schedule_m2[j_idx], schedule_m1[i_idx]"""
    
    return {
        "question": generate_paraphrased_question(question),
        "code": output_code,
        "type": 2,
        "requires_scheduler": 0,
        "raw_question_hash": generate_hash(question)
    }


def job_insert_same_machine(env_params: Dict) -> Dict:
    i, j = random.sample(range(env_params['n_j']), 2)
    m = random.choice(range(env_params['n_m']))
    place_before = random.random() < 0.5
    pos_word = "before" if place_before else "after"
    
    question = f"What if job {i} on machine {m} is moved to immediately {pos_word} job {j} on the same machine?"
    
    if place_before:
        output_code = f"""schedule_m = data['schedule'][{m}]\n
i_idx = schedule_m.index({i})\n
job = schedule_m.pop(i_idx)\n
j_idx = schedule_m.index({j})\n
schedule_m.insert(j_idx, job)"""
    else:
        output_code = f"""schedule_m = data['schedule'][{m}]\n
i_idx = schedule_m.index({i})\n
job = schedule_m.pop(i_idx)\n
j_idx = schedule_m.index({j})\n
schedule_m.insert(j_idx + 1, job)"""
    
    return {
        "question": generate_paraphrased_question(question),
        "code": output_code,
        "type": 2,
        "requires_scheduler": 0,
        "raw_question_hash": generate_hash(question)
    }


def job_insert_different_machine(env_params: Dict) -> Dict:
    i, j = random.sample(range(env_params['n_j']), 2)
    m1, m2 = random.sample(range(env_params['n_m']), 2)
    place_before = random.random() < 0.5
    pos_word = "before" if place_before else "after"
    
    question = f"What if job {i} on machine {m1} is moved to just {pos_word} job {j} on machine {m2}?"
    
    if place_before:
        output_code = f"""schedule_m1 = data['schedule'][{m1}]\n
schedule_m2 = data['schedule'][{m2}]\n
job = schedule_m1.pop(schedule_m1.index({i}))\n
j_idx = schedule_m2.index({j})\n
schedule_m2.insert(j_idx, job)"""
    else:
        output_code = f"""schedule_m1 = data['schedule'][{m1}]\n
schedule_m2 = data['schedule'][{m2}]\n
job = schedule_m1.pop(schedule_m1.index({i}))\n
j_idx = schedule_m2.index({j})\n
schedule_m2.insert(j_idx + 1, job)"""
    
    return {
        "question": generate_paraphrased_question(question),
        "code": output_code,
        "type": 2,
        "requires_scheduler": 0,
        "raw_question_hash": generate_hash(question)
    }


QUESTION_GENERATORS = {
    "precedence": generate_precedence_relationship,
    "processing-time-change": change_machine_processing_time,
    "due-date-change": change_due_date,
    "ready-time-change": change_ready_time,
    "setup-time-change": change_setup_time,
    "priority-change": change_priority,
    "fixed-machine": generate_fixed_machine,
    "machine-restricted": machine_restricted,
    "job-swap-same-machine": job_swap_same_machine,
    "job-swap-different-machines": job_swap_different_machines,
    "job-insert-same-machine": job_insert_same_machine,
    "job-insert-different-machine": job_insert_different_machine
}

def generate_single_sample(args):
    generator_func, env_params, question_type = args
    
    qa = generator_func(env_params)
    
    requires_scheduler = random.choice([0, 1])
    requires_scheduler_prompt_cand = [
        "Apply minimal repairs while maintaining the current schedule.", 
        "Use the existing scheduling algorithm to create a new schedule."
    ]
    
    suffix_prompt = generate_paraphrased_question(requires_scheduler_prompt_cand[requires_scheduler])

    question_seq = (random.random() < 0.5)
    if question_seq:
        final_question = qa["question"] + " " + suffix_prompt
    else:
        final_question = suffix_prompt + " " + qa["question"]

    qa['question'] = final_question
    qa['requires_scheduler'] = requires_scheduler
    qa['raw_question_hash'] = generate_hash(final_question)

    cls_sample = format_for_classification(final_question, qa["requires_scheduler"])
    mod_sample = format_for_modification_code(final_question, qa["code"])
    type_sample = format_for_modification_type(final_question, qa["type"])
    
    return cls_sample, mod_sample, type_sample, qa

def generate_single_sample_without_classification(args):
    generator_func, env_params, question_type = args
    
    qa = generator_func(env_params)

    qa['question'] = qa["question"]
    qa['raw_question_hash'] = generate_hash(qa["question"])

    mod_sample = format_for_modification_code(qa["question"], qa["code"])
    type_sample = format_for_modification_type(qa["question"], qa["type"])
    
    return mod_sample, type_sample, qa

def generate_dataset_parallel(env_params: Dict, samples_per_type: int = 1000, max_workers: int = 20) -> List:
    raw_data = []

    tasks = []
        
    print(f"Preparing tasks for {len(QUESTION_GENERATORS)} generators x {samples_per_type} samples...")
    
    for q_type, gen_func in QUESTION_GENERATORS.items():
        for _ in range(samples_per_type):
            tasks.append((gen_func, env_params, q_type))

    print(f"Starting parallel generation with {max_workers} workers for {len(tasks)} tasks.")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(generate_single_sample, tasks), total=len(tasks)))

    for cls_s, mod_s, type_s, raw_s in results:
        raw_data.append(raw_s)

    return raw_data

def generate_composite_queries(env_params: Dict, c: int = 2, samples_per_type: int = 500, max_workers: int = 20) -> List:
    raw_data = []

    question_generator_dict = {
        'job': ["precedence", "processing-time-change", "due-date-change", "ready-time-change", "setup-time-change", "priority-change"],
        'machine': ["fixed-machine", "machine-restricted"],
        'schedule': ["job-swap-same-machine", "job-swap-different-machines", "job-insert-same-machine", "job-insert-different-machine"]
    }

    requires_scheduler_prompt_cand = [
        "Apply minimal repairs while maintaining the current schedule.", 
        "Use the existing scheduling algorithm to create a new schedule."
    ]

    for level in ['job', 'machine', 'schedule']:
        all_tasks = []
        for _ in range(samples_per_type):
            selected_types = random.choices(question_generator_dict[level], k=c)
            for q_type in selected_types:
                all_tasks.append((QUESTION_GENERATORS[q_type], env_params, q_type))
        
        print(f"[{level}] Starting parallel generation with {max_workers} workers for {len(all_tasks)} tasks.")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            all_components = list(tqdm(executor.map(generate_single_sample_without_classification, all_tasks), total=len(all_tasks)))
        
        for i in range(0, len(all_components), c):
            group = all_components[i:i+c]
            
            questions = [comp[2]["question"] for comp in group]
            codes = [comp[2]["code"] for comp in group]
            types = [comp[2]["type"] for comp in group]
            
            full_question = " And ".join(questions)
            full_code = "\n".join(codes)
            
            requires_scheduler = random.choice([0, 1])
            suffix_prompt = generate_paraphrased_question(requires_scheduler_prompt_cand[requires_scheduler])
            
            if random.random() < 0.5:
                final_question = full_question + " " + suffix_prompt
            else:
                final_question = suffix_prompt + " " + full_question
            
            composite_raw = {
                "question": final_question,
                "code": full_code,
                "type": types,
                "requires_scheduler": requires_scheduler,
                "raw_question_hash": generate_hash(final_question)
            }
            
            raw_data.append(composite_raw)

    return raw_data

def parse_args():
    parser = argparse.ArgumentParser()
    add_auth_args(parser)
    parser.add_argument("--n-machines", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=100)
    parser.add_argument("--samples-per-type", type=int, default=300)
    parser.add_argument("--composite-min", type=int, default=2)
    parser.add_argument("--composite-max", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=30)
    parser.add_argument("--output-dir", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    apply_auth_args_to_env(args)
    os.makedirs(args.output_dir, exist_ok=True)

    env_params = {
        'n_m': args.n_machines,
        'n_j': args.n_jobs,
    }
    
    atomic_data = generate_dataset_parallel(
        env_params,
        samples_per_type=args.samples_per_type,
        max_workers=args.max_workers,
    )

    atomic_path = os.path.join(args.output_dir, "synthetic_atomic.json")
    with open(atomic_path, "w", encoding="utf-8") as f:
        json.dump(atomic_data, f, indent=2)
    print(f"Saved {len(atomic_data)} samples to {atomic_path}")

    for c in range(args.composite_min, args.composite_max + 1):
        composite_data = generate_composite_queries(
            env_params,
            c=c,
            samples_per_type=args.samples_per_type,
            max_workers=args.max_workers,
        )
        path = os.path.join(args.output_dir, f"synthetic_composite_{c}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(composite_data, f, indent=2)
        print(f"Saved {len(composite_data)} samples to {path}")
