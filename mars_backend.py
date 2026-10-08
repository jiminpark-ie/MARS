from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

from mars_adapters import (
    INT_TO_LEVEL,
    ExpConfig,
    Level,
    MarsBackend,
    default_exec_code,
)
from config import resolve_hf_token

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
   - data['precedence'][job_i][job_j] = 1 if job_i must complete before job_j starts, 0 otherwise
5. ReadyTimes: 1D array [n_j]
   - data['ready'][job_id] = earliest time when job can start
6. DueDates: 1D array [n_j]
   - data['due'][job_id] = deadline for job completion
7. Priorities: 1D array [n_j]
   - data['priority'][job_id] = priority weight for job
8. CurrentSchedule: Dict[int, List[int]]
   - data['schedule'][machine_id] = [sequenced jobs assigned to the machine]

"""

STOP_KEYWORDS = ["### User", "###", "User:", "Assistant:", "```"]


def _trim(reply: str) -> str:
    cut = len(reply)
    for kw in STOP_KEYWORDS:
        idx = reply.find(kw)
        if idx != -1:
            cut = min(cut, idx)
    return reply[:cut].strip()


def _first_int(text: str, default: int = 0) -> int:
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else default


def _json_from_reply(reply: str) -> Any:
    text = reply.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(text) if ch in "[{"]
    for start in starts:
        try:
            obj, _ = decoder.raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            continue
    raise ValueError(f"could not parse JSON from model reply: {reply!r}")


def _clean_request_text(value: Any) -> str:
    return str(value).strip().strip('"').strip("'").strip()


def _validate_request_list(value: Any, source: str) -> List[str]:
    if not isinstance(value, list):
        raise ValueError(f"{source} parser returned {type(value).__name__}, expected JSON array")
    out = [_clean_request_text(v) for v in value if _clean_request_text(v)]
    if not out:
        raise ValueError(f"{source} parser returned an empty request list")
    return out


class MarsModelBackend(MarsBackend):
    def __init__(self, cfg: ExpConfig,
                 base_model: str = "meta-llama/Llama-3.1-8B",
                 adapter_dir: Optional[str] = None,
                 icl_dir: str = "."):
        self.cfg = cfg
        self.base_model = base_model
        self.adapter_dir = adapter_dir
        self.icl_dir = icl_dir
        self.tokenizer = None
        self.model = None
        self._icl: Dict[str, List[Dict]] = {}

    def _ensure_model(self):
        if self.model is not None:
            return
        print(f"Loading tokenizer/model: {self.base_model}", flush=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import login

        token = resolve_hf_token(required=False)
        if token:
            login(token=token)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model, device_map="auto", token=token)
        if self.adapter_dir:
            for name in ("modification_type", "requires_scheduler"):
                path = os.path.join(self.adapter_dir, name)
                if os.path.isdir(path):
                    print(f"Loading adapter: {name} from {path}", flush=True)
                    self.model.load_adapter(path, adapter_name=name)
        print("Model ready", flush=True)

    def _set_adapter(self, name: Optional[str]):
        if not self.adapter_dir:
            return
        if name and hasattr(self.model, "set_adapter"):
            try:
                self.model.set_adapter(name)
            except Exception:
                pass
        elif name is None and hasattr(self.model, "disable_adapters"):
            try:
                self.model.disable_adapters()
            except Exception:
                pass

    def _generate(
        self,
        prompt: str,
        adapter: Optional[str] = None,
        max_new_tokens: int = 100,
        trim: bool = True,
    ) -> str:
        self._ensure_model()
        if adapter is not None:
            self._set_adapter(adapter)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        reply = self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
        return _trim(reply) if trim else reply

    def _load_icl(self, key: str) -> List[Dict]:
        if key not in self._icl:
            path = os.path.join(self.icl_dir, f"{key}.json")
            with open(path, "r", encoding="utf-8") as f:
                self._icl[key] = json.load(f)
        return self._icl[key]

    def _examples(self, key: str, n: Optional[int] = None) -> List[Any]:
        pool = self._load_icl(key)
        if n is not None:
            pool = pool[:n]
        out: List[Any] = []
        for ex in pool:
            if "text" in ex and not ("question" in ex and "code" in ex):
                out.append({"text": ex["text"]})
            else:
                out.append((ex.get("question", ex.get("QUESTION", "")), str(ex.get("code", ""))))
        return out

    def get_fewshot(self, level, finegrained_types=(), strategy="level_wise"):
        if level is None:
            if strategy == "all_levels":
                shots = []
                for key, k in (("icl_job", 1), ("icl_machine", 1), ("icl_schedule", 1)):
                    shots += self._examples(key, k)
                return shots
            return self._examples("icl")
        key = {Level.JOB: "icl_job", Level.MACHINE: "icl_machine", Level.SCHEDULE: "icl_schedule"}[level]
        return self._examples(key, None)

    def get_fewshot_all(self, order=(Level.JOB, Level.MACHINE, Level.SCHEDULE)):
        keymap = {Level.JOB: "icl_job", Level.MACHINE: "icl_machine", Level.SCHEDULE: "icl_schedule"}
        shots: List[Any] = []
        for lvl in order:
            shots += self._examples(keymap[lvl])
        return shots

    def build_prompt(self, query, fewshot):
        return self._prompt(query, fewshot)

    def _prompt(self, question: str, fewshot: Sequence) -> str:
        prompt = f"{SYSTEM_PROMPT}\n\n"
        for ex in fewshot:
            if isinstance(ex, dict) and "text" in ex:
                prompt += f"{ex['text']}\n\n"
            else:
                qx, cx = ex
                prompt += f"### Request:\n{qx}\n### Code:\n{cx}\n\n"
        prompt += f"### Request:\n{question}\n### Code:\n"
        return prompt

    def generate_code(self, query, data, fewshot, max_new_tokens=100):
        return self._generate(self._prompt(query, fewshot), adapter=None, max_new_tokens=max_new_tokens)

    def classify_level(self, query, data):
        prompt = f"### Request:\n{query}\n### Type:\n"
        return INT_TO_LEVEL.get(_first_int(self._generate(prompt, adapter="modification_type", max_new_tokens=1), 0), Level.JOB)

    def classify_rerun(self, query, data):
        prompt = f"### Request:\n{query}\n### Classification:\n"
        val = _first_int(self._generate(prompt, adapter="requires_scheduler", max_new_tokens=1), 0)
        return 1 if val == 1 else 0

    def split_requests(self, combined_query):
        prompt = f"""You are a parser for scheduling modification requests.
Return ONLY a valid JSON array of strings.
Remove any scheduler directive such as "Apply minimal repairs while maintaining the current schedule." or "Use the existing scheduling algorithm to create a new schedule."
Do not solve the requests and do not generate code.

Example request:
Apply minimal repairs while maintaining the current schedule. What if job 1 must be completed before job 2 starts? And What if machine 3 breaks down and becomes unusable?
Example JSON:
["What if job 1 must be completed before job 2 starts?", "What if machine 3 breaks down and becomes unusable?"]

Request:
{combined_query}
JSON:
"""
        reply = self._generate(prompt, adapter=None, max_new_tokens=384, trim=False)
        return _validate_request_list(_json_from_reply(reply), "atomic-request")

    def run_scheduler(self, data):
        from sim.schedulers import DEFAULT_RULE, run_named_scheduler

        rule = os.environ.get("MARS_SCHEDULER_RULE", DEFAULT_RULE)
        return run_named_scheduler(data, rule)

    def repair(self, data):
        from sim.repair_fixed import normalize_schedule_to_sequence, verify_repair

        seq = normalize_schedule_to_sequence(data.get("schedule", {}))
        verify_repair(data, seq)
        return seq

    def schedules_equal(self, d1, d2):
        from sim.repair_fixed import normalize_schedule_to_sequence as norm

        s1 = norm(d1.get("schedule", {}))
        s2 = norm(d2.get("schedule", {}))
        machines = set(s1) | set(s2)
        return all(s1.get(m, []) == s2.get(m, []) for m in machines)

    def exec_code(self, code, data):
        return default_exec_code(code, data)

    def count_tokens(self, text):
        self._ensure_model()
        return len(self.tokenizer(text)["input_ids"])


def build_backend(cfg: ExpConfig) -> MarsBackend:
    return MarsModelBackend(
        cfg,
        base_model=os.environ.get("MARS_BASE_MODEL", "meta-llama/Llama-3.1-8B"),
        adapter_dir=os.environ.get("MARS_ADAPTER_DIR", "./models/classifiers"),
        icl_dir=os.environ.get("MARS_ICL_DIR", "./icl"),
    )
