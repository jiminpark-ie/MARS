from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

class Level(str, Enum):
    JOB = "job"
    MACHINE = "machine"
    SCHEDULE = "schedule"


LEVEL_TO_INT = {Level.JOB: 0, Level.MACHINE: 1, Level.SCHEDULE: 2}
INT_TO_LEVEL = {v: k for k, v in LEVEL_TO_INT.items()}

JOB_LEVEL_TYPES = [
    "precedence",
    "processing_time_change",
    "due_date_change",
    "ready_time_change",
    "setup_time_change",
    "priority_change",
]
MACHINE_LEVEL_TYPES = [
    "fixed_machine_assignment",
    "machine_availability_restriction",
]
SCHEDULE_LEVEL_TYPES = [
    "job_swap_same_machine",
    "job_swap_across_machines",
    "job_insertion_same_machine",
    "job_insertion_across_machines",
]

TYPE_TO_LEVEL: Dict[str, Level] = {
    **{t: Level.JOB for t in JOB_LEVEL_TYPES},
    **{t: Level.MACHINE for t in MACHINE_LEVEL_TYPES},
    **{t: Level.SCHEDULE for t in SCHEDULE_LEVEL_TYPES},
}
LEVEL_TYPES: Dict[Level, List[str]] = {
    Level.JOB: JOB_LEVEL_TYPES,
    Level.MACHINE: MACHINE_LEVEL_TYPES,
    Level.SCHEDULE: SCHEDULE_LEVEL_TYPES,
}
ALL_TYPES = JOB_LEVEL_TYPES + MACHINE_LEVEL_TYPES + SCHEDULE_LEVEL_TYPES

SCHEDULER_DIRECTIVES = [
    "Apply minimal repairs while maintaining the current schedule.",
    "Use the existing scheduling algorithm to create a new schedule.",
]


@dataclass
class AtomicRequest:

    type: str 
    level: Level
    nl: str
    gold_code: str
    requires_scheduler: int = 0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiRequestInstance:

    c: int
    combined_nl: str
    components: List[AtomicRequest]
    data0: Dict[str, Any]
    requires_scheduler: int = 0
    kind: str = "mixed"
    same_level: Optional[Level] = None


@dataclass
class MethodResult:

    method: str
    gen_codes: List[str]
    gen_code_concat: str 
    final_data: Optional[Dict[str, Any]]
    executed: bool
    n_generation_calls: int = 0 
    n_llm_calls: int = 0 
    trace: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ExpConfig:
    c_values: Sequence[int] = (2, 3, 4, 5)
    n_per_c: int = 200
    n_repeats: int = 1
    n_machines: int = 10
    n_jobs: int = 100
    seed: int = 20260530
    mixed_shot_strategy: str = "all_levels"
    order: Tuple[Level, ...] = (Level.JOB, Level.MACHINE, Level.SCHEDULE)
    machine_forces_regen: bool = False
    evaluate_outcome: bool = False
    paraphrase: bool = False


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(to_jsonable(k)): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return to_jsonable(value.item())
    except Exception:
        pass

    return value


def _record_hash(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def _ordered_components(
    inst: MultiRequestInstance,
    order: Sequence[Level],
) -> List[AtomicRequest]:
    order_index = {level: i for i, level in enumerate(order)}
    return sorted(inst.components, key=lambda r: order_index.get(r.level, len(order_index)))


def atomic_request_to_record(req: AtomicRequest) -> Dict[str, Any]:
    return {
        "question": req.nl,
        "code": req.gold_code,
        "type_name": req.type,
        "level": req.level.value,
        "type": LEVEL_TO_INT[req.level],
        "requires_scheduler": int(req.requires_scheduler),
        "params": to_jsonable(req.params),
    }


def multirequest_to_legacy_record(
    inst: MultiRequestInstance,
    cfg: Optional[ExpConfig] = None,
    include_components: bool = True,
    include_data0: bool = False,
) -> Dict[str, Any]:
    
    order = cfg.order if cfg is not None else (Level.JOB, Level.MACHINE, Level.SCHEDULE)
    ordered = _ordered_components(inst, order)
    application_code = "\n".join(r.gold_code for r in ordered)
    query_order_code = "\n".join(r.gold_code for r in inst.components)
    record: Dict[str, Any] = {
        "question": inst.combined_nl,
        "code": application_code,
        "application_code": application_code,
        "query_order_code": query_order_code,
        "type": [LEVEL_TO_INT[r.level] for r in inst.components],
        "type_names": [r.type for r in inst.components],
        "levels": [r.level.value for r in inst.components],
        "requires_scheduler": int(inst.requires_scheduler),
        "raw_question_hash": _record_hash(inst.combined_nl),
        "c": int(inst.c),
        "kind": inst.kind,
        "same_level": None if inst.same_level is None else inst.same_level.value,
    }
    if include_components:
        record["components"] = [atomic_request_to_record(r) for r in inst.components]
        record["application_order"] = [atomic_request_to_record(r) for r in ordered]
    if include_data0:
        record["data0"] = to_jsonable(inst.data0)
    return to_jsonable(record)


def multirequests_to_legacy_records(
    instances: Sequence[MultiRequestInstance],
    cfg: Optional[ExpConfig] = None,
    include_components: bool = True,
    include_data0: bool = False,
) -> List[Dict[str, Any]]:
    return [
        multirequest_to_legacy_record(
            inst,
            cfg=cfg,
            include_components=include_components,
            include_data0=include_data0,
        )
        for inst in instances
    ]


class MarsBackend:

    def classify_level(self, query: str, data: Dict[str, Any]) -> Level:
        raise NotImplementedError

    def classify_rerun(self, query: str, data: Dict[str, Any]) -> int:
        raise NotImplementedError

    def get_fewshot(
        self,
        level: Optional[Level],
        finegrained_types: Sequence[str] = (),
        strategy: str = "level_wise",
    ) -> List[Tuple[str, str]]:
        raise NotImplementedError

    def generate_code(
        self,
        query: str,
        data: Dict[str, Any],
        fewshot: List[Tuple[str, str]],
    ) -> str:
        raise NotImplementedError

    def split_requests(self, combined_query: str) -> List[str]:
        raise NotImplementedError

    def split_requests_by_level(self, combined_query: str) -> Dict[Level, List[str]]:
        atomic = self.split_requests(combined_query)
        groups: Dict[Level, List[str]] = {}
        for query in atomic:
            level = self.classify_level(query, {})
            groups.setdefault(level, []).append(query)
        return groups

    def run_scheduler(self, data: Dict[str, Any]) -> Dict[int, List[int]]:
        raise NotImplementedError

    def repair(self, data: Dict[str, Any]) -> Dict[int, List[int]]:
        raise NotImplementedError

    def schedules_equal(self, d1: Dict[str, Any], d2: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def exec_code(self, code: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return default_exec_code(code, data)

    def count_tokens(self, text: str) -> int:
        return 0


def default_exec_code(code: str, data: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np

    local = copy.deepcopy(data)
    ns: Dict[str, Any] = {"data": local, "np": np}
    exec(code, {"__builtins__": __builtins__, "np": np}, ns)
    return ns["data"]


def simple_edd_schedule(data: Dict[str, Any]) -> Dict[int, List[int]]:
    pt = data["processing_time"]
    elig = data["machine_eligibility"]
    due = data["due"]
    n_m, n_j = pt.shape
    load = [0.0] * n_m
    assign: Dict[int, List[int]] = {m: [] for m in range(n_m)}
    for j in sorted(range(n_j), key=lambda jj: float(due[jj])):
        eligible = [m for m in range(n_m) if int(elig[m][j]) == 1]
        if not eligible:
            raise ValueError(f"job {j} has no eligible machine")
        m = min(eligible, key=lambda mm: load[mm])
        assign[m].append(j)
        load[m] += float(pt[m][j])
    for m in range(n_m):
        assign[m].sort(key=lambda jj: float(due[jj]))
    return assign
