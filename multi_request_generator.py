from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mars_adapters import (
    LEVEL_TYPES,
    SCHEDULER_DIRECTIVES,
    AtomicRequest,
    ExpConfig,
    Level,
    MultiRequestInstance,
    multirequests_to_legacy_records,
    to_jsonable,
    simple_edd_schedule,
)


def _q(rng, base: str, R=None) -> str:
    if R and R.get("paraphrase"):
        from llm_utils import generate_paraphrased_question
        return generate_paraphrased_question(base)
    return base


def _directive(regen: bool, paraphrase: bool = False) -> str:
    base = SCHEDULER_DIRECTIVES[1] if regen else SCHEDULER_DIRECTIVES[0]
    if paraphrase:
        from llm_utils import generate_paraphrased_question
        return generate_paraphrased_question(base)
    return base


def build_instance(rng: np.random.Generator, n_m: int, n_j: int) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "processing_time": rng.integers(1, 100, size=(n_m, n_j)),
        "machine_eligibility": np.ones((n_m, n_j), dtype=int),
        "setup_time": rng.integers(0, 20, size=(n_m, n_j, n_j)),
        "precedence": np.zeros((n_j, n_j), dtype=int),
        "ready": rng.integers(0, 50, size=n_j),
        "due": rng.integers(1, 100, size=n_j),
        "priority": np.ones(n_j, dtype=float),
    }
    data["schedule"] = simple_edd_schedule(data)
    return data

def _ranges(cfg: ExpConfig) -> Dict[str, int]:
    return {"job_lo": 0, "job_hi": cfg.n_jobs, "mach_lo": 0, "mach_hi": cfg.n_machines,
            "paraphrase": bool(getattr(cfg, "paraphrase", False))}


def _rand_job(rng, R) -> int:
    return int(rng.integers(R["job_lo"], R["job_hi"]))


def _rand_machine(rng, R) -> int:
    return int(rng.integers(R["mach_lo"], R["mach_hi"]))


def _machine_with_min_jobs(rng, data0, k: int, R) -> Optional[int]:
    cands = [m for m in range(R["mach_lo"], R["mach_hi"]) if len(data0["schedule"].get(m, [])) >= k]
    return int(rng.choice(cands)) if cands else None


def _two_jobs_on_machine(rng, data0, m: int) -> Tuple[int, int]:
    seq = list(data0["schedule"][m])
    a, b = rng.choice(len(seq), size=2, replace=False)
    return int(seq[a]), int(seq[b])


def _one_job_on_machine(rng, data0, m: int) -> int:
    seq = list(data0["schedule"][m])
    return int(seq[int(rng.integers(0, len(seq)))])


class _Infeasible(Exception):
    """Raised by a generator when the current instance can't satisfy a request"""


def gen_precedence(rng, data0, R) -> AtomicRequest:
    i = _rand_job(rng, R)
    j = _rand_job(rng, R)
    while j == i:
        j = _rand_job(rng, R)
    nl = _q(rng, f"What if job {i} must be completed before job {j} starts?", R)
    return AtomicRequest("precedence", Level.JOB, nl,
                         f"data['precedence'][{i}][{j}] = 1", 1, {"jobs": [i, j]})


def gen_processing_time_change(rng, data0, R) -> AtomicRequest:
    i = _rand_job(rng, R)
    m = _rand_machine(rng, R)
    t = int(rng.integers(1, 100))
    nl = _q(rng, f"What if the processing time of job {i} on machine {m} is changed to {t}?", R)
    return AtomicRequest("processing_time_change", Level.JOB, nl,
                         f"data['processing_time'][{m}, {i}] = {t}", 1, {"jobs": [i], "machines": [m]})


def gen_due_date_change(rng, data0, R) -> AtomicRequest:
    i = _rand_job(rng, R)
    t = int(rng.integers(1, 100))
    nl = _q(rng, f"What if the due date of job {i} is changed to {t}?", R)
    return AtomicRequest("due_date_change", Level.JOB, nl,
                         f"data['due'][{i}] = {t}", 1, {"jobs": [i]})


def gen_ready_time_change(rng, data0, R) -> AtomicRequest:
    i = _rand_job(rng, R)
    t = int(rng.integers(1, 100))
    nl = _q(rng, f"What if the ready time of job {i} is changed to {t}?", R)
    return AtomicRequest("ready_time_change", Level.JOB, nl,
                         f"data['ready'][{i}] = {t}", 1, {"jobs": [i]})


def gen_setup_time_change(rng, data0, R) -> AtomicRequest:
    i = _rand_job(rng, R)
    j = _rand_job(rng, R)
    while j == i:
        j = _rand_job(rng, R)
    m = _rand_machine(rng, R)
    t = int(rng.integers(1, 100))
    nl = _q(rng, f"What if the setup time on machine {m} from job {i} to job {j} is changed to {t}?", R)
    return AtomicRequest("setup_time_change", Level.JOB, nl,
                         f"data['setup_time'][{m}, {i}, {j}] = {t}", 1, {"jobs": [i, j], "machines": [m]})


def gen_priority_change(rng, data0, R) -> AtomicRequest:
    i = _rand_job(rng, R)
    nl = _q(rng, f"What if job {i} should be processed with the highest priority?", R)
    return AtomicRequest("priority_change", Level.JOB, nl,
                         f"data['priority'][{i}] = float('inf')", 1, {"jobs": [i]})


def gen_fixed_machine_assignment(rng, data0, R) -> AtomicRequest:
    i = _rand_job(rng, R)
    m = _rand_machine(rng, R)
    nl = _q(rng, f"What if job {i} can be processed only on machine {m}?", R)
    code = (
        f"data['machine_eligibility'][:, {i}] = 0\n"
        f"data['machine_eligibility'][{m}, {i}] = 1"
    )
    return AtomicRequest("fixed_machine_assignment", Level.MACHINE, nl, code, 1,
                         {"jobs": [i], "machines": [m]})


def gen_machine_availability_restriction(rng, data0, R) -> AtomicRequest:
    m = _rand_machine(rng, R)
    nl = _q(rng, f"What if machine {m} breaks down and becomes unusable?", R)
    code = f"data['machine_eligibility'][{m}, :] = 0"
    affected = list(data0["schedule"].get(m, []))  # reassigned on regeneration
    return AtomicRequest("machine_availability_restriction", Level.MACHINE, nl, code, 1,
                         {"jobs": affected, "machines": [m]})


def gen_job_swap_same_machine(rng, data0, R) -> AtomicRequest:
    m = _machine_with_min_jobs(rng, data0, 2, R)
    if m is None:
        raise _Infeasible("no machine with >=2 jobs")
    i, j = _two_jobs_on_machine(rng, data0, m)
    nl = _q(rng, f"What if the positions of job {i} and job {j} on machine {m} are swapped?", R)
    code = (
        f"schedule_m = data['schedule'][{m}]\n\n"
        f"i_idx = schedule_m.index({i})\n\n"
        f"j_idx = schedule_m.index({j})\n\n"
        f"schedule_m[i_idx], schedule_m[j_idx] = schedule_m[j_idx], schedule_m[i_idx]"
    )
    return AtomicRequest("job_swap_same_machine", Level.SCHEDULE, nl, code, 0,
                         {"jobs": [i, j], "machines": [m]})


def gen_job_swap_across_machines(rng, data0, R) -> AtomicRequest:
    m1 = _machine_with_min_jobs(rng, data0, 1, R)
    m2 = _machine_with_min_jobs(rng, data0, 1, R)
    tries = 0
    while (m2 is None or m2 == m1) and tries < 20:
        m2 = _machine_with_min_jobs(rng, data0, 1, R)
        tries += 1
    if m1 is None or m2 is None or m1 == m2:
        raise _Infeasible("need two distinct non-empty machines")
    i = _one_job_on_machine(rng, data0, m1)
    j = _one_job_on_machine(rng, data0, m2)
    nl = _q(rng, f"What if job {i} on machine {m1} is swapped with job {j} on machine {m2}?", R)
    code = (
        f"schedule_m1 = data['schedule'][{m1}]\n\n"
        f"schedule_m2 = data['schedule'][{m2}]\n\n"
        f"i_idx = schedule_m1.index({i})\n\n"
        f"j_idx = schedule_m2.index({j})\n\n"
        f"schedule_m1[i_idx], schedule_m2[j_idx] = schedule_m2[j_idx], schedule_m1[i_idx]"
    )
    return AtomicRequest("job_swap_across_machines", Level.SCHEDULE, nl, code, 0,
                         {"jobs": [i, j], "machines": [m1, m2]})


def gen_job_insertion_same_machine(rng, data0, R) -> AtomicRequest:
    m = _machine_with_min_jobs(rng, data0, 2, R)
    if m is None:
        raise _Infeasible("no machine with >=2 jobs")
    i, j = _two_jobs_on_machine(rng, data0, m)
    before = bool(rng.random() < 0.5)
    pos = "before" if before else "after"
    nl = _q(rng, f"What if job {i} on machine {m} is moved to immediately {pos} job {j} on the same machine?", R)
    insert = "j_idx" if before else "j_idx + 1"
    code = (
        f"schedule_m = data['schedule'][{m}]\n"
        f"i_idx = schedule_m.index({i})\n"
        f"job = schedule_m.pop(i_idx)\n"
        f"j_idx = schedule_m.index({j})\n"
        f"schedule_m.insert({insert}, job)"
    )
    return AtomicRequest("job_insertion_same_machine", Level.SCHEDULE, nl, code, 0,
                         {"jobs": [i, j], "machines": [m]})


def gen_job_insertion_across_machines(rng, data0, R) -> AtomicRequest:
    m1 = _machine_with_min_jobs(rng, data0, 1, R)
    m2 = _machine_with_min_jobs(rng, data0, 1, R)
    tries = 0
    while (m2 is None or m2 == m1) and tries < 20:
        m2 = _machine_with_min_jobs(rng, data0, 1, R)
        tries += 1
    if m1 is None or m2 is None or m1 == m2:
        raise _Infeasible("need two distinct non-empty machines")
    i = _one_job_on_machine(rng, data0, m1)
    j = _one_job_on_machine(rng, data0, m2)
    before = bool(rng.random() < 0.5)
    pos = "before" if before else "after"
    nl = _q(rng, f"What if job {i} on machine {m1} is moved to just {pos} job {j} on machine {m2}?", R)
    insert = "j_idx" if before else "j_idx + 1"
    code = (
        f"schedule_m1 = data['schedule'][{m1}]\n"
        f"schedule_m2 = data['schedule'][{m2}]\n"
        f"job = schedule_m1.pop(schedule_m1.index({i}))\n"
        f"j_idx = schedule_m2.index({j})\n"
        f"schedule_m2.insert({insert}, job)"
    )
    return AtomicRequest("job_insertion_across_machines", Level.SCHEDULE, nl, code, 0,
                         {"jobs": [i, j], "machines": [m1, m2]})


GENERATORS = {
    "precedence": gen_precedence,
    "processing_time_change": gen_processing_time_change,
    "due_date_change": gen_due_date_change,
    "ready_time_change": gen_ready_time_change,
    "setup_time_change": gen_setup_time_change,
    "priority_change": gen_priority_change,
    "fixed_machine_assignment": gen_fixed_machine_assignment,
    "machine_availability_restriction": gen_machine_availability_restriction,
    "job_swap_same_machine": gen_job_swap_same_machine,
    "job_swap_across_machines": gen_job_swap_across_machines,
    "job_insertion_same_machine": gen_job_insertion_same_machine,
    "job_insertion_across_machines": gen_job_insertion_across_machines,
}



def _choose_types(rng, c, kind, level, rs, machine_forces_regen) -> List[str]:
    if kind == "mixed":
        if rs == 1: # re-run
            levels_avail = [Level.JOB, Level.MACHINE]
            forced = [Level.JOB, Level.MACHINE]
        else: # shift
            levels_avail = (
                [Level.JOB, Level.SCHEDULE] if machine_forces_regen
                else [Level.JOB, Level.MACHINE, Level.SCHEDULE]
            )
            idx = rng.choice(len(levels_avail), size=2, replace=False)
            forced = [levels_avail[int(i)] for i in idx]
        type_pool = [t for L in levels_avail for t in LEVEL_TYPES[L]]
        chosen = [str(rng.choice(LEVEL_TYPES[forced[0]])), str(rng.choice(LEVEL_TYPES[forced[1]]))]
        remaining = [t for t in type_pool if t not in chosen]
        k = max(0, c - 2)
        if k > 0:
            take = min(k, len(remaining))
            ei = rng.choice(len(remaining), size=take, replace=False)
            chosen += [remaining[int(i)] for i in ei]
            while len(chosen) < c:
                chosen.append(str(rng.choice(type_pool)))
        perm = rng.permutation(len(chosen))
        return [chosen[int(i)] for i in perm][:c]
    # same level
    pool = LEVEL_TYPES[level]
    replace = c > len(pool)
    idx = rng.choice(len(pool), size=c, replace=replace)
    return [pool[int(i)] for i in idx]


def make_multirequest(
    rng: np.random.Generator,
    cfg: ExpConfig,
    c: int,
    kind: str = "mixed",
    level: Optional[Level] = None,
    max_attempts: int = 400,
) -> MultiRequestInstance:

    R = _ranges(cfg)
    for _outer in range(max_attempts):
        data0 = build_instance(rng, cfg.n_machines, cfg.n_jobs)

        if kind == "same_level" and level == Level.SCHEDULE:
            rs = 0
        elif kind == "same_level" and level == Level.MACHINE and cfg.machine_forces_regen:
            rs = 1
        else:
            rs = int(rng.integers(0, 2))

        types = _choose_types(rng, c, kind, level, rs, cfg.machine_forces_regen)

        used_jobs: set = set()
        blocked_machines: set = set()
        comps: List[AtomicRequest] = []
        ok = True
        for t in types:
            placed = False
            for _inner in range(80):
                try:
                    req = GENERATORS[t](rng, data0, R)
                except _Infeasible:
                    break
                jobs = set(req.params.get("jobs", []))
                machines = set(req.params.get("machines", []))
                struct = machines if req.level in (Level.MACHINE, Level.SCHEDULE) else set()
                if jobs & used_jobs:
                    continue
                if struct & blocked_machines:
                    continue
                used_jobs |= jobs
                blocked_machines |= struct
                comps.append(req)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not (ok and len(comps) == c):
            continue

        body = " And ".join(r.nl for r in comps)
        directive = _directive(rs == 1, paraphrase=R.get("paraphrase", False))
        combined = (directive + " " + body) if rng.random() < 0.5 else (body + " " + directive)
        return MultiRequestInstance(
            c=c, combined_nl=combined, components=comps, data0=data0,
            requires_scheduler=rs, kind=kind, same_level=level,
        )
    raise RuntimeError(
        f"could not build c={c} kind={kind} level={level}; "
        f"increase n_jobs/n_machines or lower c."
    )


def build_dataset(
    rng: np.random.Generator,
    cfg: ExpConfig,
    c: int,
    kind: str = "mixed",
    level: Optional[Level] = None,
    progress_label: Optional[str] = None,
) -> List[MultiRequestInstance]:
    out: List[MultiRequestInstance] = []
    fails = 0
    next_report = 1
    report_every = max(1, cfg.n_per_c // 10)
    while len(out) < cfg.n_per_c:
        try:
            out.append(make_multirequest(rng, cfg, c, kind, level))
            if progress_label and len(out) >= next_report:
                print(f"[data] {progress_label}: {len(out)}/{cfg.n_per_c}", flush=True)
                next_report += report_every
        except RuntimeError:
            fails += 1
            if fails > cfg.n_per_c + 50:
                raise RuntimeError(
                    f"repeated failures building c={c} kind={kind} level={level}; "
            )
    return out


def dataset_to_records(
    instances: List[MultiRequestInstance],
    cfg: Optional[ExpConfig] = None,
    include_components: bool = True,
    include_data0: bool = False,
) -> List[Dict[str, Any]]:
    return multirequests_to_legacy_records(
        instances,
        cfg=cfg,
        include_components=include_components,
        include_data0=include_data0,
    )


def save_dataset_json(
    instances: List[MultiRequestInstance],
    path: str,
    cfg: Optional[ExpConfig] = None,
    include_components: bool = True,
    include_data0: bool = False,
) -> None:

    records = dataset_to_records(
        instances,
        cfg=cfg,
        include_components=include_components,
        include_data0=include_data0,
    )
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(records), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(tmp_path, path)
