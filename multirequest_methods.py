from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

from mars_adapters import (
    ExpConfig,
    Level,
    MarsBackend,
    MethodResult,
    MultiRequestInstance,
)

Mod = Dict[str, Any]


def _has_schedule(inst) -> bool:
    return any(r.level == Level.SCHEDULE for r in inst.components)


def _token_budget(n_requests: int, has_schedule: bool) -> int:
    per = 110 if has_schedule else 48
    return min(1024, 64 + per * max(1, n_requests))


def _gen(backend, subquery, data0, fewshot, sink, max_new_tokens=100):
    prompt = backend.build_prompt(subquery, fewshot) if hasattr(backend, "build_prompt") else ""
    code = backend.generate_code(subquery, data0, fewshot, max_new_tokens=max_new_tokens)
    sink.append({"subquery": subquery, "prompt": prompt, "code": code,
                 "max_new_tokens": max_new_tokens})
    return code


def _attach(res, gen_log, inst, items=None):
    res.gen_log = gen_log
    res.subqueries = [g["subquery"] for g in gen_log]
    res.prompts = [g["prompt"] for g in gen_log]
    res.gen_budgets = [g.get("max_new_tokens") for g in gen_log]
    res.component_types = [r.type for r in inst.components]
    res.component_levels = [r.level.value for r in inst.components]
    res.component_gold_codes = [r.gold_code for r in inst.components]
    if items:
        res.split_nls = [it["nl"] for it in items]
        res.pred_levels = [it["level"].value if hasattr(it["level"], "value")
                           else str(it["level"]) for it in items]
    else:
        res.split_nls = []
        res.pred_levels = []
    return res

def code_executes(data0: Dict[str, Any], codes: List[str], backend: MarsBackend) -> bool:

    data = copy.deepcopy(data0)
    try:
        for code in codes:
            data = backend.exec_code(code, data)
        return True
    except Exception:
        return False
    
def apply_modifications(
    data0: Dict[str, Any], mods: List[Mod], rerun: int, backend: MarsBackend, cfg: ExpConfig
) -> Tuple[Dict[str, Any], bool, List[str]]:
    trace: List[str] = []
    data = copy.deepcopy(data0)
    try:
        instance_mods = [m for m in mods if m["level"] in (Level.JOB, Level.MACHINE)]
        schedule_mods = [m for m in mods if m["level"] == Level.SCHEDULE]
        instance_mods.sort(key=lambda m: cfg.order.index(m["level"]))
        for m in instance_mods:
            data = backend.exec_code(m["code"], data)
        if rerun:
            data["schedule"] = backend.run_scheduler(data)
        for m in schedule_mods:
            data = backend.exec_code(m["code"], data)
        data["schedule"] = backend.repair(data)
        return data, True, trace
    except Exception as e:
        trace.append(f"ERROR: {e!r}")
        return data, False, trace


def apply_monolithic(
    data0: Dict[str, Any], code: str, rerun: int, backend: MarsBackend, cfg: ExpConfig
) -> Tuple[Dict[str, Any], bool, List[str]]:
    trace: List[str] = []
    data = copy.deepcopy(data0)
    try:
        data = backend.exec_code(code, data)
        if rerun:
            data["schedule"] = backend.run_scheduler(data)
        data["schedule"] = backend.repair(data)
        return data, True, trace
    except Exception as e:
        trace.append(f"ERROR: {e!r}")
        return data, False, trace


def apply_level_blobs(
    data0: Dict[str, Any], blobs: Dict[Level, str], rerun: int, backend: MarsBackend, cfg: ExpConfig
) -> Tuple[Dict[str, Any], bool, List[str]]:
    trace: List[str] = []
    data = copy.deepcopy(data0)
    try:
        for lvl in (Level.JOB, Level.MACHINE):
            if lvl in blobs:
                data = backend.exec_code(blobs[lvl], data)
        if rerun:
            data["schedule"] = backend.run_scheduler(data)
        if Level.SCHEDULE in blobs:
            data = backend.exec_code(blobs[Level.SCHEDULE], data)
        data["schedule"] = backend.repair(data)
        return data, True, trace
    except Exception as e:
        trace.append(f"ERROR: {e!r}")
        return data, False, trace


def _finish(data0, ordered_codes, rerun, backend, cfg, apply_fn, apply_arg):
    if cfg.evaluate_outcome:
        return apply_fn(data0, apply_arg, rerun, backend, cfg)
    executed = code_executes(data0, ordered_codes, backend)
    return None, executed, []



def gold_apply(inst: MultiRequestInstance, backend: MarsBackend, cfg: ExpConfig):
    mods = [{"code": r.gold_code, "level": r.level} for r in inst.components]
    data, executed, _ = apply_modifications(inst.data0, mods, inst.requires_scheduler, backend, cfg)
    return data, executed


def gold_concat(inst: MultiRequestInstance, cfg: ExpConfig) -> str:
    comps = sorted(inst.components, key=lambda r: cfg.order.index(r.level))
    return "\n".join(r.gold_code for r in comps)


def _split(inst, backend, cfg) -> Tuple[List[str], int]:
    return backend.split_requests(inst.combined_nl), 1


def _join_subquery(nls: List[str]) -> str:
    return " And ".join(nls)


_ACTIONABLE_REQUEST_RE = re.compile(r"\b(?:job|machine)\s+#?\d+", re.IGNORECASE)


def _is_actionable_request(nl: str) -> bool:
    return bool(_ACTIONABLE_REQUEST_RE.search(nl or ""))


def _split_and_classify(inst, backend, cfg) -> Tuple[List[Dict[str, Any]], int, int]:
    atomic_nls, n_split = _split(inst, backend, cfg)
    actionable = [nl for nl in atomic_nls if _is_actionable_request(nl)]
    if actionable and len(actionable) < len(atomic_nls):
        atomic_nls = actionable
    items = [{"nl": nl, "level": backend.classify_level(nl, inst.data0)}
             for nl in atomic_nls]
    return items, n_split, len(items)


def monolithic(inst, backend, cfg) -> MethodResult:
    gen_log = []
    fewshot = backend.get_fewshot_all()
    mt = _token_budget(inst.c, has_schedule=True)
    code = _gen(backend, inst.combined_nl, inst.data0, fewshot, gen_log, max_new_tokens=mt)
    rerun = backend.classify_rerun(inst.combined_nl, inst.data0)
    final, executed, trace = _finish(inst.data0, [code], rerun, backend, cfg,
                                     apply_monolithic, code)
    res = MethodResult("monolithic", [code], code, final, executed,
                       n_generation_calls=1, n_llm_calls=2, trace=trace)
    return _attach(res, gen_log, inst)


def per_request(inst, backend, cfg) -> MethodResult:
    rerun = backend.classify_rerun(inst.combined_nl, inst.data0)
    items, n_split, n_classify = _split_and_classify(inst, backend, cfg)
    mods: List[Mod] = []
    gen_log = []
    n_gen = 0
    for it in items:
        fewshot = backend.get_fewshot(level=it["level"])
        mt = _token_budget(1, it["level"] == Level.SCHEDULE)
        code = _gen(backend, it["nl"], inst.data0, fewshot, gen_log, max_new_tokens=mt)
        n_gen += 1
        mods.append({"code": code, "level": it["level"]})
    ordered = sorted(mods, key=lambda m: cfg.order.index(m["level"]))
    ordered_codes = [m["code"] for m in ordered]
    final, executed, trace = _finish(inst.data0, ordered_codes, rerun, backend, cfg,
                                     apply_modifications, mods)
    concat = "\n".join(ordered_codes)
    res = MethodResult("per_request", [m["code"] for m in mods], concat, final, executed,
                       n_generation_calls=n_gen,
                       n_llm_calls=n_split + 1 + n_classify + n_gen,
                       trace=trace)
    return _attach(res, gen_log, inst, items)


def per_level(inst, backend, cfg) -> MethodResult:
    rerun = backend.classify_rerun(inst.combined_nl, inst.data0)
    items, n_split, n_classify = _split_and_classify(inst, backend, cfg)

    grouped: Dict[Level, List[str]] = {Level.JOB: [], Level.MACHINE: [], Level.SCHEDULE: []}
    for it in items:
        grouped.setdefault(it["level"], []).append(it["nl"])
    blobs: Dict[Level, str] = {}
    gen_log = []
    n_gen = 0
    for level in cfg.order:
        nls = grouped.get(level, [])
        if not nls:
            continue
        fewshot = backend.get_fewshot(level=level)
        mt = _token_budget(len(nls), level == Level.SCHEDULE)
        blobs[level] = _gen(backend, _join_subquery(nls), inst.data0, fewshot, gen_log, max_new_tokens=mt)
        n_gen += 1
    ordered_codes = [blobs[l] for l in cfg.order if l in blobs]
    final, executed, trace = _finish(inst.data0, ordered_codes, rerun, backend, cfg,
                                     apply_level_blobs, blobs)
    concat = "\n".join(ordered_codes)
    res = MethodResult("per_level", ordered_codes, concat, final, executed,
                       n_generation_calls=n_gen,
                       n_llm_calls=n_split + 1 + n_classify + n_gen,
                       trace=trace)
    return _attach(res, gen_log, inst, items)


METHODS = {
    "monolithic": monolithic,
    "per_request": per_request,
    "per_level": per_level,
}


def _classification_accuracy(result, inst) -> Tuple[float, float]:
    pred = getattr(result, "pred_levels", None)
    if not pred:
        return float("nan"), float("nan")
    gold = [r.level.value for r in inst.components]
    if len(pred) == len(gold):
        if not gold:
            return float("nan"), 1.0
        correct = sum(1.0 for p, g in zip(pred, gold) if p == g)
        return correct / len(gold), 1.0
    inter = sum((Counter(pred) & Counter(gold)).values())
    denom = max(len(pred), len(gold))
    return (inter / denom if denom else float("nan")), 0.0


def evaluate_result(result, inst, gold_final, backend, cfg) -> Dict[str, float]:
    cer = 1.0 if result.executed else 0.0
    tsr = float("nan")
    if cfg.evaluate_outcome and result.executed and result.final_data is not None and gold_final is not None:
        try:
            tsr = 1.0 if backend.schedules_equal(result.final_data, gold_final) else 0.0
        except Exception:
            tsr = 0.0
    type_acc, split_count_ok = _classification_accuracy(result, inst)
    return {
        "cer": cer,
        "tsr": tsr,
        "type_acc": type_acc,
        "split_count_ok": split_count_ok,
        "gen_calls": float(result.n_generation_calls),
        "llm_calls": float(result.n_llm_calls),
    }
