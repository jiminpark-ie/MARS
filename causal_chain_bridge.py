from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import causal_chain_audit as audit

Assignment = Tuple[int, int, float, float]


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    for nm in names:
        if isinstance(obj, dict) and nm in obj:
            return obj[nm]
        if not isinstance(obj, dict) and hasattr(obj, nm):
            return getattr(obj, nm)
    return default


def _children(node: Any) -> List[Any]:
    ch = _get(node, "children", "subnodes", "child", default=None)
    if ch is None:
        return []
    return list(ch) if isinstance(ch, (list, tuple)) else [ch]


def _unwrap(causal: Any) -> Any:
    dc = _get(causal, "direct_changes", default=None)
    return dc if dc is not None else causal



def _schedule_to_ops(schedule: Any,
                     data: Optional[dict] = None) -> Dict[str, List[Dict[str, Any]]]:
    timed_lookup: Optional[Dict[int, tuple]] = None

    def _lookup(job: int) -> tuple:
        nonlocal timed_lookup
        if timed_lookup is None:
            if data is None:
                raise ValueError(
                    "sequence-only schedule requires `data` to compute semi-active timings; refusing to fabricate placeholder times"
                )
            from sim.repair_fixed import (
                calculate_timing_from_sequence,
                normalize_schedule_to_sequence,
            )
            seq_only = normalize_schedule_to_sequence(
                {int(mk): [int(_get(it, "job", "job_id", "jid"))
                           if isinstance(it, dict) else
                           (int(it[0]) if isinstance(it, (tuple, list)) else int(it))
                           for it in sq]
                 for mk, sq in schedule.items()}
            )
            timed = calculate_timing_from_sequence(data, seq_only)
            timed_lookup = {int(j): (float(s), float(e))
                            for _m, jobs in timed.items() for (j, s, e) in jobs}
        return timed_lookup[job]

    ops: Dict[str, List[Dict[str, Any]]] = {}
    for m_key, seq in schedule.items():
        m = str(int(m_key))
        ops[m] = []
        for pos, item in enumerate(seq):
            if isinstance(item, dict):
                job = int(_get(item, "job", "job_id", "jid"))
                start = _get(item, "start", "begin", "s", default=None)
                end = _get(item, "end", "finish", "completion", "e", default=None)
                if start is None or end is None:
                    start, end = _lookup(job)
                ops[m].append({"job": job, "start": float(start), "end": float(end)})
            elif isinstance(item, (tuple, list)) and len(item) == 3:
                job, start, end = int(item[0]), float(item[1]), float(item[2])
                ops[m].append({"job": job, "start": start, "end": end})
            else:
                job = int(item)
                start, end = _lookup(job)
                ops[m].append({"job": job, "start": start, "end": end})
    return ops


def _coerce_after(raw_after: Any) -> Optional[Assignment]:
    if raw_after is None:
        return None
    m = _get(raw_after, "machine", "machine_id", "m")
    p = _get(raw_after, "position", "pos", "index", "idx")
    s = _get(raw_after, "start", "begin", "s")
    e = _get(raw_after, "end", "finish", "e")
    if None in (m, p, s, e):
        return None
    return (int(m), int(p), float(s), float(e))


def _looks_flat(causal: Any) -> bool:
    return (isinstance(causal, list) and len(causal) > 0
            and all(isinstance(x, dict)
                    and ("node_id" in x or "id" in x)
                    and ("parent_id" in x or "depth" in x)
                    and not _children(x)
                    for x in causal))


def _chain_to_nodes(causal: Any,
                    mod_assign: Dict[int, Assignment]) -> List[Dict[str, Any]]:
    nodes_attr = _get(causal, "nodes", "node_list", default=None)
    if nodes_attr is not None and _looks_flat(list(nodes_attr)):
        causal = list(nodes_attr)
    else:
        causal = _unwrap(causal)

    out: List[Dict[str, Any]] = []

    def emit(raw: Any, node_id: int, parent_id: Optional[int], depth: int) -> Dict[str, Any]:
        job = _get(raw, "job", "job_id", "jid")
        after = _coerce_after(_get(raw, "after", "to_state", "post", "new_state"))
        if after is None:
            after = _coerce_after({
                "machine": _get(raw, "machine_id", "to_machine"),
                "position": _get(raw, "curr_position"),
                "start": _get(raw, "curr_start", "to_start"),
                "end": _get(raw, "curr_end"),
            })
        if after is None and job is not None and int(job) in mod_assign:
            after = mod_assign[int(job)]
        node: Dict[str, Any] = {"node_id": node_id, "parent_id": parent_id, "depth": depth}
        if job is not None:
            node["job"] = int(job)
        if after is not None:
            node["after"] = {"machine": after[0], "position": after[1],
                             "start": after[2], "end": after[3]}
        for extra in ("kind", "label"):
            v = _get(raw, extra)
            if v is not None:
                node[extra] = v
        out.append(node)
        return node

    if _looks_flat(causal):
        for raw in causal:
            emit(raw,
                 node_id=int(_get(raw, "node_id", "id")),
                 parent_id=(None if _get(raw, "parent_id") is None
                            else int(_get(raw, "parent_id"))),
                 depth=int(_get(raw, "depth")))
        return out

    roots = causal if isinstance(causal, (list, tuple)) else [causal]
    counter = {"n": 0}

    def walk(raw: Any, parent_id: Optional[int], depth: int) -> None:
        nid = counter["n"]; counter["n"] += 1
        emit(raw, node_id=nid, parent_id=parent_id, depth=depth)
        for ch in _children(raw):
            walk(ch, parent_id=nid, depth=depth + 1)

    for r in roots:
        walk(r, parent_id=None, depth=0)
    return out

def to_audit_case(case_id: str,
                  original_schedule: Any,
                  modified_schedule: Any,
                  causal_chain: Any,
                  data: Optional[dict] = None) -> Tuple[Dict[str, Any], bool]:
    orig_ops = _schedule_to_ops(original_schedule, data)
    mod_ops = _schedule_to_ops(modified_schedule, data)

    mod_assign: Dict[int, Assignment] = {}
    for m_key, seq in mod_ops.items():
        for pos, op in enumerate(seq):
            mod_assign[int(op["job"])] = (int(m_key), pos, float(op["start"]), float(op["end"]))

    nodes = _chain_to_nodes(causal_chain, mod_assign)
    derived = any(("job" in n and "after" in n) for n in nodes) and not _chain_records_after(causal_chain)

    case = {
        "case_id": case_id,
        "original_schedule": orig_ops,
        "modified_schedule": mod_ops,
        "causal_chain": nodes,
    }
    return case, derived


def _chain_records_after(causal: Any) -> bool:
    causal = _unwrap(causal)

    def has_after(raw: Any) -> bool:
        if _coerce_after(_get(raw, "after", "to_state", "post", "new_state")) is not None:
            return True
        if _coerce_after({
            "machine": _get(raw, "machine_id", "to_machine"),
            "position": _get(raw, "curr_position"),
            "start": _get(raw, "curr_start", "to_start"),
            "end": _get(raw, "curr_end"),
        }) is not None:
            return True
        return any(has_after(c) for c in _children(raw))

    items = causal if isinstance(causal, (list, tuple)) else [causal]
    return any(has_after(it) for it in items)


def run_audit(cases: List[Dict[str, Any]],
              tol: float = 1e-6) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    results = [audit.audit_case(c, tol) for c in cases]
    return audit.aggregate(results), results