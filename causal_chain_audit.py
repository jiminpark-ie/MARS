from __future__ import annotations
import argparse
import json
import math
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

Assignment = Tuple[int, int, float, float]


def extract_assignments(schedule: Dict[Any, List[Dict[str, Any]]]) -> Dict[int, Assignment]:

    out: Dict[int, Assignment] = {}
    for machine_key, ops in schedule.items():
        machine = int(machine_key)
        for position, op in enumerate(ops):
            job = int(op["job"])
            if job in out:
                raise ValueError(f"job {job} appears twice in one schedule")
            out[job] = (machine, position, float(op["start"]), float(op["end"]))
    return out


def iter_chain_nodes(chain: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    
    for raw in chain:
        after_raw = raw.get("after")
        after: Optional[Assignment] = None
        if after_raw is not None:
            after = (
                int(after_raw["machine"]),
                int(after_raw["position"]),
                float(after_raw["start"]),
                float(after_raw["end"]),
            )
        yield {
            "node_id": int(raw["node_id"]),
            "parent_id": None if raw.get("parent_id") is None else int(raw["parent_id"]),
            "depth": int(raw["depth"]),
            "job": None if raw.get("job") is None else int(raw["job"]),
            "after": after,
        }


def _close(a: float, b: float, tol: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def _same_assignment(x: Assignment, y: Assignment, tol: float) -> bool:
    return x[0] == y[0] and x[1] == y[1] and _close(x[2], y[2], tol) and _close(x[3], y[3], tol)


def changed_jobs(orig: Dict[int, Assignment], mod: Dict[int, Assignment], tol: float) -> List[int]:
    jobs = set(orig) | set(mod)
    out = []
    for j in sorted(jobs):
        a, b = orig.get(j), mod.get(j)
        if a is None or b is None or not _same_assignment(a, b, tol):
            out.append(j)
    return out


def audit_case(case: Dict[str, Any], tol: float) -> Dict[str, Any]:

    orig = extract_assignments(case["original_schedule"])
    mod = extract_assignments(case["modified_schedule"])
    nodes = list(iter_chain_nodes(case["causal_chain"]))

    res: Dict[str, Any] = {"case_id": case.get("case_id", "?"), "errors": []}
    err = res["errors"].append

    ids = [n["node_id"] for n in nodes]
    by_id = {n["node_id"]: n for n in nodes}
    structure_ok = True
    if len(ids) != len(set(ids)):
        structure_ok = False
        err("duplicate node_id")
    for n in nodes:
        if n["parent_id"] is None:
            if n["depth"] != 0:
                structure_ok = False
                err(f"root node {n['node_id']} has depth {n['depth']} != 0")
        else:
            parent = by_id.get(n["parent_id"])
            if parent is None:
                structure_ok = False
                err(f"node {n['node_id']} references missing parent {n['parent_id']}")
            elif n["depth"] != parent["depth"] + 1:
                structure_ok = False
                err(f"node {n['node_id']} depth {n['depth']} != parent depth {parent['depth']}+1")
    job_claims: Dict[int, int] = {}
    for n in nodes:
        if n["job"] is None:
            continue
        if n["job"] in job_claims:
            structure_ok = False
            err(f"job {n['job']} claimed by nodes {job_claims[n['job']]} and {n['node_id']}")
        job_claims[n["job"]] = n["node_id"]
    res["structure_ok"] = structure_ok

    actual_changed = changed_jobs(orig, mod, tol)
    claimed = set(job_claims)
    missing = [j for j in actual_changed if j not in claimed]
    res["n_changed"] = len(actual_changed)
    res["n_covered"] = len(actual_changed) - len(missing)
    res["coverage_ok"] = not missing
    if missing:
        err(f"coverage: changed jobs missing from chain: {missing}")

    change_nodes = [n for n in nodes if n["job"] is not None]
    unsound = 0
    for n in change_nodes:
        j = n["job"]
        if j not in set(actual_changed):
            unsound += 1
            err(f"soundness: node {n['node_id']} claims unchanged/unknown job {j}")
            continue
        if n["after"] is not None:
            truth = mod.get(j)
            if truth is None or not _same_assignment(n["after"], truth, tol):
                unsound += 1
                err(
                    f"soundness: node {n['node_id']} after-state {n['after']} "
                    f"!= modified schedule {truth} for job {j}"
                )
    res["n_change_nodes"] = len(change_nodes)
    res["n_sound_nodes"] = len(change_nodes) - unsound
    res["soundness_ok"] = unsound == 0

    replay_ok: Optional[bool] = None
    if all(n["after"] is not None for n in change_nodes):
        replayed = dict(orig)
        for n in change_nodes:
            replayed[n["job"]] = n["after"]
        jobs = set(replayed) | set(mod)
        replay_ok = all(
            j in replayed and j in mod and _same_assignment(replayed[j], mod[j], tol)
            for j in jobs
        )
        if not replay_ok:
            bad = [
                j for j in sorted(jobs)
                if (j not in replayed or j not in mod
                    or not _same_assignment(replayed[j], mod[j], tol))
            ]
            err(f"replay: overlaying chain on original != modified for jobs {bad}")
    else:
        err("replay: skipped (some change nodes lack an 'after' record)")
    res["replay_ok"] = replay_ok

    res["all_ok"] = bool(
        res["structure_ok"] and res["coverage_ok"] and res["soundness_ok"]
        and (replay_ok is not False)
    )
    return res


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    pct = lambda k: 100.0 * sum(1 for r in results if r[k]) / n if n else float("nan")
    cov_pairs = [(r["n_covered"], r["n_changed"]) for r in results if r["n_changed"]]
    snd_pairs = [(r["n_sound_nodes"], r["n_change_nodes"]) for r in results if r["n_change_nodes"]]
    job_cov = 100.0 * sum(a for a, _ in cov_pairs) / max(1, sum(b for _, b in cov_pairs))
    node_snd = 100.0 * sum(a for a, _ in snd_pairs) / max(1, sum(b for _, b in snd_pairs))
    replay_eval = [r for r in results if r["replay_ok"] is not None]
    return {
        "n_cases": n,
        "case_pass_rate_pct": pct("all_ok"),
        "structure_case_pct": pct("structure_ok"),
        "coverage_case_pct": pct("coverage_ok"),
        "coverage_job_pct": job_cov,
        "soundness_case_pct": pct("soundness_ok"),
        "soundness_node_pct": node_snd,
        "replay_case_pct": (
            100.0 * sum(1 for r in replay_eval if r["replay_ok"]) / len(replay_eval)
            if replay_eval else None
        ),
        "n_replay_evaluable": len(replay_eval),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="?", help="JSONL file of audit cases")
    ap.add_argument("--report", help="write per-case + aggregate results to this JSON file")
    ap.add_argument("--tol", type=float, default=1e-6, help="absolute tolerance for time values")
    ap.add_argument("-v", "--verbose", action="store_true", help="print per-case errors")
    args = ap.parse_args(argv)

    if args.jsonl:
        def _read():
            with open(args.jsonl, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        cases = _read()
    else:
        ap.error("provide a JSONL file")

    results = [audit_case(c, args.tol) for c in cases]
    agg = aggregate(results)

    for r in results:
        flag = "PASS" if r["all_ok"] else "FAIL"
        print(f"[{flag}] {r['case_id']}: structure={r['structure_ok']} "
              f"coverage={r['coverage_ok']} soundness={r['soundness_ok']} replay={r['replay_ok']}")
        if args.verbose or (not r["all_ok"]):
            for e in r["errors"]:
                print(f"Error: {e}")

    print("\n")
    print("="*60)
    print(f"cases audited: {agg['n_cases']}")
    print(f"coverage (case-level): {agg['coverage_case_pct']:.2f}%")
    print(f"coverage (changed-job level) : {agg['coverage_job_pct']:.2f}%")
    print(f"soundness (case-level): {agg['soundness_case_pct']:.2f}%")
    print(f"soundness (node level): {agg['soundness_node_pct']:.2f}%")
    rp = agg["replay_case_pct"]
    print(f"replay (case-level): "
          f"{'(no after-states)' if rp is None else f'{rp:.2f}%'}  "
          f"(evaluable cases: {agg['n_replay_evaluable']})")
    print(f"all checks passed: {agg['case_pass_rate_pct']:.2f}%")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"aggregate": agg, "cases": results}, f, indent=2)
        print(f"\nreport written to {args.report}")

    return 0 if all(r["all_ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())