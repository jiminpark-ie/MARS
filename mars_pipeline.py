from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sim.repair_fixed import (
    ScheduleRepairer,
    calculate_timing_from_sequence,
    compress_causal_chain,
    compute_schedule_metrics,
    compute_stability_metrics,
    normalize_schedule_to_sequence,
)
from sim.schedulers import DEFAULT_RULE, SCHEDULERS, run_named_scheduler
from explainer import call_gpt_explainer, change_set, score_faithfulness
from mars_adapters import ExpConfig, Level, MarsBackend
from request_validator import parse_ops, validate_requests


@dataclass
class PipelineResult:
    query: str
    status: str
    requires_scheduler: int = 0
    levels: List[str] = field(default_factory=list)
    subqueries: List[str] = field(default_factory=list)
    codes: List[str] = field(default_factory=list)
    validation_messages: List[str] = field(default_factory=list)
    schedule_before: Optional[Dict[int, list]] = None
    schedule_after: Optional[Dict[int, list]] = None
    causal_chain: Optional[Dict[str, Any]] = None
    causal_chain_compressed: Optional[Dict[str, Any]] = None
    metrics_before: Optional[Dict[str, Any]] = None
    metrics_after: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    audit: Optional[Dict[str, Any]] = None
    faithfulness: Optional[Dict[str, Any]] = None


class MarsPipeline:

    def __init__(
        self,
        backend: MarsBackend,
        explain: bool = True,
        validate: bool = True,
        audit: bool = True,
        audit_tol: float = 1e-6,
        compress_max_jobs: int = 40,
        compress_max_depth: int = 10,
        scheduler: str = DEFAULT_RULE,
    ):
        self.backend = backend
        self.explain = explain
        self.validate = validate
        self.audit = audit
        self.audit_tol = audit_tol
        self.compress_max_jobs = compress_max_jobs
        self.compress_max_depth = compress_max_depth
        self.scheduler = scheduler


    def run_single(self, query: str, data: Dict[str, Any]) -> PipelineResult:
        rerun = self.backend.classify_rerun(query, data)
        level = self.backend.classify_level(query, data)
        fewshot = self.backend.get_fewshot(level=level)
        code = self.backend.generate_code(query, data, fewshot)
        return self._apply_and_explain(
            query=query,
            data=data,
            items=[{"nl": query, "level": level, "code": code}],
            rerun=rerun,
        )

    def run_multi(self, combined_query: str, data: Dict[str, Any]) -> PipelineResult:
        rerun = self.backend.classify_rerun(combined_query, data)
        atomic = self.backend.split_requests(combined_query)
        items: List[Dict[str, Any]] = []
        for nl in atomic:
            level = self.backend.classify_level(nl, data)
            fewshot = self.backend.get_fewshot(level=level)
            code = self.backend.generate_code(nl, data, fewshot)
            items.append({"nl": nl, "level": level, "code": code})
        return self._apply_and_explain(
            query=combined_query, data=data, items=items, rerun=rerun,
        )

    def _apply_and_explain(
        self,
        query: str,
        data: Dict[str, Any],
        items: List[Dict[str, Any]],
        rerun: int,
    ) -> PipelineResult:
        res = PipelineResult(
            query=query,
            status="ok",
            requires_scheduler=int(rerun),
            levels=[it["level"].value for it in items],
            subqueries=[it["nl"] for it in items],
            codes=[it["code"] for it in items],
        )

        prev_seq = normalize_schedule_to_sequence(data.get("schedule", {}))
        if not prev_seq:
            raise ValueError("data['schedule'] is required (run the scheduler first)")
        prev_timed = calculate_timing_from_sequence(data, prev_seq)
        res.schedule_before = prev_timed
        res.metrics_before = compute_schedule_metrics(data, prev_timed)

        codes = [it["code"] for it in items]
        if self.validate:
            report = validate_requests(codes, data, prev_seq)
            res.validation_messages = [
                f"[{e.severity.upper()}] {e.message}" for e in report.errors
            ]
            if not report.ok:
                res.status = "rejected"
                return res

        instance_items = [it for it in items
                          if it["level"] in (Level.JOB, Level.MACHINE)]
        schedule_items = [it for it in items if it["level"] == Level.SCHEDULE]
        order = {Level.JOB: 0, Level.MACHINE: 1}
        instance_items.sort(key=lambda it: order[it["level"]])

        working = copy.deepcopy(data)
        for it in instance_items:
            working = self.backend.exec_code(it["code"], working)
        if rerun:
            working["schedule"] = run_named_scheduler(working, self.scheduler)
        for it in schedule_items:
            working = self.backend.exec_code(it["code"], working)

        new_seq = normalize_schedule_to_sequence(working.get("schedule", {}))

        ops = parse_ops(codes)
        direct_ids = sorted(ops.jobs)

        repairer = ScheduleRepairer(
            data=working,
            prev_schedule=prev_seq,
            modified_schedule=new_seq,
            prev_data=data,
            modification_info={
                "user_query": query,
                "modification_type": "+".join(res.levels),
                "direct_job_ids": direct_ids,
            },
            sequence_directly_edited=(not rerun and bool(schedule_items)),
        )
        timed_after = repairer.repair()
        if not repairer.converged:
            res.status = "infeasible"
            res.validation_messages.append(
                "[ERROR] semi-active repair did not converge; the requested "
                "sequence and precedence constraints are inconsistent."
            )
            return res

        chain = repairer.causal_chain
        chain.modification_method = "rerun" if rerun else "repair"
        res.schedule_after = timed_after
        res.causal_chain = chain.to_dict()
        res.causal_chain_compressed = compress_causal_chain(
            res.causal_chain,
            max_affected_jobs=self.compress_max_jobs,
            max_depth=self.compress_max_depth,
        )
        res.metrics_after = compute_schedule_metrics(working, timed_after)
        res.metrics_after.update(compute_stability_metrics(prev_timed, timed_after))

        if self.explain:
            res.explanation = call_gpt_explainer(
                {"user_query": query,
                 "modification_type": "+".join(res.levels),
                 "direct_job_ids": direct_ids},
                res.causal_chain_compressed,
                res.metrics_before,
                res.metrics_after,
            )

        if self.audit:
            from causal_chain_audit import audit_case
            from causal_chain_bridge import to_audit_case

            case, _derived = to_audit_case(
                "pipeline",
                prev_timed,
                timed_after,
                res.causal_chain,
                data=working,
            )
            res.audit = audit_case(case, self.audit_tol)
            if res.explanation is not None:
                cs = change_set(data, prev_seq, new_seq, data_after=working)
                res.faithfulness = score_faithfulness(res.explanation, cs)

        return res


def _demo_instance(n_m: int, n_j: int, seed: int,
                   scheduler: str = DEFAULT_RULE) -> Dict[str, Any]:
    import numpy as np
    import random as _random

    from explainer import generate_random_instance

    _random.seed(seed)
    np.random.seed(seed)
    data = generate_random_instance(n_m=n_m, n_j=n_j)
    data["schedule"] = run_named_scheduler(data, scheduler)
    return data


def main():
    import argparse

    from mars_backend import build_backend
    from config import add_auth_args, apply_auth_args_to_env

    parser = argparse.ArgumentParser()
    add_auth_args(parser)
    parser.add_argument("--query", default=None)
    parser.add_argument("--scheduler", choices=list(SCHEDULERS), default=DEFAULT_RULE)
    parser.add_argument("--multi", action="store_true")
    parser.add_argument("--n-machines", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-explain", action="store_true")
    parser.add_argument("--no-audit", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    apply_auth_args_to_env(args)

    data = _demo_instance(args.n_machines, args.n_jobs, args.seed,
                          scheduler=args.scheduler)
    backend = build_backend(ExpConfig())
    pipeline = MarsPipeline(
        backend,
        explain=not args.no_explain,
        audit=not args.no_audit,
        scheduler=args.scheduler,
    )

    run = pipeline.run_multi if args.multi else pipeline.run_single
    result = run(args.query, data)

    print(f"status: {result.status}")
    print(f"requires_scheduler: {result.requires_scheduler}")
    print(f"levels: {result.levels}")
    for i, code in enumerate(result.codes):
        print(f"--- generated code [{i}] ---\n{code}")
    for msg in result.validation_messages:
        print(msg)
    if result.causal_chain is not None:
        print(f"affected jobs: {result.causal_chain['total_affected_jobs']}, "
              f"max depth: {result.causal_chain['max_depth']}")
    if result.audit is not None:
        print(f"audit: structure={result.audit['structure_ok']} "
              f"coverage={result.audit['coverage_ok']} "
              f"soundness={result.audit['soundness_ok']} "
              f"replay={result.audit['replay_ok']}")
    if result.faithfulness is not None:
        print(f"explanation faithfulness: {result.faithfulness}")
    if result.explanation is not None:
        print(f"--- explanation ---\n{result.explanation}")

    if args.out:
        def _default(o):
            try:
                import numpy as np
                if isinstance(o, np.ndarray):
                    return o.tolist()
                if isinstance(o, np.generic):
                    return o.item()
            except Exception:
                pass
            return str(o)

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result.__dict__, f, indent=2, default=_default)
        print(f"result written to {args.out}")


if __name__ == "__main__":
    main()
