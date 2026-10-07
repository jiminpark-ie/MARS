from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from mars_adapters import ExpConfig, Level, MarsBackend, to_jsonable
from mr_metrics import all_code_metrics
from multi_request_generator import build_dataset, dataset_to_records, save_dataset_json
from multirequest_methods import METHODS, evaluate_result, gold_apply, gold_concat
from config import add_auth_args, apply_auth_args_to_env

SAME_LEVELS = [Level.JOB, Level.MACHINE, Level.SCHEDULE]
DatasetKey = Tuple[str, int]


def build_backend(cfg: ExpConfig) -> MarsBackend:
    from mars_backend import build_backend as _build

    return _build(cfg)


def build_datasets(cfg: ExpConfig, rng: np.random.Generator,
                   kinds: Tuple[str, ...] = ("mixed",)) -> Dict[DatasetKey, List]:
    datasets: Dict[DatasetKey, List] = {}
    for c in cfg.c_values:
        if "mixed" in kinds:
            print(f"[stage:data] building mixed c={c} ({cfg.n_per_c} instances)", flush=True)
            datasets[("mixed", c)] = build_dataset(
                rng,
                cfg,
                c,
                kind="mixed",
                progress_label=f"mixed c={c}",
            )
        if "same_level" in kinds:
            same: List = []
            per_level = max(1, cfg.n_per_c // len(SAME_LEVELS))
            sub_cfg = dataclasses.replace(cfg, n_per_c=per_level)
            for lvl in SAME_LEVELS:
                print(
                    f"[stage:data] building same_level c={c} level={lvl.value} "
                    f"({per_level} instances)",
                    flush=True,
                )
                same += build_dataset(
                    rng,
                    sub_cfg,
                    c,
                    kind="same_level",
                    level=lvl,
                    progress_label=f"same_level c={c} level={lvl.value}",
                )
            datasets[("same_level", c)] = same
    return datasets


def validate_gold(instances: List, backend: MarsBackend, cfg: ExpConfig):
    kept, dropped = [], 0
    for inst in instances:
        gold, ok = gold_apply(inst, backend, cfg)
        if ok:
            inst._gold = gold
            inst._gold_concat = gold_concat(inst, cfg)
            kept.append(inst)
        else:
            dropped += 1
    return kept, dropped


def _mean(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else math.nan


def run(cfg, backend, datasets, routing, with_codebert):
    rows = []
    per = defaultdict(lambda: defaultdict(list))
    preds: Dict = defaultdict(list)
    golds: Dict = defaultdict(list)

    print("Starting multi-request evaluation", flush=True)
    for (kind, c), instances in sorted(datasets.items()):
        print(f"dataset={kind} c={c} instances={len(instances)}", flush=True)
        if cfg.evaluate_outcome:
            print(f"validating gold pathway for {kind} c={c}", flush=True)
            instances, dropped = validate_gold(instances, backend, cfg)
            if dropped:
                print(f"Warning: {kind} c={c}: dropped {dropped} instances (gold did not execute)")
        else:
            for inst in instances:
                inst._gold = None
                inst._gold_concat = gold_concat(inst, cfg)

        for method_name in routing.get(kind, []):
            fn = METHODS[method_name]
            total = len(instances) * cfg.n_repeats
            done = 0
            report_every = max(1, total // 10)
            print(f"[eval] {method_name} on {kind} c={c}: 0/{total}", flush=True)
            for inst_idx, inst in enumerate(instances):
                for rep in range(cfg.n_repeats):
                    res = fn(inst, backend, cfg)
                    m = evaluate_result(res, inst, inst._gold, backend, cfg)
                    for k, v in m.items():
                        per[(method_name, kind, c)][k].append(v)
                    preds[(method_name, kind, c)].append(res.gen_code_concat)
                    golds[(method_name, kind, c)].append(inst._gold_concat)
                    rows.append({
                        "method": method_name, "kind": kind, "c": c,
                        "instance_idx": inst_idx, "repeat": rep,
                        "requires_scheduler": inst.requires_scheduler,
                        "combined_nl": inst.combined_nl,
                        "component_types": "|".join(getattr(res, "component_types", [])),
                        "component_levels": "|".join(getattr(res, "component_levels", [])),
                        "subqueries": "\n---\n".join(getattr(res, "subqueries", []))
                                      or inst.combined_nl,
                        "gold_code": inst._gold_concat,
                        "gold_code_components": "\n".join(getattr(res, "component_gold_codes", [])),
                        "pred_code": res.gen_code_concat,
                        "cer": m["cer"], "tsr": m["tsr"],
                        "type_acc": m["type_acc"],
                        "split_count_ok": m["split_count_ok"],
                        "pred_levels": "|".join(getattr(res, "pred_levels", [])),
                        "gen_calls": m["gen_calls"], "llm_calls": m["llm_calls"],
                        "executed": int(res.executed),
                        "prompt": ("\n\n===== NEXT GENERATION =====\n\n"
                                   .join(getattr(res, "prompts", []))),
                    })
                    done += 1
                    if done == total or done % report_every == 0:
                        print(
                            f"Evaluation of {method_name} on {kind} c={c}: {done}/{total}",
                            flush=True,
                        )

    summary = {}
    print("Computing code-similarity metrics", flush=True)
    for key, md in per.items():
        codem = all_code_metrics(preds[key], golds[key], with_codebert=with_codebert)
        summary[key] = {
            **codem,
            "cer": 100.0 * _mean(md["cer"]),
            "tsr": 100.0 * _mean(md["tsr"]),
            "type_acc": 100.0 * _mean(md["type_acc"]),
            "split_count_ok": 100.0 * _mean(md["split_count_ok"]),
            "gen_calls": _mean(md["gen_calls"]),
            "llm_calls": _mean(md["llm_calls"]),
            "n": len(md["cer"]),
        }
    return summary, rows


# reporting


def _block(summary, kind, methods, c_values, metric, label, fmt="{:7.4f}"):
    present = [m for m in methods if any((m, kind, c) in summary for c in c_values)]
    if not present:
        return
    print(f"\n  [{kind}]  {label}")
    print("  " + f"{'method':<26}" + "".join(f"  c={c:>2}" for c in c_values))
    for m in present:
        cells = []
        for c in c_values:
            v = summary.get((m, kind, c))
            cells.append("     - " if v is None or (isinstance(v[metric], float) and math.isnan(v[metric]))
                         else fmt.format(v[metric]))
        print("  " + f"{m:<26}" + "".join(f"  {x}" for x in cells))


def print_tables(summary, cfg, routing):
    c_values = list(cfg.c_values)
    print("\n" + "=" * 72)
    print("MULTI-REQUEST RESULTS  (all four methods)")
    print("=" * 72)
    for kind in ("mixed", "same_level"):
        methods = routing.get(kind, [])
        _block(summary, kind, methods, c_values, "chrf", "chrF++")
        _block(summary, kind, methods, c_values, "cer", "CER (%)")
        _block(summary, kind, methods, c_values, "type_acc", "Type-classification acc (%)")
        if cfg.evaluate_outcome:
            _block(summary, kind, methods, c_values, "tsr", "TSR (%)")
        _block(summary, kind, methods, c_values, "gen_calls",
               "mean generation calls / query", fmt="{:7.2f}")
    print()


def save(summary, prefix):
    def round_or_none(value, digits):
        if value is None:
            return None
        try:
            if math.isnan(value) or math.isinf(value):
                return None
        except TypeError:
            pass
        return round(value, digits)

    rows = []
    for (method, kind, c), v in sorted(summary.items()):
        rows.append({
            "method": method, "kind": kind, "c": c,
            "chrf": round_or_none(v["chrf"], 3),
            "cer_pct": round_or_none(v["cer"], 2),
            "tsr_pct": round_or_none(v["tsr"], 2),
            "type_acc_pct": round_or_none(v["type_acc"], 2),
            "n": v["n"],
        })
    if not rows:
        print("no results to save")
        return
    parent = os.path.dirname(os.path.abspath(prefix))
    if parent:
        os.makedirs(parent, exist_ok=True)
    csv_path = f"{prefix}.csv"
    json_path = f"{prefix}.json"
    tmp_csv = f"{csv_path}.tmp"
    tmp_json = f"{json_path}.tmp"
    with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp_csv, csv_path)
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(
            to_jsonable({f"{m}|{k}|{c}": v for (m, k, c), v in summary.items()}),
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        f.write("\n")
    os.replace(tmp_json, json_path)
    print(f"saved: {csv_path}  and  {json_path}", flush=True)


def save_datasets_json(datasets, cfg, prefix, include_data0=False):
    parent = os.path.dirname(os.path.abspath(prefix))
    if parent:
        os.makedirs(parent, exist_ok=True)
    all_records = []
    print("[stage:save-data] saving generated datasets before evaluation", flush=True)
    for (kind, c), instances in sorted(datasets.items()):
        path = f"{prefix}_{kind}_c{c}.json"
        save_dataset_json(
            instances,
            path,
            cfg=cfg,
            include_components=True,
            include_data0=include_data0,
        )
        all_records.extend(
            dataset_to_records(
                instances,
                cfg=cfg,
                include_components=True,
                include_data0=include_data0,
            )
        )
        print(f"[save-data] saved {len(instances)} records: {path}", flush=True)

    all_path = f"{prefix}_all.json"
    tmp_all_path = f"{all_path}.tmp"
    with open(tmp_all_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(all_records), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(tmp_all_path, all_path)
    print(f"[save-data] saved {len(all_records)} records: {all_path}", flush=True)


# main


def save_results_rows(rows, prefix):
    if not rows:
        print("[save-results] no per-result rows to save")
        return
    cols = list(rows[0].keys())
    parent = os.path.dirname(os.path.abspath(prefix))
    if parent:
        os.makedirs(parent, exist_ok=True)
    csv_path = f"{prefix}_results.csv"
    tmp = csv_path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, csv_path)
    print(f"[save-results] wrote {len(rows)} rows -> {csv_path}", flush=True)
    xlsx_path = f"{prefix}_results.xlsx"
    try:
        from openpyxl import Workbook
        wb = Workbook(); ws = wb.active; ws.title = "results"
        ws.append(cols)
        for r in rows:
            ws.append([_xlsx_cell(r.get(k, "")) for k in cols])
        wb.save(xlsx_path)
        print(f"[save-results] wrote {len(rows)} rows -> {xlsx_path}", flush=True)
    except ImportError:
        print("[save-results] openpyxl not installed; skipped XLSX (CSV written). "
              "pip install openpyxl to enable.", flush=True)


def _xlsx_cell(v):
    if isinstance(v, (int, float)):
        return v
    sv = "" if v is None else str(v)
    return sv if len(sv) <= 32767 else sv[:32760] + "...[cut]"


def _assert_full_icl(backend):
    try:
        sizes = {k: len(backend._examples(k)) for k in ("icl_job", "icl_machine", "icl_schedule")}
    except Exception as e:  # noqa: BLE001
        print(f"[icl-check] could not load ICL pools: {e!r}", flush=True)
        return
    total = sum(sizes.values())
    got = len(backend.get_fewshot_all())
    status = "OK (no truncation)" if got == total else "MISMATCH"
    print(f"[icl-check] pool sizes job={sizes['icl_job']} machine={sizes['icl_machine']} "
          f"schedule={sizes['icl_schedule']} -> get_fewshot_all={got} / expected={total} "
          f"[{status}]", flush=True)
    if got != total:
        print("[icl-check] WARNING: few-shot count != full pool size; ICL is being "
              "truncated somewhere -- check get_fewshot_all / _examples.", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    add_auth_args(p)
    p.add_argument("--no-outcome", action="store_true")
    p.add_argument("--evaluate-outcome", action="store_true")
    p.add_argument("--paraphrase", action="store_true")
    p.add_argument("--no-codebert", action="store_true", help="skip CodeBERTScore (no model download)")
    p.add_argument("--n-per-c", type=int, default=200)
    p.add_argument("--c", type=int, nargs="+", default=[4, 5])
    p.add_argument("--n-repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260530)
    p.add_argument("--n-machines", type=int, default=10)
    p.add_argument("--n-jobs", type=int, default=100)
    p.add_argument("--mixed-shot-strategy", choices=["all_levels", "generic"], default="all_levels")
    p.add_argument("--machine-forces-regen", action="store_true")
    p.add_argument("--mixed-methods", default="monolithic,per_request,per_level")
    p.add_argument("--same-methods", default="",
                   help="methods for same-level datasets; empty (default) skips them")
    p.add_argument("--out-prefix", default="multirequest_results")
    p.add_argument("--save-datasets", action="store_true")
    p.add_argument("--skip-save-datasets", action="store_true")
    p.add_argument("--dataset-prefix", default=None)
    p.add_argument("--include-data0", action="store_true")
    p.add_argument("--base-model", default=None)
    p.add_argument("--adapter-dir", default=None)
    p.add_argument("--icl-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    apply_auth_args_to_env(args)
    if args.base_model:
        os.environ["MARS_BASE_MODEL"] = args.base_model
    if args.adapter_dir:
        os.environ["MARS_ADAPTER_DIR"] = args.adapter_dir
    if args.icl_dir:
        os.environ["MARS_ICL_DIR"] = args.icl_dir
    cfg = ExpConfig(
        c_values=tuple(args.c),
        n_per_c=args.n_per_c,
        n_repeats=args.n_repeats,
        n_machines=args.n_machines,
        n_jobs=args.n_jobs,
        seed=args.seed,
        mixed_shot_strategy=args.mixed_shot_strategy,
        machine_forces_regen=args.machine_forces_regen,
        evaluate_outcome=not args.no_outcome,
        paraphrase=args.paraphrase,
    )
    rng = np.random.default_rng(cfg.seed)

    routing = {
        "mixed": [m for m in args.mixed_methods.split(",") if m],
        "same_level": [m for m in args.same_methods.split(",") if m],
    }
    kinds = tuple(k for k, ms in routing.items() if ms)

    print("[stage:data] generating all datasets before any model evaluation", flush=True)
    datasets = build_datasets(cfg, rng, kinds=kinds)
    for (kind, c), ds in sorted(datasets.items()):
        print(f"[data] complete {kind:<11} c={c}: {len(ds)} instances", flush=True)

    if not args.skip_save_datasets:
        save_datasets_json(
            datasets,
            cfg,
            prefix=args.dataset_prefix or args.out_prefix,
            include_data0=args.include_data0,
        )

    print("[stage:backend] loading MARS backend/model", flush=True)
    backend = build_backend(cfg)

    _assert_full_icl(backend)

    print(f"[stage:eval] routing={routing}", flush=True)
    summary, rows = run(cfg, backend, datasets, routing, with_codebert=not args.no_codebert)
    print_tables(summary, cfg, routing)
    save(summary, args.out_prefix)
    save_results_rows(rows, args.out_prefix)


if __name__ == "__main__":
    main()
