from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

ERROR = "error"
WARNING = "warning"


@dataclass
class ValidationError:
    code: str
    severity: str
    message: str
    request_index: Optional[int] = None


@dataclass
class ValidationReport:
    errors: List[ValidationError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(e.severity == ERROR for e in self.errors)

    def messages(self, severity: Optional[str] = None) -> List[str]:
        return [e.message for e in self.errors
                if severity is None or e.severity == severity]


_ALLOWED_METHODS = {"index", "insert", "pop", "append"}
_ALLOWED_CALLS = {"float", "int"}
_FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
    ast.ClassDef, ast.Lambda, ast.For, ast.AsyncFor, ast.While, ast.With,
    ast.AsyncWith, ast.Try, ast.Global, ast.Nonlocal, ast.Delete, ast.Raise,
    ast.Assert, ast.Import, ast.ListComp, ast.SetComp, ast.DictComp,
    ast.GeneratorExp, ast.Await, ast.Yield, ast.YieldFrom, ast.Starred,
)

_FORBIDDEN_NAMES = {
    "eval", "exec", "open", "compile", "__import__", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "input", "exit", "quit",
    "breakpoint", "memoryview", "os", "sys", "subprocess", "importlib",
}


def validate_code_safety(code: str, request_index: Optional[int] = None
                         ) -> List[ValidationError]:
    errs: List[ValidationError] = []
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return [ValidationError("SYNTAX_ERROR", ERROR,
                                f"generated code does not parse: {e}", request_index)]

    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            errs.append(ValidationError(
                "FORBIDDEN_CONSTRUCT", ERROR,
                f"disallowed construct {type(node).__name__} in modification code",
                request_index))

        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr not in _ALLOWED_METHODS:
                errs.append(ValidationError(
                    "FORBIDDEN_ATTRIBUTE", ERROR,
                    f"disallowed attribute access '.{node.attr}'", request_index))
                
        if isinstance(node, ast.Call):
            is_method = (isinstance(node.func, ast.Attribute)
                         and node.func.attr in _ALLOWED_METHODS)
            is_cast = (isinstance(node.func, ast.Name)
                       and node.func.id in _ALLOWED_CALLS)
            if not (is_method or is_cast):
                errs.append(ValidationError(
                    "FORBIDDEN_CALL", ERROR,
                    "only list .index/.insert/.pop/.append and float()/int() are allowed",
                    request_index))

        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            errs.append(ValidationError(
                "FORBIDDEN_NAME", ERROR,
                f"disallowed name '{node.id}'", request_index))
    return errs


@dataclass
class ParsedOps:
    jobs: Set[int] = field(default_factory=set)
    machines: Set[int] = field(default_factory=set)
    new_precedences: List[Tuple[int, int]] = field(default_factory=list)
    restricted_machines: Set[int] = field(default_factory=set)
    fixed_assignments: List[Tuple[int, int]] = field(default_factory=list)
    moves: List[Dict[str, Any]] = field(default_factory=list)
    swaps: List[Dict[str, Any]] = field(default_factory=list)


def _const_int(node: ast.AST) -> Optional[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _const_int(node.operand)
        return -inner if inner is not None else None
    return None


def _subscript_chain(node: ast.AST) -> Optional[Tuple[Union[str, int], ...]]:
    levels: List[List[Union[str, int]]] = []
    cur = node
    while isinstance(cur, ast.Subscript):
        sl = cur.slice
        lvl: List[Union[str, int]] = []
        if isinstance(sl, ast.Tuple):
            for elt in sl.elts:
                if isinstance(elt, ast.Slice):
                    lvl.append(":")
                else:
                    iv = _const_int(elt)
                    lvl.append(iv if iv is not None else "?")
        elif isinstance(sl, ast.Slice):
            lvl.append(":")
        else:
            iv = _const_int(sl)
            if iv is not None:
                lvl.append(iv)
            elif isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                lvl.append(sl.value)
            else:
                lvl.append("?")
        levels.append(lvl)
        cur = cur.value
    if not (isinstance(cur, ast.Name) and cur.id == "data"):
        return None
    levels.reverse()
    flat = [k for lvl in levels for k in lvl]
    return tuple(flat)


class _OpExtractor(ast.NodeVisitor):

    def __init__(self) -> None:
        self.ops = ParsedOps()
        self.alias_to_machine: Dict[str, int] = {}
        self._pending_fix_clear: Set[int] = set()

    def _note_job(self, j: Any) -> None:
        if isinstance(j, int):
            self.ops.jobs.add(j)

    def _note_machine(self, m: Any) -> None:
        if isinstance(m, int):
            self.ops.machines.add(m)

    def _index_arg_job(self, call: ast.Call) -> Optional[int]:
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == "index" and call.args):
            return _const_int(call.args[0])
        return None

    def visit_Assign(self, node: ast.Assign) -> None:
        target = node.targets[0]

        chain = _subscript_chain(node.value)
        if (isinstance(target, ast.Name) and chain and len(chain) == 2
                and chain[0] == "schedule" and isinstance(chain[1], int)):
            self.alias_to_machine[target.id] = chain[1]
            self._note_machine(chain[1])

        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) \
                and node.value.func.attr == "pop":
            base = node.value.func.value
            if isinstance(base, ast.Name) and base.id in self.alias_to_machine:
                m = self.alias_to_machine[base.id]
                jarg = self._index_arg_job(node.value.args[0]) if node.value.args else None
                if jarg is not None:
                    self._note_job(jarg)
                    self.ops.moves.append({"job": jarg, "from_machine": m})

        tchain = _subscript_chain(target)
        if tchain and tchain[0] == "machine_eligibility":
            rhs = _const_int(node.value)
            if len(tchain) == 3 and tchain[2] == ":" and isinstance(tchain[1], int) and rhs == 0:
                self.ops.restricted_machines.add(tchain[1])
                self._note_machine(tchain[1])
            elif len(tchain) == 3 and tchain[1] == ":" and isinstance(tchain[2], int) and rhs == 0:
                self._pending_fix_clear.add(tchain[2])
                self._note_job(tchain[2])
            elif len(tchain) == 3 and isinstance(tchain[1], int) and isinstance(tchain[2], int) and rhs == 1:
                self.ops.fixed_assignments.append((tchain[2], tchain[1]))
                self._note_job(tchain[2]); self._note_machine(tchain[1])

        if tchain and tchain[0] == "precedence" and len(tchain) == 3 and isinstance(tchain[1], int) and isinstance(tchain[2], int):
            self.ops.new_precedences.append((tchain[1], tchain[2]))
            self._note_job(tchain[1]); self._note_job(tchain[2])

        if tchain and tchain[0] in {"due", "ready", "priority"} and len(tchain) == 2 and isinstance(tchain[1], int):
            self._note_job(tchain[1])
        if tchain and tchain[0] == "processing_time" and len(tchain) == 3:
            if isinstance(tchain[1], int):
                self._note_machine(tchain[1])
            if isinstance(tchain[2], int):
                self._note_job(tchain[2])
        if tchain and tchain[0] == "setup_time" and len(tchain) == 4:
            if isinstance(tchain[1], int):
                self._note_machine(tchain[1])
            for k in (tchain[2], tchain[3]):
                if isinstance(k, int):
                    self._note_job(k)

        if isinstance(target, ast.Tuple):
            machines_in_swap, jobs_unknown = set(), True
            for elt in target.elts:
                if isinstance(elt, ast.Subscript) and isinstance(elt.value, ast.Name) and elt.value.id in self.alias_to_machine:
                    machines_in_swap.add(self.alias_to_machine[elt.value.id])
            if machines_in_swap:
                self.ops.swaps.append({"machines": sorted(machines_in_swap), "jobs": []})

        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        call = node.value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
            base = call.func.value
            if isinstance(base, ast.Name) and base.id in self.alias_to_machine:
                m = self.alias_to_machine[base.id]
                if call.func.attr == "insert":
                    self.ops.moves.append({"to_machine": m})
        self.generic_visit(node)


def parse_ops(codes: Union[str, Sequence[str]]) -> ParsedOps:
    if isinstance(codes, str):
        codes = [codes]
    merged = ParsedOps()
    for code in codes:
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError:
            continue
        ex = _OpExtractor()
        ex.visit(tree)
        o = ex.ops
        merged.jobs |= o.jobs
        merged.machines |= o.machines
        merged.new_precedences += o.new_precedences
        merged.restricted_machines |= o.restricted_machines
        merged.fixed_assignments += o.fixed_assignments
        merged.moves += o.moves
        merged.swaps += o.swaps
    return merged


def _dims(data: Dict[str, Any], env_params: Optional[Dict[str, int]]) -> Tuple[int, int]:
    if "processing_time" in data and hasattr(data["processing_time"], "shape"):
        n_m, n_j = data["processing_time"].shape
        return int(n_m), int(n_j)
    if "machine_eligibility" in data and hasattr(data["machine_eligibility"], "shape"):
        n_m, n_j = data["machine_eligibility"].shape
        return int(n_m), int(n_j)
    if env_params:
        return int(env_params["n_m"]), int(env_params["n_j"])
    raise ValueError("cannot determine (n_m, n_j) from data or env_params")


def _existing_precedences(data: Dict[str, Any]) -> List[Tuple[int, int]]:

    P = data.get("precedence")
    if P is None:
        return []
    edges = []
    n = P.shape[0] if hasattr(P, "shape") else len(P)
    for i in range(n):
        for j in range(n):
            if int(P[i][j]) == 1:
                edges.append((i, j))
    return edges


def _machine_of_job_in_schedule(schedule: Dict[int, List[int]], job: int) -> Optional[int]:
    for m, seq in schedule.items():
        if job in seq:
            return int(m)
    return None


def _has_cycle(edges: List[Tuple[int, int]]) -> Optional[List[int]]:
    from collections import defaultdict
    g = defaultdict(list)
    nodes = set()
    for i, j in edges:
        g[i].append(j)
        nodes.add(i); nodes.add(j)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}
    stack_path: List[int] = []

    def dfs(u: int) -> Optional[List[int]]:
        color[u] = GRAY
        stack_path.append(u)
        for v in g[u]:
            if color[v] == GRAY:
                return stack_path[stack_path.index(v):] + [v]
            if color[v] == WHITE:
                c = dfs(v)
                if c:
                    return c
        color[u] = BLACK
        stack_path.pop()
        return None

    for n in nodes:
        if color[n] == WHITE:
            c = dfs(n)
            if c:
                return c
    return None

def validate_requests(
    codes: Union[str, Sequence[str]],
    data: Dict[str, Any],
    schedule: Optional[Dict[int, List[int]]] = None,
    env_params: Optional[Dict[str, int]] = None,
    check_safety: bool = True,
) -> ValidationReport:
    if isinstance(codes, str):
        codes = [codes]
    report = ValidationReport()

    if check_safety:
        for idx, code in enumerate(codes):
            report.errors.extend(validate_code_safety(code, request_index=idx))
        if not report.ok:
            return report

    ops = parse_ops(codes)
    n_m, n_j = _dims(data, env_params)
    schedule = schedule if schedule is not None else data.get("schedule", {})
    elig = data.get("machine_eligibility")
    existing = _existing_precedences(data)

    def add(code, sev, msg):
        report.errors.append(ValidationError(code, sev, msg))

    for j in sorted(ops.jobs):
        if not (0 <= j < n_j):
            add("JOB_OUT_OF_RANGE", ERROR, f"job {j} does not exist (0..{n_j - 1}).")
    for m in sorted(ops.machines):
        if not (0 <= m < n_m):
            add("MACHINE_OUT_OF_RANGE", ERROR, f"machine {m} does not exist (0..{n_m - 1}).")

    existing_set = set(existing)
    for (i, j) in ops.new_precedences:
        if (j, i) in existing_set:
            add("PRECEDENCE_CONTRADICTION", ERROR,
                f"requested precedence {i}->{j} contradicts existing {j}->{i}.")
        if i == j:
            add("PRECEDENCE_SELF", ERROR, f"job {i} cannot precede itself.")
    cyc = _has_cycle(existing + [pc for pc in ops.new_precedences])
    if cyc:
        add("PRECEDENCE_CYCLE", ERROR,
            "requested precedence introduces a cycle: "
            + " -> ".join(map(str, cyc)) + ".")

    for (job, m) in ops.fixed_assignments:
        if m in ops.restricted_machines:
            add("FIX_TO_RESTRICTED", ERROR,
                f"job {job} is fixed to machine {m}, which is also made unavailable.")
        elif elig is not None and 0 <= m < n_m and 0 <= job < n_j and int(elig[m][job]) == 0:
            add("FIX_TO_INELIGIBLE", WARNING,
                f"job {job} is fixed to machine {m}, which is not currently eligible "
                f"to process it (the edit forces eligibility; verify this is intended).")

    if elig is not None:
        for m in sorted(ops.restricted_machines):
            if not (0 <= m < n_m):
                continue
            for j in range(n_j):
                if int(elig[m][j]) == 1 and int(elig[:, j].sum()) == 1:
                    add("RESTRICT_ORPHANS_JOB", ERROR,
                        f"making machine {m} unavailable orphans job {j} "
                        f"(its only eligible machine); the reschedule would be infeasible.")

    for sw in ops.swaps:
        js = sw.get("jobs", [])
        if len(js) == 2 and js[0] == js[1]:
            add("SWAP_SELF", ERROR, f"job {js[0]} cannot be swapped with itself.")
        for m in sw.get("machines", []):
            if m in ops.restricted_machines:
                add("EDIT_ON_RESTRICTED", ERROR,
                    f"schedule edit references machine {m}, which is made unavailable.")
    for mv in ops.moves:
        m = mv.get("to_machine")
        if m in ops.restricted_machines:
            add("EDIT_ON_RESTRICTED", ERROR,
                f"a job is inserted on machine {m}, which is made unavailable.")
    for mv in ops.moves:
        job, fm = mv.get("job"), mv.get("from_machine")
        if job is not None and fm is not None and schedule:
            if job not in schedule.get(fm, []):
                add("JOB_NOT_ON_MACHINE", ERROR,
                    f"job {job} is not on machine {fm} in the current schedule.")

    moved_jobs = [mv["job"] for mv in ops.moves if "job" in mv]
    dup = {j for j in moved_jobs if moved_jobs.count(j) > 1}
    for j in sorted(dup):
        add("JOB_MOVED_TWICE", WARNING,
            f"job {j} is relocated by more than one request; the outcome is order-dependent.")

    return report




def detect_conflicts(
    modification_codes: Union[str, Sequence[str]],
    instance: Dict[str, Any],
    schedule: Optional[Dict[int, List[int]]] = None,
    env_params: Optional[Dict[str, int]] = None,
) -> Union[List[str], bool]:
    report = validate_requests(modification_codes, instance, schedule, env_params)
    msgs = [f"[{e.severity.upper()}] {e.message}" for e in report.errors]
    return msgs if msgs else False
