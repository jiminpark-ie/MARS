from __future__ import annotations
from typing import Dict, List
import re

_NAN = float("nan")


def _normalize_code(code: str) -> str:
    code = re.sub(r"\s+", " ", code.strip())
    return code.replace('"', "'")


def functional_correctness(predictions: List[str], labels: List[str]) -> float:
    pairs = [(p, l) for p, l in zip(predictions, labels) if l.strip()]
    if not pairs:
        return _NAN
    hits = sum(1 for p, l in pairs
               if _normalize_code(p) == _normalize_code(l))
    return 100.0 * hits / len(pairs)


def chrf(predictions: List[str], labels: List[str]) -> float:
    """Mean sentence-level chrF++ over non-empty pairs."""
    try:
        from sacrebleu.metrics import CHRF
    except Exception:
        return _NAN

    metric = CHRF(word_order=2)
    scores = [
        metric.sentence_score(prediction, [label]).score
        for prediction, label in zip(predictions, labels)
        if prediction.strip() and label.strip()
    ]
    return sum(scores) / len(scores) if scores else 0.0


def codebert(predictions: List[str], labels: List[str],
             lang: str = "python") -> Dict[str, float]:
    """CodeBERTScore precision/recall/F1 (means over non-empty pairs)."""
    pairs = [(p, l) for p, l in zip(predictions, labels)
             if p.strip() and l.strip()]
    if not pairs:
        return {"codebert_precision": _NAN, "codebert_recall": _NAN,
                "codebert_f1": _NAN}
    try:
        import code_bert_score
    except Exception:
        return {"codebert_precision": _NAN, "codebert_recall": _NAN,
                "codebert_f1": _NAN}

    preds, refs = zip(*pairs)
    precision, recall, f1, _f3 = code_bert_score.score(
        cands=list(preds), refs=list(refs), lang=lang,
    )
    return {
        "codebert_precision": float(precision.mean()),
        "codebert_recall": float(recall.mean()),
        "codebert_f1": float(f1.mean()),
    }


def all_code_metrics(
    predictions: List[str],
    labels: List[str],
    with_codebert: bool = True,
) -> Dict[str, float]:
    out: Dict[str, float] = {
        "functional_correctness": functional_correctness(predictions, labels),
        "chrf": chrf(predictions, labels),
    }
    if with_codebert:
        out.update(codebert(predictions, labels))
    else:
        out.update({"codebert_precision": _NAN, "codebert_recall": _NAN,
                    "codebert_f1": _NAN})
    return out
