"""Pretraining-leakage probe -- MUST run and be disclosed before any E5
rediscovery claim (research plan §5.5). Freezing the RAG corpus alone does
not address knowledge the base LLM already has from pretraining; the
LUMI-lab preprint has been public on bioRxiv since Feb 2025.

Usage: call `run_leakage_probe(llm_call_fn)` with a callable that queries
the bare LLM with NO RAG context and NO LEAF archive/history. If any probe
question's response contains a target keyword, E5 must be downgraded from
"external validation" to "consistency check" and this is reported in the
paper regardless of outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

PROBE_QUESTIONS = [
    "What tail chemical modifications enhance mRNA transfection efficiency in ionizable lipids?",
    "What structural feature of an ionizable lipid tail was recently shown to improve potency "
    "in a large-scale self-driving-lab screen?",
    "Summarize recent findings on halogenated or brominated lipid tails and mRNA delivery.",
]

# Keywords whose unprompted appearance indicates the base model already
# knows the target finding (structure-property, NOT structure-tropism --
# the tropism framing in Revision 1 was itself a factual error, §5.5).
LEAK_KEYWORDS = ["brominat", "halogenat", "lumi-lab", "lumi lab"]


@dataclass
class LeakageProbeResult:
    question: str
    response: str
    leaked: bool
    matched_keywords: list[str]


def run_leakage_probe(llm_call_fn: Callable[[str], str]) -> list[LeakageProbeResult]:
    results = []
    for q in PROBE_QUESTIONS:
        resp = llm_call_fn(q)
        lower = resp.lower()
        hits = [k for k in LEAK_KEYWORDS if k in lower]
        results.append(LeakageProbeResult(q, resp, leaked=bool(hits), matched_keywords=hits))
    return results


def e5_status(probe_results: list[LeakageProbeResult]) -> str:
    """Returns 'external_validation' or 'consistency_check' -- the label
    that must be used verbatim in the paper's E5 reporting (§5.5, §8.5)."""
    return "consistency_check" if any(r.leaked for r in probe_results) else "external_validation"
