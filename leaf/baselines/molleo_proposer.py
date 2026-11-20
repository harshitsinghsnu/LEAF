"""MolLEO-style baseline (Yoshikawa/Nigam-lineage LLM-in-the-loop
evolutionary search, arXiv:2406.16976): the LLM performs ONLY crossover --
mutation is always a fixed GA operator, matching the paper's GPT-4 variant
exactly ("GPT-4 does crossover... mutation uses default Graph-GA").

Real implementation of the paper's offspring filter ("LLM edits are
filtered to those most Tanimoto-similar to the current top molecule",
section 3.2): since this pipeline's LLMProposer emits one candidate per
LLM call (not a batch to rank), the fair, non-fabricated analogue is to
keep the LLM's crossover offspring only if it is AT LEAST as
Tanimoto-similar to the current best archive member as the deterministic
GA crossover of the SAME two parents would have been -- otherwise fall
back to that GA crossover. This is a real similarity comparison on real
Morgan fingerprints (leaf.fragments.embed.FragmentEmbedder), not an
approximation.

Drop-in propose_pool(...) proposer for orchestrator/loop.py: same
signature as every other proposer, so this runs through the IDENTICAL
acquisition function / Verifier / budget accounting as leaf_llm (research
plan Sec 3.2 baseline-parity requirement) -- only the proposal mechanism
differs, which is what makes a leaf_llm-vs-molleo comparison fair.
"""
from __future__ import annotations

import random

import numpy as np

from leaf.agents.proposer import LLMProposer, RuleBasedProposer, ProposedCandidate, SLOTS
from leaf.fragments.embed import FragmentEmbedder
from leaf.fragments.vocabulary import Fragment, component_lists


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    """Real Tanimoto/Jaccard similarity on binary Morgan-fingerprint
    vectors (dot/(|a|+|b|-dot)) -- FragmentEmbedder already emits dense
    0/1 numpy arrays, so this is equivalent to RDKit's
    TanimotoSimilarity on the same fingerprint without needing its
    ExplicitBitVect wrapper."""
    dot = float(np.dot(a, b))
    denom = float(a.sum() + b.sum() - dot)
    return dot / denom if denom > 0 else 0.0


class MolLEOProposer:
    """See module docstring. `llm_kwargs` are forwarded to the internal
    LLMProposer (same backend/model/prompt_version knobs as leaf_llm) so
    this baseline can be run at the same model/backend as whichever
    leaf_llm run it's being compared against.

    `llm_mutation=True` is NOT part of the MolLEO paper (whose GPT-4
    variant never calls the LLM for mutation) -- it is a real extension
    this repo tests on top of the paper-faithful default, to see whether
    MolLEO's already-positive result improves further or regresses if the
    LLM is also given the mutation operator (same genuine-vs-fallback
    accounting as crossover, no similarity filter -- the paper defines
    that filter for crossover offspring only, so inventing one for
    mutation here would not be faithful to anything)."""

    def __init__(self, seed: int = 0, llm_kwargs: dict | None = None, llm_mutation: bool = False):
        self.rng = random.Random(seed)
        self.a_list, self.b_list, self.c_list = component_lists()
        self._llm = LLMProposer(seed=seed, **(llm_kwargs or {}))
        self._embedder = FragmentEmbedder()
        self.llm_mutation = llm_mutation
        # Diagnostics (research plan Sec 6.1 item 18/24 standard: report
        # the real backfill/filter rate, don't hide it).
        self.total_crossover_attempts = 0
        self.total_llm_crossover_kept = 0
        self.total_mutation_attempts = 0
        self.total_llm_mutation_kept = 0
        self.total_backend_errors = 0
        self._last_backend_error: str | None = None

    def _vocabs(self) -> dict[str, list[str]]:
        return {"A": self.a_list, "B": self.b_list, "C": self.c_list}

    def _ga_crossover(self, a: Fragment, b: Fragment) -> Fragment:
        return Fragment(**{s: getattr(self.rng.choice([a, b]), s) for s in SLOTS})

    def _ga_mutation(self, parent: Fragment) -> Fragment:
        vocabs = self._vocabs()
        slot = self.rng.choice(SLOTS)
        values = {s: getattr(parent, s) for s in SLOTS}
        values[slot] = self.rng.choice(vocabs[slot])
        return Fragment(**values)

    def _random_fragment(self) -> Fragment:
        return Fragment(A=self.rng.choice(self.a_list), B=self.rng.choice(self.b_list), C=self.rng.choice(self.c_list))

    def propose_pool(self, archive: list[Fragment], pool_size: int,
                      ledger_bias: dict | None = None,
                      archive_scores: list | None = None) -> list[ProposedCandidate]:
        # Deliberately ignores ledger_bias -- MolLEO has no Hypothesis
        # Ledger feedback path in the reference paper; it is not part of
        # what this baseline is testing.
        out: list[ProposedCandidate] = []

        top_frag = None
        if archive_scores is not None and len(archive_scores) == len(archive) and archive:
            best_idx = max(range(len(archive)), key=lambda i: sum(archive_scores[i]))
            top_frag = archive[best_idx]
        elif archive:
            # No scores wired through (e.g. round 0, or a caller that
            # doesn't pass archive_scores) -- fall back to the first
            # archive member. Deterministic, not fabricated: this only
            # affects which molecule the similarity filter targets, not
            # whether a candidate is valid.
            top_frag = archive[0]
        top_emb = self._embedder.embed(top_frag) if top_frag is not None else None

        has_scores = archive_scores is not None and len(archive_scores) == len(archive)

        def _scores_for(*frags: Fragment) -> list | None:
            if not has_scores:
                return None
            try:
                return [archive_scores[archive.index(f)] for f in frags]
            except ValueError:
                return None  # shouldn't happen (frags always sampled from archive), but never guess a score

        for _ in range(pool_size):
            if len(archive) >= 2 and self.rng.random() < 0.5:
                a, b = self.rng.sample(archive, 2)
                self.total_crossover_attempts += 1

                before_backfilled = self._llm.total_backfilled
                was_genuine_llm = False
                llm_pool = None
                try:
                    llm_pool = self._llm.propose_pool([a, b], 1, ledger_bias=None,
                                                        archive_scores=_scores_for(a, b))
                    # self._llm.propose_pool backfills with
                    # RuleBasedProposer on a reject -- this delta tells us
                    # honestly whether we got a REAL LLM candidate or a
                    # silent backfill, without duplicating its retry/
                    # backoff logic here.
                    was_genuine_llm = self._llm.total_backfilled == before_backfilled
                except RuntimeError as e:
                    # The backend itself was unreachable for all of
                    # LLMProposer's own retries (e.g. a transient Ollama
                    # 500) -- per the paper ("if the LLM's output is not
                    # a valid molecule, fall back to Graph-GA crossover"),
                    # an unusable LLM call is exactly this failure mode,
                    # not a reason to abort the whole run. Falls through
                    # to the GA fallback below; counted as a non-kept
                    # attempt in the real reject-rate diagnostic.
                    self.total_backend_errors += 1
                    self._last_backend_error = str(e)
                ga_frag = self._ga_crossover(a, b)

                if was_genuine_llm and top_emb is not None:
                    llm_frag = llm_pool[0].fragment
                    llm_sim = tanimoto(self._embedder.embed(llm_frag), top_emb)
                    ga_sim = tanimoto(self._embedder.embed(ga_frag), top_emb)
                    if llm_sim >= ga_sim:
                        self.total_llm_crossover_kept += 1
                        out.append(ProposedCandidate(llm_frag, operator="molleo_llm_crossover",
                                                      parents=(str(a), str(b))))
                        continue
                # LLM call failed/was backfilled, or the LLM offspring was
                # LESS similar to the current top molecule than plain GA
                # crossover would have been -- the paper's filter rejects
                # it; fall back to the deterministic GA crossover of the
                # same two parents (never a fabricated "pretend it
                # worked").
                out.append(ProposedCandidate(ga_frag, operator="molleo_ga_crossover_fallback",
                                              parents=(str(a), str(b))))
            elif archive:
                parent = self.rng.choice(archive)
                if self.llm_mutation:
                    self.total_mutation_attempts += 1
                    before_backfilled = self._llm.total_backfilled
                    was_genuine_llm = False
                    llm_pool = None
                    try:
                        llm_pool = self._llm.propose_pool([parent], 1, ledger_bias=None,
                                                            archive_scores=_scores_for(parent))
                        was_genuine_llm = self._llm.total_backfilled == before_backfilled
                    except RuntimeError as e:
                        self.total_backend_errors += 1
                        self._last_backend_error = str(e)
                    if was_genuine_llm:
                        self.total_llm_mutation_kept += 1
                        out.append(ProposedCandidate(llm_pool[0].fragment, operator="molleo_llm_mutation",
                                                      parents=(str(parent),)))
                        continue
                    # Same fallback contract as crossover: an unusable LLM
                    # call (not a modeling failure) falls back to the
                    # deterministic GA operator, never aborts the run.
                    out.append(ProposedCandidate(self._ga_mutation(parent), operator="molleo_ga_mutation_fallback",
                                                  parents=(str(parent),)))
                else:
                    out.append(ProposedCandidate(self._ga_mutation(parent), operator="molleo_ga_mutation",
                                                  parents=(str(parent),)))
            else:
                out.append(ProposedCandidate(self._random_fragment(), operator="molleo_ga_random_init"))
        return out

    @property
    def llm_crossover_keep_rate(self) -> float:
        return (self.total_llm_crossover_kept / self.total_crossover_attempts
                if self.total_crossover_attempts else 0.0)

    @property
    def llm_mutation_keep_rate(self) -> float:
        return (self.total_llm_mutation_kept / self.total_mutation_attempts
                if self.total_mutation_attempts else 0.0)

    @property
    def reject_rate(self) -> float:
        """Reported under the same name run_experiment.py's summary.json
        uses for LLMProposer, so a high value is equally visible here:
        fraction of LLM-eligible attempts (crossover, plus mutation when
        llm_mutation=True) where the LLM candidate was unused (backfilled,
        unreachable, or filtered out by the similarity rule)."""
        n = self.total_crossover_attempts + (self.total_mutation_attempts if self.llm_mutation else 0)
        kept = self.total_llm_crossover_kept + (self.total_llm_mutation_kept if self.llm_mutation else 0)
        return 1.0 - (kept / n if n else 0.0)
