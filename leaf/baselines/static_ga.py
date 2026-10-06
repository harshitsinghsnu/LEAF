"""Static, non-LLM genetic algorithm baseline.

Same fragment vocabulary, same frozen O_GT, and CRITICALLY the same
acquisition function and same q as full LEAF (research plan §3.2 baseline-
parity requirement) -- otherwise RQ1 conflates "has an LLM proposer" with
"has an acquisition step at all," which is not a fair ablation.
"""
from __future__ import annotations

import random

from leaf.fragments.vocabulary import Fragment, component_lists
from leaf.agents.proposer import ProposedCandidate, SLOTS


class StaticGAProposer:
    """Fixed (non-LLM) crossover/mutation operators. Drop-in replacement for
    RuleBasedProposer / the LLM proposer in orchestrator/loop.py -- same
    propose_pool(...) signature so it can be swapped via LoopConfig.use_proposer."""

    def __init__(self, seed: int = 0, mutation_rate: float = 0.5):
        # Bumped from 0.2 -> 0.5 after finding a real structural asymmetry
        # against RuleBasedProposer: that proposer does a straight 50%
        # pure-mutation / 50% crossover split (see agents/proposer.py),
        # always injecting fresh single-slot diversity half the time.
        # This class instead ALWAYS crosses over two archive parents and
        # only sometimes (mutation_rate) perturbs one slot afterward --
        # at 0.2 that's much weaker diversity injection, not a like-for-
        # like comparison. 0.5 is a more standard, defensible rate for a
        # 3-gene combinatorial GA and puts the two proposers' exploration
        # pressure on comparable footing.
        self.rng = random.Random(seed)
        self.mutation_rate = mutation_rate
        self.a_list, self.b_list, self.c_list = component_lists()

    def _vocabs(self) -> dict[str, list[str]]:
        return {"A": self.a_list, "B": self.b_list, "C": self.c_list}

    def _random_fragment(self) -> Fragment:
        return Fragment(A=self.rng.choice(self.a_list), B=self.rng.choice(self.b_list), C=self.rng.choice(self.c_list))

    def propose_pool(self, archive: list[Fragment], pool_size: int, ledger_bias=None) -> list[ProposedCandidate]:
        # Deliberately ignores ledger_bias -- the static GA has no
        # Hypothesis Ledger feedback path by design (it's the non-agentic
        # comparator, not Ablation (d)).
        out = []
        vocabs = self._vocabs()
        for _ in range(pool_size):
            if len(archive) >= 2:
                a, b = self.rng.sample(archive, 2)
                values = {s: getattr(self.rng.choice([a, b]), s) for s in SLOTS}
                if self.rng.random() < self.mutation_rate:
                    s = self.rng.choice(SLOTS)
                    values[s] = self.rng.choice(vocabs[s])
                out.append(ProposedCandidate(Fragment(**values), operator="ga_crossover_mutation"))
            else:
                out.append(ProposedCandidate(self._random_fragment(), operator="ga_random_init"))
        return out
