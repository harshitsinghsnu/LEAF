"""Random search baseline: uniform random fragment tuples, same q and budget
as LEAF, but no acquisition function -- the weakest reference point for
RQ1 (research plan §7 Experimental Design, E2)."""
from __future__ import annotations

import random

from leaf.fragments.vocabulary import Fragment, component_lists
from leaf.agents.proposer import ProposedCandidate


class RandomSearchProposer:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.a_list, self.b_list, self.c_list = component_lists()

    def propose_pool(self, archive, pool_size: int, ledger_bias=None) -> list[ProposedCandidate]:
        out = []
        for _ in range(pool_size):
            frag = Fragment(A=self.rng.choice(self.a_list), B=self.rng.choice(self.b_list), C=self.rng.choice(self.c_list))
            out.append(ProposedCandidate(frag, operator="random"))
        return out
