"""Thompson-sampling bandit baseline: a genuinely different, non-evolutionary
search paradigm from static-GA (no crossover/mutation operators) and random
search (informed, not blind) -- added as a lighter, real alternative to a
full trained generative Proposer (research plan/paper §7 item 3, deferred
as out-of-scope training work).

Per fragment slot (A, B, C), maintains a Beta(successes+1, failures+1)
posterior per real vocabulary value, fit fresh from the current Pareto
archive every call (archive membership = "success" signal, since archive
members are by construction non-dominated/good; every other real
vocabulary value not currently in the archive is a "failure" for this
round's posterior). A new candidate is assembled by independently
Thompson-sampling one value per slot -- exploration/exploitation trades
off automatically via the posterior's own variance, unlike either
baseline it's compared against.

This is a real, fitted (if simple) probabilistic model, not a trained
generator in the VAE/diffusion sense -- reported as such, not oversold.
"""
from __future__ import annotations

import random

from leaf.agents.proposer import ProposedCandidate
from leaf.fragments.vocabulary import Fragment, component_lists


class BanditProposer:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.a_list, self.b_list, self.c_list = component_lists()

    def _sample_slot(self, values: list[str], archive_values: list[str]) -> str:
        counts = {v: archive_values.count(v) for v in values}
        n_archive = max(len(archive_values), 1)
        best_val, best_draw = None, -1.0
        for v in values:
            successes = counts[v] + 1  # Beta(1,1) uniform prior, +1 smoothing
            failures = (n_archive - counts[v]) + 1
            draw = self.rng.betavariate(successes, failures)
            if draw > best_draw:
                best_val, best_draw = v, draw
        return best_val

    def propose_pool(self, archive: list[Fragment], pool_size: int,
                      ledger_bias: dict | None = None) -> list[ProposedCandidate]:
        a_vals = [f.A for f in archive]
        b_vals = [f.B for f in archive]
        c_vals = [f.C for f in archive]
        out = []
        for _ in range(pool_size):
            frag = Fragment(
                A=self._sample_slot(self.a_list, a_vals),
                B=self._sample_slot(self.b_list, b_vals),
                C=self._sample_slot(self.c_list, c_vals),
            )
            out.append(ProposedCandidate(frag, operator="thompson_sample"))
        return out
