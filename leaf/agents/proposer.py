"""Proposer Agent: LLM-driven crossover/mutation over the fragment vocabulary.

No trained generative model. Emits a POOL P_t (|P_t| = m >> q) which the
acquisition function then subselects for actual ground-truth querying --
in Revision 1 the Proposer's output was queried directly, which is why
the surrogate/acquisition split never mattered. See research plan §3.2.

Fragment slots are (A, B, C) -- the three real Ugi-3CR reactants (amine,
aldehyde, isocyanide) from the vendored AGILE dataset, not the earlier
illustrative 6-slot placeholder. See fragments/vocabulary.py.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from leaf._env import load_dotenv_once
from leaf.fragments.vocabulary import Fragment, component_lists

SLOTS = ["A", "B", "C"]


def _candidate_schema() -> dict:
    a_list, b_list, c_list = component_lists()
    return {
        "type": "object",
        "properties": {
            "A": {"type": "string", "enum": a_list},
            "B": {"type": "string", "enum": b_list},
            "C": {"type": "string", "enum": c_list},
            "provenance": {
                "type": "object",
                "properties": {
                    "parents": {"type": "array", "items": {"type": "string"}},
                    "operator": {"type": "string", "enum": ["crossover", "mutation"]},
                },
                "required": ["operator"],
            },
        },
        "required": ["A", "B", "C", "provenance"],
    }


@dataclass
class ProposedCandidate:
    fragment: Fragment
    operator: str
    parents: tuple[str, ...] = ()
    rationale: str | None = None
    thinking: str | None = None


class RuleBasedProposer:
    """A deterministic stand-in for the LLM proposer, used for tests/CI and
    as the non-agentic reference implementation the LLM proposer is
    diffed against during development. The real system swaps this for an
    LLM call constrained by CANDIDATE_SCHEMA (agents/prompts/proposer_v*.md)."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.a_list, self.b_list, self.c_list = component_lists()

    def _vocabs(self) -> dict[str, list[str]]:
        return {"A": self.a_list, "B": self.b_list, "C": self.c_list}

    def _random_fragment(self) -> Fragment:
        return Fragment(
            A=self.rng.choice(self.a_list), B=self.rng.choice(self.b_list), C=self.rng.choice(self.c_list),
        )

    def crossover(self, parent_a: Fragment, parent_b: Fragment) -> ProposedCandidate:
        child = {s: getattr(self.rng.choice([parent_a, parent_b]), s) for s in SLOTS}
        return ProposedCandidate(Fragment(**child), operator="crossover", parents=(str(parent_a), str(parent_b)))

    def mutation(self, parent: Fragment, ledger_bias: dict | None = None) -> ProposedCandidate:
        vocabs = self._vocabs()
        slot = self.rng.choice(SLOTS)
        values = {s: getattr(parent, s) for s in SLOTS}
        # Hypothesis-Ledger bias: prefer validated fragment values if available.
        if ledger_bias and slot in ledger_bias and self.rng.random() < 0.7:
            values[slot] = self.rng.choice(ledger_bias[slot])
        else:
            values[slot] = self.rng.choice(vocabs[slot])
        return ProposedCandidate(Fragment(**values), operator="mutation", parents=(str(parent),))

    def propose_pool(self, archive: list[Fragment], pool_size: int,
                      ledger_bias: dict | None = None) -> list[ProposedCandidate]:
        """Generate a pool P_t of size pool_size via crossover/mutation seeded
        from the current Pareto archive (falls back to random fragments if
        the archive is empty, e.g. round 0)."""
        out = []
        for _ in range(pool_size):
            if len(archive) >= 2 and self.rng.random() < 0.5:
                a, b = self.rng.sample(archive, 2)
                out.append(self.crossover(a, b))
            elif archive:
                parent = self.rng.choice(archive)
                out.append(self.mutation(parent, ledger_bias))
            else:
                out.append(ProposedCandidate(self._random_fragment(), operator="mutation"))
        return out

    def validate(self, payload: str) -> ProposedCandidate | None:
        """Parse+validate a raw LLM JSON response against the candidate
        schema (`_candidate_schema()`). Returns None on failure (caller
        logs the reject rate, per §6.1 item 24)."""
        try:
            d = json.loads(payload)
            frag = Fragment(d["A"], d["B"], d["C"])
            prov = d["provenance"]
            return ProposedCandidate(frag, operator=prov["operator"], parents=tuple(prov.get("parents", ())))
        except (json.JSONDecodeError, KeyError, ValueError):
            return None


class LLMProposer:
    """The real Proposer Agent: an LLM performs crossover/mutation over the
    fragment vocabulary. No trained generative model -- the LLM's
    combinatorial reasoning over the real (A, B, C) Ugi-3CR component lists
    is the entire generative mechanism (research plan §3.2). Drop-in
    replacement for RuleBasedProposer: same propose_pool(...) signature, so
    it plugs into orchestrator/loop.py and baselines/*.py without any other
    code changes.

    Three backends, selected via `backend`:
      - "ollama" (default): a local Ollama server (default
        http://localhost:11434) with the given model pulled
        (`ollama pull llama3.1:8b`).
      - "groq": Groq's OpenAI-compatible chat-completions API
        (api.groq.com), for real hosted 70B-class models
        (e.g. model="llama-3.3-70b-versatile"). Reads the API key from the
        GROQ_API_KEY environment variable -- never pass a key as a literal
        argument or hardcode one in a config file.
      - "gemini": Google's Gemini API (generativelanguage.googleapis.com),
        for a real frontier-class hosted model (e.g.
        model="gemini-2.0-flash"). Reads the API key from the
        GEMINI_API_KEY environment variable.

    This is NOT a rule-based stand-in -- if the backend is unreachable, the
    model is missing, or the required API key environment variable is
    unset, this raises rather than silently falling back to something
    else, since that would misrepresent which method actually ran
    (research plan's "no dummy implementations" standard).
    """

    def __init__(self, seed: int = 0, model: str = "llama3.1:8b",
                 host: str = "http://localhost:11434", temperature: float = 0.7,
                 max_retries: int = 2, timeout: float = 120.0, prompt_version: str = "v1",
                 backend: str = "ollama", vocab_sample_size: int | None = None,
                 think: bool = False):
        # think=True requests the model's own native extended reasoning
        # (Ollama's "think" API field; genuine for reasoning-tuned models
        # like qwen3, not a prompted request for a sentence of rationale
        # -- see v5_cot for that, a different and much cheaper mechanism).
        # Only meaningful for backend="ollama" with a reasoning-capable
        # model; silently has no effect on models that don't support it.
        # Real reasoning traces are captured in self.last_thinking for
        # direct inspection (external review: test an actual reasoning
        # model, not just a prompted rationale, following arXiv:2610.00898's
        # distinction between models that reason over their inputs and
        # models merely asked to).
        self.think = think
        self.last_thinking: str | None = None
        load_dotenv_once()
        self.rng = random.Random(seed)
        self.model = model
        self.host = host
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        # None = embed the full vocabulary every call (default, matches
        # §6.2's local-model results). A finite value samples that many
        # values per slot instead -- cuts prompt size substantially for
        # metered hosted APIs (measured: full extended vocabulary,
        # 29/74/14 values, costs ~1,850 tokens/call on Groq, exceeding
        # its 100K-tokens/day free tier for even one seed-run; see §7
        # item 5). Validation against the FULL real component lists
        # (self.a_list/b_list/c_list) is unchanged regardless of this
        # setting -- only what's shown in the prompt shrinks, not what
        # counts as a valid, non-hallucinated value.
        self.vocab_sample_size = vocab_sample_size
        if prompt_version not in ("v1", "v2_diversity", "v3_scored", "v4_rank", "v5_cot", "v6_shuffled"):
            raise ValueError(f"unknown prompt_version {prompt_version!r}")
        self.prompt_version = prompt_version
        if backend not in ("ollama", "groq", "gemini", "cerebras"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        if backend == "groq" and not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError(
                "backend='groq' requires the GROQ_API_KEY environment variable to be set "
                "(get a free key at console.groq.com). Not reading it from anywhere else, "
                "so it can't leak into a config file or command history."
            )
        if backend == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError(
                "backend='gemini' requires the GEMINI_API_KEY environment variable to be set "
                "(get a free key at aistudio.google.com). Not reading it from anywhere else, "
                "so it can't leak into a config file or command history."
            )
        if backend == "cerebras" and not os.environ.get("CEREBRAS_API_KEY"):
            raise RuntimeError(
                "backend='cerebras' requires the CEREBRAS_API_KEY environment variable to be "
                "set (get a free key at cloud.cerebras.ai). Not reading it from anywhere else, "
                "so it can't leak into a config file or command history."
            )
        self.a_list, self.b_list, self.c_list = component_lists()
        self._fallback = RuleBasedProposer(seed=seed)
        # Reproducibility/reject-rate bookkeeping (research plan §6.1 item 18/24).
        self.total_requested = 0
        self.total_llm_valid = 0
        self.total_backfilled = 0

    def _call_llm(self, prompt: str) -> str:
        if self.backend == "ollama":
            return self._call_ollama(prompt)
        if self.backend == "groq":
            return self._call_groq(prompt)
        if self.backend == "cerebras":
            return self._call_cerebras(prompt)
        return self._call_gemini(prompt)

    def _call_ollama(self, prompt: str) -> str:
        payload_dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "seed": self.rng.randint(0, 2**31 - 1)},
        }
        if self.think:
            # Real, verified constraint: Ollama's qwen3 produces NO
            # reasoning trace at all when format="json" is also set --
            # the schema-constrained JSON mode this pipeline otherwise
            # relies on for zero-hallucination parsing appears to suppress
            # the reasoning phase entirely (confirmed directly via the raw
            # API, not assumed). Genuine extended reasoning requires
            # dropping the forced JSON format; the model still reliably
            # emits parseable JSON after reasoning in practice, and
            # _parse_one's json.loads already tolerates this (it does not
            # require format="json" to have been set upstream).
            payload_dict["think"] = True
            # Second real, verified constraint found with deepseek-r1:8b:
            # Ollama's default num_ctx (2048-4096, never previously set
            # here) is too small to hold the prompt plus a long reasoning
            # trace plus the final JSON, causing the model to run out of
            # context mid-generation and emit incoherent garbage instead
            # of a parseable response -- confirmed directly by replaying
            # the exact failing prompt with num_ctx=8192, which restored a
            # coherent multi-thousand-character trace and valid JSON.
            # Explicit num_ctx is therefore required whenever think=True,
            # not just for this one model.
            payload_dict["options"]["num_ctx"] = 8192
        else:
            payload_dict["format"] = "json"
        payload = json.dumps(payload_dict).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "leaf-research/1.0"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read())
        self.last_thinking = result.get("thinking")
        return result["response"]

    def _call_groq(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "seed": self.rng.randint(0, 2**31 - 1),
            "response_format": {"type": "json_object"},
            # Without an explicit cap Groq's free-tier limiter appears to
            # reserve room for a very large completion and reject the
            # whole request outright (confirmed: x-should-retry:false,
            # x-ratelimit-remaining-tokens still full, retry-after >1000s
            # -- a per-request budget check, not real throughput
            # exhaustion). The response is one small JSON object.
            "max_tokens": 300,
        }).encode()
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions", data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
                "User-Agent": "leaf-research/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"]

    def _call_cerebras(self, prompt: str) -> str:
        # Cerebras is also an OpenAI-compatible chat-completions API.
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "seed": self.rng.randint(0, 2**31 - 1),
            "response_format": {"type": "json_object"},
            "max_tokens": 300,
        }).encode()
        req = urllib.request.Request(
            "https://api.cerebras.ai/v1/chat/completions", data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['CEREBRAS_API_KEY']}",
                "User-Agent": "leaf-research/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"]

    def _call_gemini(self, prompt: str) -> str:
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
            },
        }).encode()
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent?key={os.environ['GEMINI_API_KEY']}")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "leaf-research/1.0"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read())
        return result["candidates"][0]["content"]["parts"][0]["text"]

    def _build_prompt(self, archive: list[Fragment], ledger_bias: dict | None,
                       archive_scores: list | None = None) -> str:
        # A local 8B model in JSON mode reliably produces ONE well-formed
        # object per call but unreliably follows "return an array of N
        # items" -- observed directly: it silently collapsed a
        # pool_size=4 request down to a single top-level object. Asking
        # for exactly one candidate per call and looping is far more
        # robust than trying to coax a small model into a batched array.
        if archive_scores is not None and len(archive_scores) == len(archive) and archive:
            sample_idx = self.rng.sample(range(len(archive)), min(len(archive), 8))
            archive_sample = [archive[i] for i in sample_idx]
            score_sample = [archive_scores[i] for i in sample_idx]
        else:
            archive_sample = self.rng.sample(archive, min(len(archive), 8)) if archive else []
            score_sample = None

        if self.prompt_version == "v4_rank" and score_sample is not None:
            # External review follow-up: v3_scored shows raw magnitudes,
            # which confounds "genuine preference reasoning" with
            # "anchoring on/interpolating between absolute numbers." This
            # variant shows only each sampled archive member's RANK among
            # the sample (combining potency and potency_raw into one
            # summed-rank ordering, since the LLM is never shown which
            # scalarization the acquisition function itself uses), never
            # the underlying values or scale -- isolating whether relative
            # preference alone (not magnitude) drives any effect.
            order = sorted(range(len(score_sample)),
                            key=lambda k: -(score_sample[k][0] + score_sample[k][1]))
            rank_of = {idx: r + 1 for r, idx in enumerate(order)}
            n_shown = len(archive_sample)
            archive_lines = "\n".join(
                f"- A={f.A} B={f.B} C={f.C} | rank {rank_of[k]} of {n_shown} shown (1 = best)"
                for k, f in enumerate(archive_sample)
            ) or "(empty -- propose a diverse random triple)"
            objective_text = (
                "\nEach archive member above is ranked only relative to the others shown here "
                "(rank 1 = most promising of this sample, by real transfection-potency "
                "measurements on two assays). Exact scores are not given, only relative "
                "preference. Prefer proposals you expect to rank at least as well as the "
                "top-ranked archive members shown above.\n"
            )
        elif self.prompt_version == "v6_shuffled" and score_sample is not None:
            # Perturbation control (arXiv:2610.00898's methodology applied
            # to LEAF): v3_scored shows REAL scores attached to their
            # correct fragments. This variant shows the same real score
            # VALUES but randomly reassigns which fragment each score is
            # attached to, breaking the true score-to-fragment mapping
            # while keeping the surface form (numbers, phrasing, objective
            # description) identical. If the model is genuinely using the
            # score-to-fragment correspondence to decide what to propose,
            # scrambling it should degrade the result back toward the
            # default-prompt baseline; if performance is unaffected, the
            # v3_scored effect was driven by something other than reading
            # the specific mapping (e.g. noticing numbers are present at
            # all), exactly the "shuffled input" diagnostic used to show
            # BioReason ignores Evo2 DNA features in the reference paper.
            shuffled_scores = list(score_sample)
            self.rng.shuffle(shuffled_scores)
            archive_lines = "\n".join(
                f"- A={f.A} B={f.B} C={f.C} | potency={s[0]:.3f} potency_raw={s[1]:.3f}"
                for f, s in zip(archive_sample, shuffled_scores)
            ) or "(empty -- propose a diverse random triple)"
            objective_text = (
                "\nObjectives (both real transfection-potency assays, both to be MAXIMIZED): "
                "'potency' = HeLa cell transfection efficiency; 'potency_raw' = RAW264.7 cell "
                "transfection efficiency. Higher values on both axes are better. Prefer proposals "
                "you expect to score at least as well as the highest-potency archive members "
                "shown above on one or both objectives.\n"
            )
        elif self.prompt_version == "v3_scored" and score_sample is not None:
            # External review item #2: the default prompt shows only
            # fragment identities, never the real objective values or a
            # description of what is being optimized -- this variant adds
            # both, testing whether quantitative feedback (not just
            # archive membership) changes proposal quality.
            archive_lines = "\n".join(
                f"- A={f.A} B={f.B} C={f.C} | potency={s[0]:.3f} potency_raw={s[1]:.3f}"
                for f, s in zip(archive_sample, score_sample)
            ) or "(empty -- propose a diverse random triple)"
            objective_text = (
                "\nObjectives (both real transfection-potency assays, both to be MAXIMIZED): "
                "'potency' = HeLa cell transfection efficiency; 'potency_raw' = RAW264.7 cell "
                "transfection efficiency. Higher values on both axes are better. Prefer proposals "
                "you expect to score at least as well as the highest-potency archive members "
                "shown above on one or both objectives.\n"
            )
        else:
            archive_lines = "\n".join(f"- A={f.A} B={f.B} C={f.C}" for f in archive_sample) or "(empty -- propose a diverse random triple)"
            objective_text = ""
        bias_text = ""
        if ledger_bias:
            bias_text = ("\nValidated structure-property hypotheses to bias mutations toward "
                         f"(prefer these values when mutating the given slot): {json.dumps(ledger_bias)}")

        if self.prompt_version == "v1":
            diversity_text = ""
        else:
            # v2_diversity: §6.2/§7 diagnosed the dominant real-LLM failure
            # mode as re-proposing candidates already in the archive
            # (62-68% exact duplicates for the two <=8B models tested).
            # This variant names the failure mode explicitly and lists a
            # larger "already evaluated, do not repeat" set, to test
            # whether the underperformance is a proposal-diversity problem
            # fixable by prompting, or persists regardless (research plan
            # §7 item 9).
            avoid_sample = self.rng.sample(archive, min(len(archive), 20)) if archive else []
            avoid_lines = "\n".join(f"- A={f.A} B={f.B} C={f.C}" for f in avoid_sample) or "(none yet)"
            diversity_text = f"""
IMPORTANT: proposing a triple that exactly matches one already evaluated wastes a
real query and gains no information. Do NOT propose any of the following
already-evaluated triples, and prefer a mutation/crossover that changes more than
one slot if it still stays plausible:
{avoid_lines}
"""

        if self.vocab_sample_size is None:
            a_shown, b_shown, c_shown = self.a_list, self.b_list, self.c_list
        else:
            n = self.vocab_sample_size
            a_shown = self.rng.sample(self.a_list, min(n, len(self.a_list)))
            b_shown = self.rng.sample(self.b_list, min(n, len(self.b_list)))
            c_shown = self.rng.sample(self.c_list, min(n, len(self.c_list)))

        if self.prompt_version == "v5_cot":
            # External review item #4: require a short structural rationale
            # BEFORE committing to a candidate, without ever showing scores
            # -- tests whether forcing explicit mechanistic reasoning (not
            # label access) changes proposal quality. rationale_required
            # text is enforced in _parse_one (empty/missing rationale is
            # rejected exactly like an invalid fragment value, so a model
            # cannot skip the reasoning step and still count as valid).
            cot_instruction = (
                "\nBefore choosing values, briefly reason (1-2 sentences) about which structural "
                "features (head-group polarity, tail length/unsaturation, linker chemistry) are "
                "likely to affect transfection potency for this candidate, then commit to one "
                "triple consistent with that reasoning. Include this reasoning as \"rationale\" "
                "in your JSON output.\n"
            )
            schema_line = (
                '{"A": "<one of the A values>", "B": "<one of the B values>", "C": "<one of the C values>",\n'
                '  "rationale": "<1-2 sentence structural reasoning>",\n'
                '  "provenance": {"operator": "crossover" or "mutation", "parents": ["..."]}}'
            )
        else:
            cot_instruction = ""
            schema_line = (
                '{"A": "<one of the A values>", "B": "<one of the B values>", "C": "<one of the C values>",\n'
                '  "provenance": {"operator": "crossover" or "mutation", "parents": ["..."]}}'
            )

        return f"""You are the Proposer Agent in an evolutionary lipid-design loop. Candidates are Ugi
three-component-reaction lipids, each a triple (A, B, C):
A = amine, one of: {json.dumps(a_shown)}
B = aldehyde, one of: {json.dumps(b_shown)}
C = isocyanide, one of: {json.dumps(c_shown)}

Current Pareto archive (sample):
{archive_lines}
{objective_text}{bias_text}
{diversity_text}{cot_instruction}
Propose exactly ONE new candidate via crossover (recombining A/B/C from two archive
parents) or mutation (changing exactly one of A/B/C from a single archive parent, or
a random triple if the archive is empty). The value of A, B, and C in your output
MUST be copied EXACTLY (character for character) from the lists above -- do not
paraphrase or alter the SMILES strings in any way.

Respond with ONLY a single JSON object shaped like:
{schema_line}
"""

    def _parse_one(self, raw: str) -> ProposedCandidate | None:
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            # Without forced format="json" (required for genuine reasoning
            # traces, see think=True in _call_ollama), a model may wrap the
            # object in markdown fences or surrounding prose. Extract the
            # first top-level {...} block and retry once before rejecting
            # -- still reject, don't guess, if that also fails to parse.
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return None
            try:
                item = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        if isinstance(item, list):
            item = item[0] if item and isinstance(item[0], dict) else None
        if not isinstance(item, dict):
            return None
        A, B, C = item.get("A"), item.get("B"), item.get("C")
        if A not in self.a_list or B not in self.b_list or C not in self.c_list:
            return None  # hallucinated/malformed value -- reject, don't guess
        rationale = None
        if self.prompt_version == "v5_cot":
            rationale = item.get("rationale")
            if not isinstance(rationale, str) or not rationale.strip():
                return None  # missing/empty rationale is rejected exactly like an invalid fragment value
        prov = item.get("provenance")
        if not isinstance(prov, dict):
            # Malformed provenance (e.g. a model returns a string instead
            # of the nested object, observed directly from a reasoning-mode
            # model without forced JSON schema) does not invalidate an
            # otherwise-valid, already-validated (A, B, C) -- provenance is
            # bookkeeping metadata, not a hallucination-risk field, so we
            # fall back to safe defaults rather than reject the candidate.
            prov = {}
        op = prov.get("operator", "mutation")
        if op not in ("crossover", "mutation"):
            op = "mutation"
        parents_raw = prov.get("parents", ())
        parents = tuple(parents_raw) if isinstance(parents_raw, (list, tuple)) else ()
        return ProposedCandidate(Fragment(A=A, B=B, C=C), operator=op,
                                   parents=parents, rationale=rationale)

    def propose_pool(self, archive: list[Fragment], pool_size: int,
                      ledger_bias: dict | None = None,
                      archive_scores: list | None = None) -> list[ProposedCandidate]:
        self.total_requested += pool_size
        prompt = self._build_prompt(archive, ledger_bias, archive_scores=archive_scores)

        valid: list[ProposedCandidate] = []
        for i in range(pool_size):
            # Proactive pacing for hosted backends: reactive backoff alone
            # (below) lets a round's pool_size calls fire back-to-back,
            # burn the whole per-minute token budget in seconds, then
            # spend the rest of the round recovering from repeated 429s.
            # A fixed inter-call delay keeps sustained throughput under
            # the free-tier ceiling (Groq: 12K tokens/min at ~1K
            # tokens/prompt here -> >=6s/call keeps this comfortably
            # under budget) instead of bursting and backing off.
            if self.backend != "ollama" and i > 0:
                time.sleep(7.0)
            candidate = None
            last_error: Exception | None = None
            for _attempt in range(self.max_retries + 1):
                try:
                    raw = self._call_llm(prompt)
                    candidate = self._parse_one(raw)
                    if candidate is not None:
                        if self.think and self.last_thinking:
                            candidate.thinking = self.last_thinking
                        break
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                    last_error = e
                    # Hosted-API free tiers (Groq especially: 12K tokens/min
                    # on llama-3.3-70b-versatile) rate-limit on requests
                    # this loop issues one-at-a-time -- back off on 429
                    # rather than burning retries instantly, using the
                    # server's own Retry-After hint when it provides one.
                    if isinstance(e, urllib.error.HTTPError) and e.code == 429:
                        # Gemini's free tier doesn't set a Retry-After header
                        # (unlike Groq) -- its error body reports a ~29s
                        # window instead, so 30s is a closer default than
                        # the old 20s when no header is present.
                        wait_s = float(e.headers.get("Retry-After", 30)) if e.headers else 30.0
                        time.sleep(min(wait_s, 60.0))
            if candidate is None and last_error is not None:
                hint = ({"ollama": f"Is `ollama serve` running and has `{self.model}` been pulled?",
                         "groq": "Check GROQ_API_KEY is set and valid, and the model name is a real Groq model ID.",
                         "gemini": "Check GEMINI_API_KEY is set and valid, and the model name is a real Gemini model ID.",
                         "cerebras": "Check CEREBRAS_API_KEY is set and valid, and the model name is a real Cerebras model ID."}
                        [self.backend])
                raise RuntimeError(
                    f"LLMProposer: backend={self.backend!r} unreachable after "
                    f"{self.max_retries + 1} attempts (model={self.model}): {last_error}. {hint}"
                )
            if candidate is not None:
                valid.append(candidate)

        self.total_llm_valid += len(valid)

        # Backfill to exactly pool_size if the LLM returned fewer valid,
        # non-hallucinated candidates than requested -- guarantees the
        # orchestrator always gets a full pool, while the reject rate
        # (total_backfilled / total_requested) is tracked and should be
        # reported, not hidden (research plan §6.1 item 18).
        if len(valid) < pool_size:
            n_missing = pool_size - len(valid)
            self.total_backfilled += n_missing
            valid.extend(self._fallback.propose_pool(archive, n_missing, ledger_bias))
        elif len(valid) > pool_size:
            valid = valid[:pool_size]

        return valid

    @property
    def reject_rate(self) -> float:
        return self.total_backfilled / self.total_requested if self.total_requested else 0.0
