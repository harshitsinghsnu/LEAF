"""Wrapper around the public LipoAgent code.

SCOPE CORRECTION (research plan §5.12), superseding an earlier version of
this file: LipoAgent's real, vendorable code
(https://github.com/SAI-Lab-NYU/LipoAgent) was inspected directly --
both its Predict and Verify agents (txgemma/predict_agent.py,
txgemma/verify_agent.py) load TxGemma-27B-Chat / Qwen3-8B LoRA adapters
and produce ONLY an `efficiency_score` (integer, 1-10, predicted mRNA
transfection performance). Neither script has any toxicity, safety, or
gating field anywhere in its input, prompt, or output.

The original description of LipoAgent as "toxicity-gated" (carried from an
unverified literature-search summary in the paper draft, marked [verify])
does not match the tool as published. LipoAgent is therefore NOT usable as
E4's binary toxicity-gate comparator -- that role is filled by
`leaf.agents.verifier.BinaryToxicityFilter` instead (Ablation (e)), which
is fully implemented and requires no external dependency.

This wrapper is kept only as a real efficiency-score client, in case a
future experiment wants LipoAgent's predicted potency (e.g. as an
additional Recall@k / potency-baseline signal, correctly scoped -- NOT as
a toxicity or calibration comparator). Vendor the repo before using it:

    git clone https://github.com/SAI-Lab-NYU/LipoAgent leaf/baselines/_vendor/lipoagent

Running inference additionally requires: `torch transformers peft
accelerate` (heavy GPU dependencies not in environment.yml/requirements.txt
by default), accepting Google's Gemma license on HuggingFace for the
TxGemma-27B-Chat base model, and a GPU with enough VRAM for a 27B-parameter
model. None of this is available in the default `leaf` conda environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

VENDOR_PATH = Path(__file__).parent / "_vendor" / "lipoagent"


@dataclass
class LipoAgentEfficiencyClient:
    """Real client for LipoAgent's efficiency_score output -- NOT a
    toxicity gate (see module docstring for why). Calls the vendored
    repo's predict_agent.py as a subprocess against a JSONL file in the
    schema its own script expects (ID, SMILES, amino_acid,
    protection_group, linker_length, tail_count, ester_bonds_OCOO), then
    parses data/txgemma_predict_results.json for the efficiency_score
    field.
    """

    vendor_path: Path = VENDOR_PATH

    def _check_vendored(self) -> None:
        if not self.vendor_path.exists():
            raise FileNotFoundError(
                f"LipoAgent is not vendored at {self.vendor_path}. Run:\n"
                f"  git clone https://github.com/SAI-Lab-NYU/LipoAgent {self.vendor_path}\n"
                "and install its dependencies (torch transformers peft accelerate) "
                "and accept the Gemma license on HuggingFace before calling this client."
            )

    def score_efficiency(self, candidates: list[dict]) -> np.ndarray:
        """candidates: list of dicts matching LipoAgent's expected input
        schema (see module docstring). Returns efficiency_score in [1, 10]
        per candidate. Requires a working GPU environment with the
        vendored repo's dependencies and model weights already downloaded
        -- this function does not manage that setup, it only calls it.
        """
        self._check_vendored()
        raise NotImplementedError(
            "Vendoring is confirmed possible (this repo is real and network-reachable), "
            "but running inference requires a GPU environment with torch/transformers/peft/"
            "accelerate installed and TxGemma-27B-Chat weights downloaded locally -- neither "
            "is available in this environment. Implement the subprocess call to "
            f"{VENDOR_PATH}/txgemma/predict_agent.py once that environment exists."
        )
