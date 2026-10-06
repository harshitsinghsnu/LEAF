"""Research plan §5.5: E5 must be downgraded to 'consistency_check' if the
bare LLM already names the target finding pre-RAG."""
from leaf.agents.leakage_probe import run_leakage_probe, e5_status


def test_leak_detected_downgrades_status():
    def leaky_llm(q):
        return "Brominated tails have recently been shown to enhance transfection."
    results = run_leakage_probe(leaky_llm)
    assert any(r.leaked for r in results)
    assert e5_status(results) == "consistency_check"


def test_no_leak_keeps_external_validation():
    def clean_llm(q):
        return "I don't have specific information on that."
    results = run_leakage_probe(clean_llm)
    assert all(not r.leaked for r in results)
    assert e5_status(results) == "external_validation"
