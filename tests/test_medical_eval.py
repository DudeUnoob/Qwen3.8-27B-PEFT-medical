from serving.medical_eval import (mcnemar_exact, parse_answer, parse_stated_answer, summarise, to_item,
                                  wilson)


def test_parse_answer_letters_and_ynm():
    assert parse_answer("B", "letter") == "B"
    assert parse_answer("The answer is C. Because...", "letter") == "C"
    assert parse_answer("<think>maybe A</think>D", "letter") == "D"
    assert parse_answer("none of these", "letter") is None
    assert parse_answer("Yes.", "ynm") == "yes"
    assert parse_answer("Maybe", "ynm") == "maybe"
    assert parse_answer("unclear", "ynm") is None


def test_parse_stated_answer_ignores_incidental_letters():
    assert parse_stated_answer("Vitamin A deficiency fits.\n\nThe correct answer is: **C. Zinc**", "letter") == "C"
    assert parse_stated_answer("Findings suggest...\n**B. Lupus** is most likely", "letter") == "B"
    assert parse_stated_answer("Vitamin A deficiency is unlikely here.", "letter") is None
    assert parse_stated_answer("The study shows...\n\n**Answer: yes**", "ynm") == "yes"


def test_summarise_lenient_uses_retry():
    records = [{"dataset": "medqa", "id": "0", "model": m, "parsed": None, "correct": False,
                "parsed_lenient": "A", "correct_lenient": True, "error": None}
               for m in ("qwen-medical", "Qwen/Qwen3-8B")]
    assert summarise(records)["medqa"]["qwen-medical"]["accuracy"] == 0
    assert summarise(records, "lenient")["medqa"]["qwen-medical"]["accuracy"] == 1


def test_to_item_formats():
    medqa = to_item("medqa", {"question": "Q?", "options": {k: k.lower() for k in "ABCD"}, "answer_idx": "C"})
    assert medqa["answer"] == "C" and "D. d" in medqa["prompt"]
    medmcqa = to_item("medmcqa", {"id": "x", "question": "Q?", "opa": "a", "opb": "b", "opc": "c", "opd": "d",
                                  "cop": 3, "subject_name": "Anatomy"})
    assert medmcqa["answer"] == "D" and medmcqa["subject"] == "Anatomy"
    pubmed = to_item("pubmedqa", {"pubid": 1, "question": "Q?", "context": {"contexts": ["c1", "c2"]},
                                  "final_decision": "no"})
    assert pubmed["answer"] == "no" and "c1\nc2" in pubmed["prompt"]


def test_statistics():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(20, 2) < 0.001
    low, high = wilson(50, 100)
    assert low < 0.5 < high


def test_summarise_pairs_models():
    records = []
    for i, (a, b) in enumerate([(True, True), (True, False), (False, False)]):
        for model, ok in (("qwen-medical", a), ("Qwen/Qwen3-8B", b)):
            records.append({"dataset": "medqa", "id": str(i), "model": model, "correct": ok,
                            "parsed": "A" if ok else "B", "error": None})
    entry = summarise(records)["medqa"]
    assert entry["questions"] == 3
    assert entry["qwen-medical"]["accuracy"] == 2 / 3
    assert entry["paired"]["adapter_only_correct"] == 1 and entry["paired"]["base_only_correct"] == 0


def test_letter_choice_from_top_logprobs():
    import math

    from serving.medical_eval import letter_choice
    top = [{"token": "B", "logprob": math.log(0.6)}, {"token": " C", "logprob": math.log(0.3)},
           {"token": "**", "logprob": math.log(0.05)}, {"token": "Because", "logprob": math.log(0.01)}]
    letter, probs = letter_choice(top)
    assert letter == "B" and set(probs) == {"B", "C"}
    assert letter_choice([{"token": "**", "logprob": 0.0}]) == (None, {})
