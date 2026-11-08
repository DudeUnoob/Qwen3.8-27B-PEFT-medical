from serving import parity


def test_first_difference():
    assert parity.first_difference("same", "same") is None
    assert parity.first_difference("abcd", "abXd") == 2
    assert parity.first_difference("abc", "abcdef") == 3


def test_compare_reports_exact_matches_and_shared_prefix():
    result = parity.compare(["aaaa", "bbbb"], ["aaaa", "bbXX"])
    assert result["exact_match_rate"] == 0.5
    assert result["mean_first_diff_char"] == 2
    assert result["mean_shared_prefix_fraction"] == 0.75


def test_parity_prompt_set_is_fixed():
    assert len(parity.PARITY_PROMPTS) == 16
    assert len(set(parity.PARITY_PROMPTS)) == 16
