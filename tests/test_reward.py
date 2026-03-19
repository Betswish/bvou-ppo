from bvou_ppo.reward import exact_match_reward, normalize_prediction


def test_boolq_reward():
    assert exact_match_reward("boolq", "yes", "yes") == 1.0
    assert exact_match_reward("boolq", "No.", "yes") == 0.0


def test_option_reward():
    assert normalize_prediction("commonsenseqa", "The answer is C") == "C"
    assert exact_match_reward("arc_challenge", "B", "B") == 1.0
