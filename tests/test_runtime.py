from riichi_analysis_engine.runtime import candidate_action_index


def test_candidate_action_mapping() -> None:
    assert candidate_action_index({"type": "dahai", "pai": "5mr"}) == 34
    assert candidate_action_index({"type": "reach"}) == 37
    assert candidate_action_index(
        {"type": "chi", "pai": "3m", "consumed": ["1m", "2m"]}
    ) == 40
    assert candidate_action_index({"type": "pon"}) == 41
    assert candidate_action_index({"type": "ankan"}) == 42
    assert candidate_action_index({"type": "hora"}) == 43
    assert candidate_action_index({"type": "ryukyoku"}) == 44
    assert candidate_action_index({"type": "none"}) == 45
