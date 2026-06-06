from mtp2_audio_tool_rl.rewards.tool_use_reward import (
    compute_tool_use_reward,
    exact_match_reward,
    normalize_answer,
)


def test_exact_answer_match():
    assert exact_match_reward("two speakers", "two speakers") == 1.0


def test_normalized_answer_match():
    assert normalize_answer("  Two,   Speakers! ") == "two speakers"
    assert exact_match_reward("Two, Speakers!", "two speakers") == 1.0


def test_wrong_answer():
    assert exact_match_reward("one speaker", "two speakers") == 0.0


def test_valid_tool_call_reward():
    reward = compute_tool_use_reward(
        prediction="two speakers",
        gold_answer="two speakers",
        tool_call_present=True,
        tool_call_valid=True,
        tool_was_needed=True,
        tool_result_used=True,
    )

    assert reward["answer_reward"] == 1.0
    assert reward["tool_validity_reward"] > 0
    assert reward["tool_helpfulness_reward"] > 0


def test_invalid_tool_penalty():
    reward = compute_tool_use_reward(
        prediction="two speakers",
        gold_answer="two speakers",
        tool_call_present=True,
        tool_call_valid=False,
        tool_was_needed=True,
    )

    assert reward["invalid_tool_penalty"] < 0


def test_unnecessary_tool_penalty():
    reward = compute_tool_use_reward(
        prediction="yes",
        gold_answer="yes",
        tool_call_present=True,
        tool_call_valid=True,
        tool_was_needed=False,
    )

    assert reward["unnecessary_tool_penalty"] < 0


def test_total_equals_sum_of_components():
    reward = compute_tool_use_reward(
        prediction="wrong",
        gold_answer="right",
        tool_call_present=True,
        tool_call_valid=False,
        tool_was_needed=False,
        tool_result_used=False,
    )
    component_total = (
        reward["answer_reward"]
        + reward["tool_validity_reward"]
        + reward["tool_helpfulness_reward"]
        + reward["unnecessary_tool_penalty"]
        + reward["invalid_tool_penalty"]
    )

    assert reward["total_reward"] == component_total
