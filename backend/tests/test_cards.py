from neuroforge.cards import get_card, list_cards


def test_list_cards():
    cards = list_cards()
    ids = {c["id"] for c in cards}
    assert {"state_estimator", "binding_surrogate", "generator", "pkpd", "eeg_simulator"} <= ids
    assert all(c["title"] for c in cards)


def test_get_card_content():
    content = get_card("state_estimator")
    assert content is not None
    assert "Model Card" in content
    assert "NOT intended for" in content


def test_get_unknown_card():
    assert get_card("does_not_exist") is None
    assert get_card("../config") is None  # path traversal guarded
