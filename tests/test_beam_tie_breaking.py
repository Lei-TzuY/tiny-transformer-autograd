import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nn.beam import _rank_beam_tokens, _select_children


def test_beam_token_ranking_uses_descending_token_id_for_exact_ties():
    log_probs = np.array([-2.0, -1.0, -1.0, -1.0, -3.0])

    assert _rank_beam_tokens(log_probs, 1).tolist() == [3]
    assert _rank_beam_tokens(log_probs, 2).tolist() == [3, 2]
    assert _rank_beam_tokens(log_probs, 3).tolist() == [3, 2, 1]


def test_beam_tie_winner_is_independent_of_beam_width():
    sequence = np.array([[0]], dtype=np.int64)
    logits = np.zeros((1, 1, 5), dtype=np.float64)
    beams = [(sequence, 0.0, logits, None)]

    winners = []
    for beam_width in (1, 2, 3, 5, 8):
        selected = _select_children(beams, beam_width, 1.0)
        winners.append(int(selected[0][0][0, -1]))

    assert winners == [4, 4, 4, 4, 4]


def test_beam_tie_breaking_does_not_override_higher_score():
    sequence = np.array([[0]], dtype=np.int64)
    logits = np.array([[[0.0, 2.0, 2.0, 1.0]]], dtype=np.float64)
    beams = [(sequence, 0.0, logits, None)]

    selected = _select_children(beams, 3, 1.0)
    tokens = [int(item[0][0, -1]) for item in selected]

    assert tokens == [2, 1, 3]
