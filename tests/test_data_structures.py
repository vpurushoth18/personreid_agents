import numpy as np
import pytest

from core.data_structures import (
    AnswerType, Attribute, BeliefState, DialogueContext, Identity, CandidateSet, Question
)

def test_dialogue_context_add_turn_and_history():
    ctx = DialogueContext("white male")
    attr = Attribute(name="bag", description="carrying a bag")
    ctx.add_turn("Is the person carrying a bag?", AnswerType.NO, attribute=attr)

    assert len(ctx.turns) == 1
    assert "Initial: white male" in ctx.get_history_string()
    assert "Q0:" in ctx.get_history_string()
    assert ctx.get_rejected_attributes()[0].name == "bag"
    assert ctx.get_confirmed_attributes() == []

def test_belief_state_add_confirmed_rejected_and_conflict_resolution():
    b = BeliefState()
    hat = Attribute(name="hat", description="wearing a hat")

    b.add_confirmed(hat, confidence=0.8)
    assert "hat" in b.confirmed_attributes
    assert "hat" not in b.rejected_attributes

    # If later user says NO, it should move across
    b.add_rejected(hat, confidence=0.9)
    assert "hat" not in b.confirmed_attributes
    assert "hat" in b.rejected_attributes

def test_belief_state_contradiction_check():
    b = BeliefState()
    glasses = Attribute(name="glasses", description="wearing glasses")

    b.add_confirmed(glasses)
    assert b.is_contradictory(glasses, AnswerType.NO) is True
    assert b.is_contradictory(glasses, AnswerType.YES) is False

def test_identity_validates_num_views_matches_embeddings():
    emb = np.zeros((3, 512), dtype=np.float32)
    Identity(identity_id="p1", image_paths=["a.jpg", "b.jpg", "c.jpg"], embeddings=emb)

    bad_emb = np.zeros((2, 512), dtype=np.float32)
    with pytest.raises(ValueError):
        Identity(identity_id="p1", image_paths=["a.jpg", "b.jpg", "c.jpg"], embeddings=bad_emb)

def test_candidate_set_top_k():
    emb = np.zeros((1, 512), dtype=np.float32)
    ids = [
        Identity("p1", ["a.jpg"], emb),
        Identity("p2", ["b.jpg"], emb),
        Identity("p3", ["c.jpg"], emb),
    ]
    scores = np.array([0.1, 0.9, 0.3], dtype=np.float32)
    cs = CandidateSet(ids, scores)

    top2 = cs.get_top_k(2)
    assert len(top2) == 2
    assert top2.identities[0].identity_id == "p2"
    assert top2.identities[1].identity_id == "p3"

def test_question_selection_score_default():
    q = Question(
        text="Is the person wearing a hat?",
        attribute=Attribute(name="hat", description="hat"),
        expected_remaining_mass=0.25,
        visual_confidence=0.8,
    )
    # score = (1 - remaining_mass) * visual_confidence
    assert abs(q.selection_score - ((1 - 0.25) * 0.8)) < 1e-6

def test_question_selection_score_override():
    q = Question(
        text="Is the person wearing a hat?",
        attribute=Attribute(name="hat", description="hat"),
        expected_remaining_mass=0.99,
        visual_confidence=0.01,
    )
    q.selection_score = 123.0
    assert q.selection_score == 123.0
