import numpy as np

from core.data_structures import Identity, CandidateSet, BeliefState, Attribute
from core.config import RetrievalConfig
from modules.constraint_rerank import ConstraintReranker

def test_rejected_attribute_penalizes_candidates():
    cfg = RetrievalConfig()
    cfg.negation_penalty_alpha = 2.0
    cfg.positive_boost_beta = 0.0
    cfg.top_m_views = 1

    reranker = ConstraintReranker(cfg)

    q = np.array([1.0, 0.0], dtype=np.float32)

    bag = Attribute(
        name="bag",
        description="carrying a bag",
        embedding=np.array([0.0, 1.0], dtype=np.float32),
    )

    # p1: matches query, not bag
    p1 = Identity("p1", ["a.jpg"], embeddings=np.array([[1.0, 0.0]], dtype=np.float32))

    # p2: matches query but also matches bag strongly -> should be penalized when bag is rejected
    p2 = Identity("p2", ["b.jpg"], embeddings=np.array([[0.8, 0.6]], dtype=np.float32))

    cs = CandidateSet([p1, p2], scores=np.array([0.0, 0.0], dtype=np.float32))

    b = BeliefState()
    b.add_rejected(bag, confidence=1.0)

    out = reranker.rerank_with_constraints(cs, q, b)

    assert out.identities[0].identity_id == "p1"
