import numpy as np

from core.config import RetrievalConfig
from core.data_structures import Identity
from modules.identity_retrieval import IdentityRetriever

def test_retrieve_top_k_identities_topm_mean():
    cfg = RetrievalConfig(aggregation_method="top_m_mean", top_m_views=2)
    retriever = IdentityRetriever(cfg)

    # Two IDs, each with 3 views, 2D embeddings
    q = np.array([1.0, 0.0], dtype=np.float32)

    id1_views = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)  # strong match
    id2_views = np.array([[0.0, 1.0], [0.2, 0.8], [0.1, 0.9]], dtype=np.float32)  # weak match

    identities = [
        Identity(identity_id="p1", image_paths=["a","b","c"], embeddings=id1_views),
        Identity(identity_id="p2", image_paths=["d","e","f"], embeddings=id2_views),
    ]

    cs = retriever.retrieve_top_k(identities, q, k=1)
    assert len(cs.identities) == 1
    assert cs.identities[0].identity_id == "p1"
