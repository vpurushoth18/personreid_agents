"""
Identity-Level Retrieval Module (Upgrade 1)
Implements multi-view set aggregation for identity-level ranking
"""

import numpy as np
from typing import List, Tuple

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# from ..core.data_structures import Identity, CandidateSet
# from ..core.config import RetrievalConfig

from core.data_structures import Identity, CandidateSet
from core.config import RetrievalConfig


class IdentityRetriever:
    """
    Identity-level retrieval with multi-view set aggregation
    
    Instead of ranking individual images, this module ranks person identities
    by aggregating scores across all views/angles of the same person.
    """
    
    def __init__(self, config: RetrievalConfig):
        """
        Initialize the identity retriever
        
        Args:
            config: Retrieval configuration
        """
        self.config = config
        self.aggregation_method = config.aggregation_method
        self.top_m = config.top_m_views
    
    def compute_identity_scores(
        self,
        identities: List[Identity],
        query_embedding: np.ndarray,
        method: str = None
    ) -> np.ndarray:
        """
        Compute identity-level scores from multi-view embeddings
        
        Args:
            identities: List of Identity objects with multi-view embeddings
            query_embedding: Query embedding vector (D,)
            method: Aggregation method ('max', 'mean', 'top_m_mean')
        
        Returns:
            Identity scores array (N,)
        """
        if method is None:
            method = self.aggregation_method
        
        scores = []
        for identity in identities:
            score = self._aggregate_views(
                identity.embeddings,
                query_embedding,
                method
            )
            scores.append(score)
        
        return np.array(scores)
    
    def _aggregate_views(
        self,
        view_embeddings: np.ndarray,
        query_embedding: np.ndarray,
        method: str
    ) -> float:
        """
        Aggregate similarity scores across multiple views
        
        Args:
            view_embeddings: View embeddings (M, D) where M is number of views
            query_embedding: Query embedding (D,)
            method: Aggregation method
        
        Returns:
            Aggregated identity score
        """
        # Compute cosine similarities for all views
        # Normalize embeddings
        view_embeddings_norm = view_embeddings / (
            np.linalg.norm(view_embeddings, axis=1, keepdims=True) + 1e-8
        )
        query_embedding_norm = query_embedding / (
            np.linalg.norm(query_embedding) + 1e-8
        )
        
        # Cosine similarities: (M,)
        similarities = np.dot(view_embeddings_norm, query_embedding_norm)
        
        if method == "max":
            # Max pooling: take the best matching view
            return np.max(similarities)
        
        elif method == "mean":
            # Mean pooling: average across all views
            return np.mean(similarities)
        
        elif method == "top_m_mean":
            # Top-m mean pooling: average of top-m best views
            m = min(self.top_m, len(similarities))
            top_m_sims = np.partition(similarities, -m)[-m:]
            return np.mean(top_m_sims)
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def retrieve_top_k(
        self,
        identities: List[Identity],
        query_embedding: np.ndarray,
        k: int
    ) -> CandidateSet:
        """
        Retrieve top-k identities for a query
        
        Args:
            identities: List of all identities in gallery
            query_embedding: Query embedding
            k: Number of top identities to retrieve
        
        Returns:
            CandidateSet containing top-k identities and their scores
        """
        # Compute identity-level scores
        scores = self.compute_identity_scores(identities, query_embedding)
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        # Create candidate set
        top_k_identities = [identities[i] for i in top_k_indices]
        top_k_scores = scores[top_k_indices]
        
        return CandidateSet(
            identities=top_k_identities,
            scores=top_k_scores
        )
    
    def rerank_identities(
        self,
        candidate_set: CandidateSet,
        query_embedding: np.ndarray,
        method: str = None
    ) -> CandidateSet:
        """
        Re-rank identities in candidate set with potentially different method
        
        Args:
            candidate_set: Current candidate set
            query_embedding: Updated query embedding
            method: Aggregation method (optional override)
        
        Returns:
            Re-ranked candidate set
        """
        # Recompute scores
        new_scores = self.compute_identity_scores(
            candidate_set.identities,
            query_embedding,
            method
        )
        
        # Sort by new scores
        sorted_indices = np.argsort(new_scores)[::-1]
        
        return CandidateSet(
            identities=[candidate_set.identities[i] for i in sorted_indices],
            scores=new_scores[sorted_indices]
        )


class IdentityAggregationComparison:
    """
    Utility class for comparing different aggregation methods
    Used for ablation studies
    """
    
    @staticmethod
    def compare_methods(
        identities: List[Identity],
        query_embedding: np.ndarray,
        target_identity_id: str,
        methods: List[str] = None
    ) -> dict:
        """
        Compare different aggregation methods on the same query
        
        Args:
            identities: All identities
            query_embedding: Query embedding
            target_identity_id: Ground truth identity ID
            methods: List of methods to compare
        
        Returns:
            Dictionary with method -> (rank, score) mapping
        """
        if methods is None:
            methods = ["max", "mean", "top_m_mean"]
        
        results = {}
        config = RetrievalConfig()
        
        for method in methods:
            config.aggregation_method = method
            retriever = IdentityRetriever(config)
            
            # Get scores
            scores = retriever.compute_identity_scores(
                identities,
                query_embedding
            )
            
            # Find rank of target
            sorted_indices = np.argsort(scores)[::-1]
            identity_ids = [identities[i].identity_id for i in sorted_indices]
            
            try:
                rank = identity_ids.index(target_identity_id) + 1
                score = scores[sorted_indices[rank - 1]]
            except ValueError:
                rank = -1
                score = 0.0
            
            results[method] = {
                'rank': rank,
                'score': score
            }
        
        return results
