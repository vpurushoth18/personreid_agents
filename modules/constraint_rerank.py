"""
Constraint-Based Reranking Module (Upgrade 2)
Implements soft logic state and constraint enforcement for negation correctness
"""

import numpy as np
from typing import List, Dict, Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from core.data_structures import (
    Identity, CandidateSet, BeliefState, Attribute, AnswerType
)
from core.config import RetrievalConfig


class ConstraintReranker:
    """
    Constraint-based reranking with explicit belief state
    
    Maintains an explicit set of confirmed and rejected attributes,
    and uses them to rerank candidates with guaranteed negation correctness.
    """
    
    def __init__(self, config: RetrievalConfig, text_encoder=None):
        """
        Initialize constraint reranker
        
        Args:
            config: Retrieval configuration
            text_encoder: Text encoder for embedding attributes (optional)
        """
        self.config = config
        self.alpha = config.negation_penalty_alpha  # Weight for rejected attributes
        self.beta = config.positive_boost_beta  # Weight for confirmed attributes
        self.text_encoder = text_encoder
        
        # Cache for attribute embeddings
        self.attribute_embedding_cache = {}
    
    def rerank_with_constraints(
        self,
        candidate_set: CandidateSet,
        query_embedding: np.ndarray,
        belief_state: BeliefState
    ) -> CandidateSet:
        """
        Rerank candidates using constraint-based scoring
        
        Score formula:
        S(v) = cos(v, q) + β * Σ[w_a * cos(v, e(a)) for a in A+]
                          - α * Σ[w_a * cos(v, e(a)) for a in A-]
        
        Args:
            candidate_set: Current candidate identities
            query_embedding: Base query embedding
            belief_state: Current belief state with constraints
        
        Returns:
            Reranked candidate set
        """
        new_scores = []
        
        constraints = belief_state.get_all_constraints()
        confirmed = constraints['confirmed']
        rejected = constraints['rejected']
        
        for identity in candidate_set.identities:
            # For each identity, aggregate score across its views
            view_scores = []
            
            for view_embedding in identity.embeddings:
                # Base similarity
                base_score = self._cosine_similarity(view_embedding, query_embedding)
                
                # Positive boost from confirmed attributes
                positive_boost = 0.0
                if confirmed:
                    for attr, confidence in confirmed:
                        attr_emb = self._get_attribute_embedding(attr)
                        if attr_emb is not None:
                            attr_sim = self._cosine_similarity(view_embedding, attr_emb)
                            positive_boost += confidence * attr_sim
                    positive_boost = self.beta * positive_boost / len(confirmed)
                
                # Negative penalty from rejected attributes
                negative_penalty = 0.0
                if rejected:
                    for attr, confidence in rejected:
                        attr_emb = self._get_attribute_embedding(attr)
                        if attr_emb is not None:
                            attr_sim = self._cosine_similarity(view_embedding, attr_emb)
                            # Only penalize if similarity is positive (image shows rejected attr)
                            negative_penalty += confidence * max(0, attr_sim)
                    negative_penalty = self.alpha * negative_penalty / len(rejected)
                
                # Combined score for this view
                view_score = base_score + positive_boost - negative_penalty
                view_scores.append(view_score)
            
            # Aggregate across views (using top-m mean as default)
            identity_score = self._aggregate_view_scores(view_scores)
            new_scores.append(identity_score)
        
        # Convert to numpy and sort
        new_scores = np.array(new_scores)
        sorted_indices = np.argsort(new_scores)[::-1]
        
        return CandidateSet(
            identities=[candidate_set.identities[i] for i in sorted_indices],
            scores=new_scores[sorted_indices]
        )
    
    def compute_negation_penalty(
        self,
        image_embedding: np.ndarray,
        rejected_attributes: List[Tuple[Attribute, float]]
    ) -> float:
        """
        Compute the negation penalty for an image
        
        Args:
            image_embedding: Image embedding vector
            rejected_attributes: List of (attribute, confidence) tuples
        
        Returns:
            Negation penalty score
        """
        if not rejected_attributes:
            return 0.0
        
        penalties = []
        for attr, confidence in rejected_attributes:
            attr_emb = self._get_attribute_embedding(attr)
            if attr_emb is not None:
                similarity = self._cosine_similarity(image_embedding, attr_emb)
                # Only penalize positive similarities (image shows rejected attribute)
                penalties.append(confidence * max(0, similarity))
        
        if not penalties:
            return 0.0
        
        # Return max penalty (strictest rejection)
        return self.alpha * max(penalties)
    
    def compute_confirmation_boost(
        self,
        image_embedding: np.ndarray,
        confirmed_attributes: List[Tuple[Attribute, float]]
    ) -> float:
        """
        Compute the confirmation boost for an image
        
        Args:
            image_embedding: Image embedding vector
            confirmed_attributes: List of (attribute, confidence) tuples
        
        Returns:
            Confirmation boost score
        """
        if not confirmed_attributes:
            return 0.0
        
        boosts = []
        for attr, confidence in confirmed_attributes:
            attr_emb = self._get_attribute_embedding(attr)
            if attr_emb is not None:
                similarity = self._cosine_similarity(image_embedding, attr_emb)
                boosts.append(confidence * similarity)
        
        if not boosts:
            return 0.0
        
        # Return mean boost
        return self.beta * np.mean(boosts)
    
    def _get_attribute_embedding(self, attribute: Attribute) -> np.ndarray:
        """
        Get embedding for an attribute (with caching)
        
        Args:
            attribute: Attribute object
        
        Returns:
            Attribute embedding or None if encoder not available
        """
        # Check cache first
        if attribute.name in self.attribute_embedding_cache:
            return self.attribute_embedding_cache[attribute.name]
        
        # If attribute already has embedding, use it
        if attribute.embedding is not None:
            self.attribute_embedding_cache[attribute.name] = attribute.embedding
            return attribute.embedding
        
        # Otherwise, encode using text encoder
        if self.text_encoder is not None:
            # Format attribute as text for encoding
            text = f"person {attribute.description}"
            embedding = self.text_encoder.encode_text(text)
            
            # Cache and return
            self.attribute_embedding_cache[attribute.name] = embedding
            return embedding
        
        return None
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return float(np.dot(vec1_norm, vec2_norm))
    
    def _aggregate_view_scores(self, view_scores: List[float]) -> float:
        """Aggregate scores across multiple views"""
        if not view_scores:
            return 0.0
        
        # Use top-m mean pooling
        m = min(self.config.top_m_views, len(view_scores))
        top_m = sorted(view_scores, reverse=True)[:m]
        return np.mean(top_m)
    
    def evaluate_constraint_violations(
        self,
        candidate_set: CandidateSet,
        belief_state: BeliefState,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate how well candidates satisfy constraints
        
        Args:
            candidate_set: Candidates to evaluate
            belief_state: Current constraints
            threshold: Similarity threshold for violation detection
        
        Returns:
            Dictionary with violation statistics
        """
        constraints = belief_state.get_all_constraints()
        rejected = constraints['rejected']
        
        if not rejected:
            return {
                'total_violations': 0,
                'violation_rate': 0.0,
                'avg_violation_score': 0.0
            }
        
        violations = []
        violation_scores = []
        
        for identity in candidate_set.identities:
            has_violation = False
            max_violation_score = 0.0
            
            for view_embedding in identity.embeddings:
                for attr, confidence in rejected:
                    attr_emb = self._get_attribute_embedding(attr)
                    if attr_emb is not None:
                        similarity = self._cosine_similarity(view_embedding, attr_emb)
                        if similarity > threshold:
                            has_violation = True
                            max_violation_score = max(max_violation_score, similarity)
            
            if has_violation:
                violations.append(identity.identity_id)
                violation_scores.append(max_violation_score)
        
        return {
            'total_violations': len(violations),
            'violation_rate': len(violations) / len(candidate_set) if len(candidate_set) > 0 else 0,
            'avg_violation_score': np.mean(violation_scores) if violation_scores else 0.0,
            'violated_identities': violations
        }


class BeliefStateManager:
    """
    Manages belief state updates based on user feedback
    Handles conflicts and maintains consistency
    """
    
    def __init__(self):
        self.belief_state = BeliefState()
    
    def update_from_answer(
        self,
        attribute: Attribute,
        answer: AnswerType,
        confidence: float = 1.0
    ) -> bool:
        """
        Update belief state from user answer
        
        Args:
            attribute: Attribute being answered about
            answer: User's answer (YES/NO/UNCERTAIN)
            confidence: Confidence in the answer
        
        Returns:
            True if update successful, False if contradictory
        """
        # Check for contradiction
        if self.belief_state.is_contradictory(attribute, answer):
            # Handle conflict - in real system, might want to ask user
            # For now, trust the most recent answer
            if answer == AnswerType.YES:
                self.belief_state.add_confirmed(attribute, confidence)
            elif answer == AnswerType.NO:
                self.belief_state.add_rejected(attribute, confidence)
            return False
        
        # Update based on answer
        if answer == AnswerType.YES:
            self.belief_state.add_confirmed(attribute, confidence)
        elif answer == AnswerType.NO:
            self.belief_state.add_rejected(attribute, confidence)
        elif answer == AnswerType.UNCERTAIN:
            self.belief_state.add_uncertain(attribute.name)
        
        return True
    
    def get_state(self) -> BeliefState:
        """Get current belief state"""
        return self.belief_state
    
    def reset(self):
        """Reset belief state"""
        self.belief_state = BeliefState()