"""
Question Selection via Counterfactual Expected Elimination (Upgrade 3)
Implements decision-theoretic question selection to maximize identity elimination
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.stats import entropy
from scipy.spatial.distance import euclidean

from core.data_structures import (
    Identity, CandidateSet, Question, Attribute, BeliefState
)
from core.config import RetrievalConfig


class CounterfactualQuestionSelector:
    """
    Selects questions by maximizing expected identity elimination
    
    Unlike KL divergence methods, this explicitly optimizes for reducing
    the number of remaining candidates under YES/NO outcomes.
    """
    
    def __init__(self, config: RetrievalConfig, attribute_verifier=None):
        """
        Initialize question selector
        
        Args:
            config: Retrieval configuration
            attribute_verifier: Module to verify attributes in images (optional)
        """
        self.config = config
        self.attribute_verifier = attribute_verifier
    
    def select_best_question(
        self,
        candidate_questions: List[Question],
        candidate_set: CandidateSet,
        belief_state: BeliefState
    ) -> Question:
        """
        Select the question that maximizes expected elimination
        
        Args:
            candidate_questions: Pool of candidate questions
            candidate_set: Current candidate identities
            belief_state: Current belief state
        
        Returns:
            Best question to ask
        """
        if not candidate_questions:
            raise ValueError("No candidate questions provided")
        
        # Compute selection score for each question
        for question in candidate_questions:
            self._compute_elimination_score(question, candidate_set)
        
        # Select question with best score
        best_question = max(candidate_questions, key=lambda q: q.selection_score)
        
        return best_question
    
    def _compute_elimination_score(
        self,
        question: Question,
        candidate_set: CandidateSet
    ):
        """
        Compute expected elimination score for a question
        
        Updates question object in-place with computed scores
        
        Args:
            question: Question to evaluate
            candidate_set: Current candidates
        """
        # Step 1: Partition candidates into YES and NO groups
        yes_group, no_group = self._partition_candidates(
            question.attribute,
            candidate_set
        )
        
        # Step 2: Compute probabilities
        total_mass = np.sum(candidate_set.scores)
        p_yes = np.sum([candidate_set.scores[i] for i in yes_group]) / total_mass
        p_no = np.sum([candidate_set.scores[i] for i in no_group]) / total_mass
        
        # Handle edge cases
        p_yes = max(min(p_yes, 0.99), 0.01)  # Avoid extreme probabilities
        p_no = 1.0 - p_yes
        
        # Step 3: Compute expected remaining mass after each answer
        remaining_mass_yes = np.sum([candidate_set.scores[i] for i in yes_group]) / total_mass
        remaining_mass_no = np.sum([candidate_set.scores[i] for i in no_group]) / total_mass
        
        # Step 4: Expected remaining mass
        expected_remaining_mass = p_yes * remaining_mass_yes + p_no * remaining_mass_no
        
        # Step 5: Compute visual confidence (separation in embedding space)
        visual_confidence = self._compute_visual_confidence(
            yes_group,
            no_group,
            candidate_set
        )
        
        # Update question object
        question.expected_yes_elimination = 1.0 - remaining_mass_yes
        question.expected_no_elimination = 1.0 - remaining_mass_no
        question.expected_remaining_mass = expected_remaining_mass
        question.visual_confidence = visual_confidence
        
        # Selection score (to be maximized)
        # Lower remaining mass is better (more elimination)
        elimination_score = 1.0 - expected_remaining_mass
        
        # Weight by visual confidence (avoid hallucinated attributes)
        question.selection_score = elimination_score * visual_confidence
    
    def _partition_candidates(
        self,
        attribute: Attribute,
        candidate_set: CandidateSet
    ) -> Tuple[List[int], List[int]]:
        """
        Partition candidates into YES and NO groups based on attribute
        
        Args:
            attribute: Attribute to verify
            candidate_set: Candidates to partition
        
        Returns:
            (yes_indices, no_indices) tuple
        """
        yes_group = []
        no_group = []
        
        if self.attribute_verifier is not None:
            # Use verifier if available
            for i, identity in enumerate(candidate_set.identities):
                if self.attribute_verifier.verify(identity, attribute):
                    yes_group.append(i)
                else:
                    no_group.append(i)
        else:
            # Fallback: use embedding similarity
            attr_embedding = attribute.embedding
            if attr_embedding is None:
                # If no embedding, split evenly (worst case)
                mid = len(candidate_set) // 2
                return list(range(mid)), list(range(mid, len(candidate_set)))
            
            for i, identity in enumerate(candidate_set.identities):
                # Compute max similarity across views
                max_sim = -1.0
                for view_emb in identity.embeddings:
                    sim = self._cosine_similarity(view_emb, attr_embedding)
                    max_sim = max(max_sim, sim)
                
                # Threshold-based partitioning
                if max_sim > 0.3:  # Threshold for "has attribute"
                    yes_group.append(i)
                else:
                    no_group.append(i)
        
        # Ensure both groups are non-empty
        if not yes_group:
            yes_group = [0]
            no_group = list(range(1, len(candidate_set)))
        elif not no_group:
            no_group = [len(candidate_set) - 1]
            yes_group = list(range(len(candidate_set) - 1))
        
        return yes_group, no_group
    
    def _compute_visual_confidence(
        self,
        yes_group: List[int],
        no_group: List[int],
        candidate_set: CandidateSet
    ) -> float:
        """
        Compute visual confidence as separation between YES and NO groups
        
        Higher confidence means the attribute creates clear visual separation
        
        Args:
            yes_group: Indices of YES candidates
            no_group: Indices of NO candidates
            candidate_set: Candidate set
        
        Returns:
            Visual confidence score (0 to 1)
        """
        # Compute centroids of each group
        yes_embeddings = []
        no_embeddings = []
        
        for idx in yes_group:
            identity = candidate_set.identities[idx]
            # Use mean of all views as representative
            yes_embeddings.append(np.mean(identity.embeddings, axis=0))
        
        for idx in no_group:
            identity = candidate_set.identities[idx]
            no_embeddings.append(np.mean(identity.embeddings, axis=0))
        
        if not yes_embeddings or not no_embeddings:
            return 0.5  # Neutral confidence
        
        yes_centroid = np.mean(yes_embeddings, axis=0)
        no_centroid = np.mean(no_embeddings, axis=0)
        
        # Euclidean distance between centroids
        distance = euclidean(yes_centroid, no_centroid)
        
        # Normalize to [0, 1] using sigmoid
        confidence = 1.0 / (1.0 + np.exp(-distance))
        
        return confidence
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity"""
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return float(np.dot(vec1_norm, vec2_norm))
    
    def compute_information_gain_kl(
        self,
        question: Question,
        candidate_set: CandidateSet
    ) -> float:
        """
        Compute information gain using KL divergence (for comparison)
        
        This is the baseline PlugIR method included for comparison
        
        Args:
            question: Question to evaluate
            candidate_set: Current candidates
        
        Returns:
            KL divergence (information gain)
        """
        # Partition candidates
        yes_group, no_group = self._partition_candidates(
            question.attribute,
            candidate_set
        )
        
        # Current distribution
        p_current = candidate_set.scores / np.sum(candidate_set.scores)
        
        # YES distribution
        p_yes = np.zeros_like(p_current)
        for idx in yes_group:
            p_yes[idx] = candidate_set.scores[idx]
        if np.sum(p_yes) > 0:
            p_yes = p_yes / np.sum(p_yes)
        
        # NO distribution
        p_no = np.zeros_like(p_current)
        for idx in no_group:
            p_no[idx] = candidate_set.scores[idx]
        if np.sum(p_no) > 0:
            p_no = p_no / np.sum(p_no)
        
        # KL divergences (with small epsilon to avoid log(0))
        eps = 1e-10
        kl_yes = entropy(p_yes + eps, p_current + eps)
        kl_no = entropy(p_no + eps, p_current + eps)
        
        # Expected KL divergence
        prob_yes = len(yes_group) / len(candidate_set)
        prob_no = len(no_group) / len(candidate_set)
        
        expected_kl = prob_yes * kl_yes + prob_no * kl_no
        
        return expected_kl


class AttributeVerifier:
    """
    Verifies whether an identity/image has a specific attribute
    Used for partitioning candidates during question selection
    """
    
    def __init__(self, text_encoder=None, threshold: float = 0.4):
        """
        Initialize attribute verifier
        
        Args:
            text_encoder: Text encoder for embeddings
            threshold: Similarity threshold for verification
        """
        self.text_encoder = text_encoder
        self.threshold = threshold
    
    def verify(self, identity: Identity, attribute: Attribute) -> bool:
        """
        Verify if an identity has an attribute
        
        Args:
            identity: Identity to verify
            attribute: Attribute to check
        
        Returns:
            True if identity has attribute, False otherwise
        """
        if attribute.embedding is None:
            return False
        
        # Check max similarity across all views
        max_similarity = -1.0
        for view_emb in identity.embeddings:
            sim = self._cosine_similarity(view_emb, attribute.embedding)
            max_similarity = max(max_similarity, sim)
        
        return max_similarity > self.threshold
    
    def verify_from_caption(
        self,
        ground_truth_caption: str,
        attribute: Attribute
    ) -> bool:
        """
        Verify attribute from ground truth caption (for simulation)
        
        Args:
            ground_truth_caption: Ground truth text description
            attribute: Attribute to verify
        
        Returns:
            True if attribute is in caption
        """
        # Simple keyword matching for simulation
        attribute_keywords = attribute.description.lower().split()
        caption_lower = ground_truth_caption.lower()
        
        # Check if any keyword appears in caption
        return any(keyword in caption_lower for keyword in attribute_keywords)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity"""
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return float(np.dot(vec1_norm, vec2_norm))