"""
Question Selection via Counterfactual Expected Elimination (Upgrade 3)
Implements decision-theoretic question selection to maximize identity elimination.

Key properties:
- Works with your flat repo layout (core/ and modules/ are siblings)
- No SciPy dependency (cluster-friendly)
- Uses belief_state to avoid asking already-known attributes
- Handles zero/negative candidate scores safely
- Counterfactual elimination assumes hard update:
    YES -> keep YES group only
    NO  -> keep NO group only
  So expected remaining mass = p_yes^2 + p_no^2
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional

from core.data_structures import Identity, CandidateSet, Question, Attribute, BeliefState
from core.config import RetrievalConfig


class CounterfactualQuestionSelector:
    """
    Selects questions by maximizing expected identity elimination.

    For each candidate question, we partition IDs into YES / NO groups.
    Using current candidate probability mass P(id), we compute:
        p_yes = sum_{id in YES} P(id)
        p_no  = 1 - p_yes

    Under hard elimination updates (keep only the consistent set):
        remaining_mass_yes = p_yes
        remaining_mass_no  = p_no

    Expected remaining mass:
        E[remaining_mass] = p_yes * p_yes + p_no * p_no

    We maximize:
        selection_score = (1 - E[remaining_mass]) * visual_confidence
    """

    def __init__(self, config: RetrievalConfig, attribute_verifier: Optional["AttributeVerifier"] = None):
        self.config = config
        self.attribute_verifier = attribute_verifier

        # Optional knobs (safe if not present in RetrievalConfig)
        self.attr_presence_threshold = getattr(config, "attr_presence_threshold", 0.3)
        self.visual_confidence_temperature = getattr(config, "visual_confidence_temperature", 1.0)

    def select_best_question(
        self,
        candidate_questions: List[Question],
        candidate_set: CandidateSet,
        belief_state: BeliefState,
    ) -> Question:
        """
        Select the best question to ask next.

        Filters out questions whose attribute is already confirmed/rejected.
        Computes elimination-based score + visual confidence for each remaining question.

        Returns:
            The best Question
        """
        if not candidate_questions:
            raise ValueError("No candidate questions provided")

        # Filter out attributes that are already known
        constraints = belief_state.get_all_constraints()
        known = set([a.name for a, _ in constraints["confirmed"]] + [a.name for a, _ in constraints["rejected"]])
        remaining_questions = [q for q in candidate_questions if q.attribute.name not in known]

        if not remaining_questions:
            raise ValueError("No candidate questions left after filtering known constraints")

        # Score each question
        for q in remaining_questions:
            self._compute_elimination_score(q, candidate_set)

        # Pick the highest scoring
        return max(remaining_questions, key=lambda q: q.selection_score)

    def _compute_elimination_score(self, question: Question, candidate_set: CandidateSet) -> None:
        """
        Compute and store:
          - expected_remaining_mass
          - expected_yes_elimination / expected_no_elimination
          - visual_confidence
          - selection_score
        """
        yes_group, no_group = self._partition_candidates(question.attribute, candidate_set)

        # Normalize scores -> probability mass over IDs
        scores = candidate_set.scores.astype(np.float32)
        total = float(scores.sum())
        if total <= 0:
            p = np.ones_like(scores) / len(scores)
        else:
            p = scores / total

        # Counterfactual: if answer is YES we keep YES group only, etc.
        p_yes = float(p[yes_group].sum())
        p_no = 1.0 - p_yes

        remaining_mass_yes = p_yes
        remaining_mass_no = p_no

        expected_remaining_mass = p_yes * remaining_mass_yes + p_no * remaining_mass_no
        # equals p_yes^2 + p_no^2

        visual_confidence = self._compute_visual_confidence(yes_group, no_group, candidate_set)

        # Update question fields (paper-ready logging)
        question.expected_yes_elimination = 1.0 - remaining_mass_yes
        question.expected_no_elimination = 1.0 - remaining_mass_no
        question.expected_remaining_mass = expected_remaining_mass
        question.visual_confidence = visual_confidence

        elimination_score = 1.0 - expected_remaining_mass
        question.selection_score = elimination_score * visual_confidence

    def _partition_candidates(self, attribute: Attribute, candidate_set: CandidateSet) -> Tuple[List[int], List[int]]:
        """
        Partition candidates into YES and NO sets for a given attribute.
        Uses verifier if provided; otherwise uses embedding similarity.
        """
        yes_group: List[int] = []
        no_group: List[int] = []

        if self.attribute_verifier is not None:
            for i, identity in enumerate(candidate_set.identities):
                if self.attribute_verifier.verify(identity, attribute):
                    yes_group.append(i)
                else:
                    no_group.append(i)
        else:
            attr_embedding = attribute.embedding
            if attr_embedding is None:
                # Worst-case: cannot verify attribute -> split evenly
                mid = len(candidate_set) // 2
                return list(range(mid)), list(range(mid, len(candidate_set)))

            for i, identity in enumerate(candidate_set.identities):
                # max similarity across views
                max_sim = -1.0
                for view_emb in identity.embeddings:
                    sim = self._cosine_similarity(view_emb, attr_embedding)
                    if sim > max_sim:
                        max_sim = sim

                if max_sim > self.attr_presence_threshold:
                    yes_group.append(i)
                else:
                    no_group.append(i)

        # Ensure both groups are non-empty (avoid degenerate questions)
        if not yes_group:
            yes_group = [0]
            no_group = list(range(1, len(candidate_set)))
        elif not no_group:
            no_group = [len(candidate_set) - 1]
            yes_group = list(range(len(candidate_set) - 1))

        return yes_group, no_group

    def _compute_visual_confidence(self, yes_group: List[int], no_group: List[int], candidate_set: CandidateSet) -> float:
        """
        Visual confidence measures how separable YES vs NO groups are in embedding space.

        We represent each identity by the mean over its views.
        Then compute centroid distance and map it to (0,1) via a temperatured sigmoid.
        """
        yes_embs = [np.mean(candidate_set.identities[i].embeddings, axis=0) for i in yes_group]
        no_embs = [np.mean(candidate_set.identities[i].embeddings, axis=0) for i in no_group]

        if len(yes_embs) == 0 or len(no_embs) == 0:
            return 0.5

        yes_centroid = np.mean(np.stack(yes_embs, axis=0), axis=0)
        no_centroid = np.mean(np.stack(no_embs, axis=0), axis=0)

        distance = float(np.linalg.norm(yes_centroid - no_centroid))

        t = max(float(self.visual_confidence_temperature), 1e-6)
        return float(1.0 / (1.0 + np.exp(-distance / t)))

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return float(np.dot(vec1_norm, vec2_norm))


class AttributeVerifier:
    """
    Verifies whether an identity has a specific attribute.
    Used for partitioning candidates during question selection.
    """

    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold

    def verify(self, identity: Identity, attribute: Attribute) -> bool:
        """
        Returns True if max cosine similarity across views exceeds threshold.
        """
        if attribute.embedding is None:
            return False

        max_similarity = -1.0
        for view_emb in identity.embeddings:
            sim = self._cosine_similarity(view_emb, attribute.embedding)
            if sim > max_similarity:
                max_similarity = sim

        return max_similarity > self.threshold

    def verify_from_caption(self, ground_truth_caption: str, attribute: Attribute) -> bool:
        """
        Simple caption-based verifier for simulation.
        """
        if not ground_truth_caption:
            return False

        keywords = attribute.description.lower().split()
        cap = ground_truth_caption.lower()
        return any(k in cap for k in keywords)

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return float(np.dot(vec1_norm, vec2_norm))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Optional utility: KL(p || q) without SciPy.
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def expected_kl_information_gain(
    question: Question,
    candidate_set: CandidateSet,
    selector: CounterfactualQuestionSelector,
) -> float:
    """
    Optional baseline utility for PlugIR-style expected KL information gain.
    Uses the selector's partitioning logic for a fair comparison.
    """
    yes_group, no_group = selector._partition_candidates(question.attribute, candidate_set)

    scores = candidate_set.scores.astype(np.float32)
    total = float(scores.sum())
    if total <= 0:
        p_current = np.ones_like(scores) / len(scores)
    else:
        p_current = scores / total

    p_yes = np.zeros_like(p_current)
    p_yes[yes_group] = p_current[yes_group]
    if float(p_yes.sum()) > 0:
        p_yes = p_yes / float(p_yes.sum())

    p_no = np.zeros_like(p_current)
    p_no[no_group] = p_current[no_group]
    if float(p_no.sum()) > 0:
        p_no = p_no / float(p_no.sum())

    # Expected KL under answer distribution (mass-based, not count-based)
    prob_yes = float(p_current[yes_group].sum())
    prob_no = 1.0 - prob_yes

    kl_yes = kl_divergence(p_yes, p_current)
    kl_no = kl_divergence(p_no, p_current)

    return prob_yes * kl_yes + prob_no * kl_no
