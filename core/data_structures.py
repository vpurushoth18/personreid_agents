"""
Core data structures for Agentic ReID
Defines classes for dialogue context, belief state, and identity information
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import numpy as np


class AnswerType(Enum):
    """Enumeration for user answer types"""
    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"


@dataclass
class Attribute:
    """Represents a visual attribute with confidence"""
    name: str
    description: str
    embedding: Optional[np.ndarray] = None
    confidence: float = 1.0
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Attribute):
            return self.name == other.name
        return False


@dataclass
class DialogueTurn:
    """Represents a single turn in the dialogue"""
    turn_id: int
    question: str
    answer: AnswerType
    attribute: Optional[Attribute] = None
    timestamp: Optional[float] = None


@dataclass
class DialogueContext:
    """Maintains the full dialogue history"""
    initial_description: str
    turns: List[DialogueTurn] = field(default_factory=list)
    reformulated_query: Optional[str] = None
    
    def add_turn(self, question: str, answer: AnswerType, attribute: Optional[Attribute] = None):
        """Add a new dialogue turn"""
        turn = DialogueTurn(
            turn_id=len(self.turns),
            question=question,
            answer=answer,
            attribute=attribute
        )
        self.turns.append(turn)
    
    def get_history_string(self) -> str:
        """Get formatted dialogue history as string"""
        history = [f"Initial: {self.initial_description}"]
        for turn in self.turns:
            history.append(f"Q{turn.turn_id}: {turn.question}")
            history.append(f"A{turn.turn_id}: {turn.answer.value}")
        return "\n".join(history)
    
    def get_confirmed_attributes(self) -> List[Attribute]:
        """Get all attributes confirmed with YES"""
        return [turn.attribute for turn in self.turns 
                if turn.answer == AnswerType.YES and turn.attribute is not None]
    
    def get_rejected_attributes(self) -> List[Attribute]:
        """Get all attributes rejected with NO"""
        return [turn.attribute for turn in self.turns 
                if turn.answer == AnswerType.NO and turn.attribute is not None]


@dataclass
class BeliefState:
    """
    Maintains explicit constraint state over attributes
    Implements the soft logic state for constraint-based reranking
    """
    confirmed_attributes: Dict[str, Tuple[Attribute, float]] = field(default_factory=dict)
    rejected_attributes: Dict[str, Tuple[Attribute, float]] = field(default_factory=dict)
    uncertain_attributes: Set[str] = field(default_factory=set)
    
    def add_confirmed(self, attribute: Attribute, confidence: float = 1.0):
        """Add a confirmed (YES) attribute"""
        # Remove from rejected if it was there (handle conflicts)
        if attribute.name in self.rejected_attributes:
            del self.rejected_attributes[attribute.name]
        
        # Update or add to confirmed
        self.confirmed_attributes[attribute.name] = (attribute, confidence)
        self.uncertain_attributes.discard(attribute.name)
    
    def add_rejected(self, attribute: Attribute, confidence: float = 1.0):
        """Add a rejected (NO) attribute"""
        # Remove from confirmed if it was there (handle conflicts)
        if attribute.name in self.confirmed_attributes:
            del self.confirmed_attributes[attribute.name]
        
        # Update or add to rejected
        self.rejected_attributes[attribute.name] = (attribute, confidence)
        self.uncertain_attributes.discard(attribute.name)
    
    def add_uncertain(self, attribute_name: str):
        """Mark an attribute as uncertain"""
        self.uncertain_attributes.add(attribute_name)
    
    def is_contradictory(self, attribute: Attribute, answer: AnswerType) -> bool:
        """Check if adding this attribute would contradict existing beliefs"""
        if answer == AnswerType.YES:
            return attribute.name in self.rejected_attributes
        elif answer == AnswerType.NO:
            return attribute.name in self.confirmed_attributes
        return False
    
    def get_all_constraints(self) -> Dict[str, List[Tuple[Attribute, float]]]:
        """Get all constraints organized by type"""
        return {
            'confirmed': list(self.confirmed_attributes.values()),
            'rejected': list(self.rejected_attributes.values())
        }


@dataclass
class Identity:
    """
    Represents a person identity with multiple views/images
    This is for identity-level retrieval (Upgrade 1)
    """
    identity_id: str
    image_paths: List[str]
    embeddings: np.ndarray  # Shape: (num_views, embedding_dim)
    ground_truth_caption: Optional[str] = None
    ground_truth_attributes: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate that embeddings match number of images"""
        if len(self.image_paths) != self.embeddings.shape[0]:
            raise ValueError(
                f"Number of images ({len(self.image_paths)}) must match "
                f"number of embeddings ({self.embeddings.shape[0]})"
            )
    
    @property
    def num_views(self) -> int:
        """Get number of views for this identity"""
        return len(self.image_paths)


@dataclass
class CandidateSet:
    """Represents a set of candidate identities during retrieval"""
    identities: List[Identity]
    scores: np.ndarray  # Shape: (num_identities,)
    
    def __post_init__(self):
        """Validate scores match identities"""
        if len(self.identities) != len(self.scores):
            raise ValueError("Number of identities must match number of scores")
    
    def get_top_k(self, k: int) -> 'CandidateSet':
        """Get top-k candidates by score"""
        top_k_indices = np.argsort(self.scores)[::-1][:k]
        return CandidateSet(
            identities=[self.identities[i] for i in top_k_indices],
            scores=self.scores[top_k_indices]
        )
    
    def __len__(self) -> int:
        return len(self.identities)


@dataclass
class Question:
    """Represents a candidate question for the user"""
    text: str
    attribute: Attribute
    expected_yes_elimination: float = 0.0
    expected_no_elimination: float = 0.0
    expected_remaining_mass: float = 1.0
    visual_confidence: float = 0.0
    information_gain: float = 0.0
    _selection_score: float = 0.0
    
    @property
    def selection_score(self) -> float:
        """Combined score for question selection"""
        if self._selection_score > 0:
            return self._selection_score
        # Lower remaining mass is better (more elimination)
        elimination_score = 1.0 - self.expected_remaining_mass
        # Weight by visual confidence
        return elimination_score * self.visual_confidence
    
    @selection_score.setter
    def selection_score(self, value: float):
        """Set selection score"""
        self._selection_score = value