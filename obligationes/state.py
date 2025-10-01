"""
State management for obligationes disputations.

This module defines the core data structures for tracking the state of a disputation,
including propositions, responses, and commitment sets.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Set, List, Optional, Dict, Any
import json
from datetime import datetime


class ResponseType(Enum):
    """Possible responses in obligationes disputation."""

    CONCEDO = "concedo"  # Grant/accept the proposition
    NEGO = "nego"  # Deny the proposition
    DUBITO = "dubito"  # Doubt (neither grant nor deny)

    def __str__(self) -> str:
        return self.value.upper()


class PropositionStatus(Enum):
    """Status of a proposition in the disputation."""

    POSITUM = "positum"  # Initial position to defend
    CONCESSA = "concessa"  # Granted proposition
    NEGATA = "negata"  # Denied proposition
    DUBITATA = "dubitata"  # Doubted proposition
    IRRELEVANT = "irrelevant"  # Not yet addressed

    def __str__(self) -> str:
        return self.value


@dataclass
class Proposition:
    """
    Represents a proposition in the disputation.

    Attributes:
        content: Natural language proposition text
        status: Current status in the disputation
        turn_introduced: Turn number when proposition was introduced
        follows_from: List of proposition contents this follows from
        incompatible_with: List of proposition contents this is incompatible with
        inference_chain: Optional explanation of how this was derived
    """

    content: str
    status: PropositionStatus
    turn_introduced: int
    follows_from: List[str] = field(default_factory=list)
    incompatible_with: List[str] = field(default_factory=list)
    inference_chain: Optional[str] = None

    def __hash__(self) -> int:
        """Hash based on content for use in sets."""
        return hash(self.content)

    def __eq__(self, other) -> bool:
        """Equality based on content."""
        if isinstance(other, Proposition):
            return self.content == other.content
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "status": self.status.value,
            "turn_introduced": self.turn_introduced,
            "follows_from": self.follows_from,
            "incompatible_with": self.incompatible_with,
            "inference_chain": self.inference_chain,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Proposition":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            status=PropositionStatus(data["status"]),
            turn_introduced=data["turn_introduced"],
            follows_from=data.get("follows_from", []),
            incompatible_with=data.get("incompatible_with", []),
            inference_chain=data.get("inference_chain"),
        )


@dataclass
class Turn:
    """
    Represents a single turn in the disputation.

    Attributes:
        number: Turn number (0-indexed)
        proposition: The proposition proposed by the Opponent
        response: The Respondent's response
        reasoning: Explanation for the response
        rule_applied: Which of Burley's rules was applied (1-5)
        consistency_maintained: Whether consistency was maintained after this turn
    """

    number: int
    proposition: str
    response: ResponseType
    reasoning: str
    rule_applied: int
    consistency_maintained: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "number": self.number,
            "proposition": self.proposition,
            "response": self.response.value,
            "reasoning": self.reasoning,
            "rule_applied": self.rule_applied,
            "consistency_maintained": self.consistency_maintained,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Turn":
        """Create from dictionary."""
        return cls(
            number=data["number"],
            proposition=data["proposition"],
            response=ResponseType(data["response"]),
            reasoning=data["reasoning"],
            rule_applied=data["rule_applied"],
            consistency_maintained=data["consistency_maintained"],
        )


class ObligationesState:
    """
    Maintains complete state of an obligationes disputation.

    This is the single source of truth for the game state. State updates
    are treated as immutable transitions to support debugging and replay.

    Attributes:
        positum: The initial position the Respondent must defend
        concessa: Set of propositions the Respondent has granted
        negata: Set of propositions the Respondent has denied
        dubitata: Set of propositions the Respondent has doubted
        turn_count: Current turn number
        disputation_active: Whether the disputation is ongoing
        history: Complete history of turns
        common_knowledge: Background facts both parties accept
    """

    def __init__(self, common_knowledge: Optional[Set[str]] = None):
        """
        Initialize a new disputation state.

        Args:
            common_knowledge: Set of background facts both parties accept
        """
        self.positum: Optional[Proposition] = None
        self.concessa: Set[Proposition] = set()
        self.negata: Set[Proposition] = set()
        self.dubitata: Set[Proposition] = set()
        self.turn_count: int = 0
        self.disputation_active: bool = False
        self.history: List[Turn] = []
        self.common_knowledge: Set[str] = common_knowledge or set()
        self.metadata: Dict[str, Any] = {
            "started_at": None,
            "ended_at": None,
            "winner": None,
            "reason": None,
        }

    def set_positum(self, proposition: str) -> None:
        """
        Set the positum (initial position to defend).

        Args:
            proposition: The positum proposition
        """
        self.positum = Proposition(
            content=proposition, status=PropositionStatus.POSITUM, turn_introduced=0
        )
        self.disputation_active = True
        self.metadata["started_at"] = datetime.utcnow().isoformat()

    def add_response(
        self,
        proposition: str,
        response: ResponseType,
        reasoning: str,
        rule_applied: int,
    ) -> None:
        """
        Record a response and update state accordingly.

        Args:
            proposition: The proposition being responded to
            response: The response type (CONCEDO, NEGO, DUBITO)
            reasoning: Explanation for the response
            rule_applied: Which Burley rule was applied (1-5)
        """
        # Create proposition object
        prop = Proposition(
            content=proposition,
            status=self._response_to_status(response),
            turn_introduced=self.turn_count,
        )

        # Add to appropriate set
        if response == ResponseType.CONCEDO:
            self.concessa.add(prop)
        elif response == ResponseType.NEGO:
            self.negata.add(prop)
        elif response == ResponseType.DUBITO:
            self.dubitata.add(prop)

        # Record turn in history
        turn = Turn(
            number=self.turn_count,
            proposition=proposition,
            response=response,
            reasoning=reasoning,
            rule_applied=rule_applied,
            consistency_maintained=True,  # Will be updated by consistency check
        )
        self.history.append(turn)

        self.turn_count += 1

    def get_all_commitments(self) -> Set[str]:
        """
        Get all propositions the Respondent is committed to.

        Returns:
            Set of proposition contents (positum + concessa)
        """
        commitments = set()
        if self.positum:
            commitments.add(self.positum.content)
        commitments.update(p.content for p in self.concessa)
        return commitments

    def get_all_negations(self) -> Set[str]:
        """
        Get all propositions the Respondent has denied.

        Returns:
            Set of negated proposition contents
        """
        return {p.content for p in self.negata}

    def get_all_doubts(self) -> Set[str]:
        """
        Get all propositions the Respondent has doubted.

        Returns:
            Set of doubted proposition contents
        """
        return {p.content for p in self.dubitata}

    def end_disputation(self, winner: str, reason: str) -> None:
        """
        Mark the disputation as ended.

        Args:
            winner: "OPPONENT" or "RESPONDENT"
            reason: Explanation for why this party won
        """
        self.disputation_active = False
        self.metadata["ended_at"] = datetime.utcnow().isoformat()
        self.metadata["winner"] = winner
        self.metadata["reason"] = reason

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize state to dictionary.

        Returns:
            Dictionary representation of the state
        """
        return {
            "positum": self.positum.to_dict() if self.positum else None,
            "concessa": [p.to_dict() for p in self.concessa],
            "negata": [p.to_dict() for p in self.negata],
            "dubitata": [p.to_dict() for p in self.dubitata],
            "turn_count": self.turn_count,
            "disputation_active": self.disputation_active,
            "history": [turn.to_dict() for turn in self.history],
            "common_knowledge": list(self.common_knowledge),
            "metadata": self.metadata,
        }

    def to_json(self, filepath: Optional[str] = None) -> str:
        """
        Serialize state to JSON.

        Args:
            filepath: Optional path to save JSON file

        Returns:
            JSON string representation
        """
        json_str = json.dumps(self.to_dict(), indent=2)

        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)

        return json_str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObligationesState":
        """
        Deserialize state from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ObligationesState instance
        """
        state = cls(common_knowledge=set(data.get("common_knowledge", [])))

        if data.get("positum"):
            state.positum = Proposition.from_dict(data["positum"])

        state.concessa = {Proposition.from_dict(p) for p in data.get("concessa", [])}
        state.negata = {Proposition.from_dict(p) for p in data.get("negata", [])}
        state.dubitata = {Proposition.from_dict(p) for p in data.get("dubitata", [])}
        state.turn_count = data.get("turn_count", 0)
        state.disputation_active = data.get("disputation_active", False)
        state.history = [Turn.from_dict(t) for t in data.get("history", [])]
        state.metadata = data.get("metadata", {})

        return state

    @classmethod
    def from_json(cls, json_str: str) -> "ObligationesState":
        """
        Deserialize state from JSON string.

        Args:
            json_str: JSON string or filepath

        Returns:
            ObligationesState instance
        """
        # Try to parse as JSON string first
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # If that fails, try reading as file
            with open(json_str, "r") as f:
                data = json.load(f)

        return cls.from_dict(data)

    def _response_to_status(self, response: ResponseType) -> PropositionStatus:
        """Convert ResponseType to PropositionStatus."""
        mapping = {
            ResponseType.CONCEDO: PropositionStatus.CONCESSA,
            ResponseType.NEGO: PropositionStatus.NEGATA,
            ResponseType.DUBITO: PropositionStatus.DUBITATA,
        }
        return mapping[response]

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"ObligationesState(positum={self.positum.content if self.positum else None}, "
            f"concessa={len(self.concessa)}, negata={len(self.negata)}, "
            f"dubitata={len(self.dubitata)}, turn={self.turn_count}, "
            f"active={self.disputation_active})"
        )
