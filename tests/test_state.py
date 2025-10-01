"""
Unit tests for state management.
"""

import json
from obligationes.state import (
    ResponseType,
    PropositionStatus,
    Proposition,
    Turn,
    ObligationesState,
)


class TestResponseType:
    """Tests for ResponseType enum."""

    def test_response_values(self):
        """Test that response types have correct values."""
        assert ResponseType.CONCEDO.value == "concedo"
        assert ResponseType.NEGO.value == "nego"
        assert ResponseType.DUBITO.value == "dubito"

    def test_response_string(self):
        """Test string representation."""
        assert str(ResponseType.CONCEDO) == "CONCEDO"
        assert str(ResponseType.NEGO) == "NEGO"
        assert str(ResponseType.DUBITO) == "DUBITO"


class TestPropositionStatus:
    """Tests for PropositionStatus enum."""

    def test_status_values(self):
        """Test that status types have correct values."""
        assert PropositionStatus.POSITUM.value == "positum"
        assert PropositionStatus.CONCESSA.value == "concessa"
        assert PropositionStatus.NEGATA.value == "negata"
        assert PropositionStatus.DUBITATA.value == "dubitata"
        assert PropositionStatus.IRRELEVANT.value == "irrelevant"


class TestProposition:
    """Tests for Proposition dataclass."""

    def test_proposition_creation(self):
        """Test creating a proposition."""
        prop = Proposition(
            content="Socrates is mortal",
            status=PropositionStatus.CONCESSA,
            turn_introduced=1,
        )
        assert prop.content == "Socrates is mortal"
        assert prop.status == PropositionStatus.CONCESSA
        assert prop.turn_introduced == 1
        assert prop.follows_from == []
        assert prop.incompatible_with == []
        assert prop.inference_chain is None

    def test_proposition_with_metadata(self):
        """Test proposition with inference metadata."""
        prop = Proposition(
            content="Socrates is mortal",
            status=PropositionStatus.CONCESSA,
            turn_introduced=2,
            follows_from=["All men are mortal", "Socrates is a man"],
            inference_chain="modus ponens",
        )
        assert len(prop.follows_from) == 2
        assert prop.inference_chain == "modus ponens"

    def test_proposition_equality(self):
        """Test that propositions are equal based on content."""
        prop1 = Proposition("Socrates is mortal", PropositionStatus.CONCESSA, 1)
        prop2 = Proposition("Socrates is mortal", PropositionStatus.NEGATA, 2)
        assert prop1 == prop2

    def test_proposition_hash(self):
        """Test that propositions can be used in sets."""
        prop1 = Proposition("Socrates is mortal", PropositionStatus.CONCESSA, 1)
        prop2 = Proposition("Socrates is mortal", PropositionStatus.CONCESSA, 2)
        prop_set = {prop1, prop2}
        assert len(prop_set) == 1  # Same content, only one in set

    def test_proposition_to_dict(self):
        """Test serialization to dictionary."""
        prop = Proposition(
            content="Socrates is mortal",
            status=PropositionStatus.CONCESSA,
            turn_introduced=1,
            follows_from=["All men are mortal"],
            inference_chain="syllogism",
        )
        data = prop.to_dict()
        assert data["content"] == "Socrates is mortal"
        assert data["status"] == "concessa"
        assert data["turn_introduced"] == 1
        assert data["follows_from"] == ["All men are mortal"]
        assert data["inference_chain"] == "syllogism"

    def test_proposition_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "content": "Socrates is mortal",
            "status": "concessa",
            "turn_introduced": 1,
            "follows_from": ["All men are mortal"],
            "inference_chain": "syllogism",
        }
        prop = Proposition.from_dict(data)
        assert prop.content == "Socrates is mortal"
        assert prop.status == PropositionStatus.CONCESSA
        assert prop.turn_introduced == 1
        assert prop.follows_from == ["All men are mortal"]
        assert prop.inference_chain == "syllogism"


class TestTurn:
    """Tests for Turn dataclass."""

    def test_turn_creation(self):
        """Test creating a turn."""
        turn = Turn(
            number=1,
            proposition="Socrates is mortal",
            response=ResponseType.CONCEDO,
            reasoning="Follows from positum",
            rule_applied=1,
            consistency_maintained=True,
        )
        assert turn.number == 1
        assert turn.proposition == "Socrates is mortal"
        assert turn.response == ResponseType.CONCEDO
        assert turn.reasoning == "Follows from positum"
        assert turn.rule_applied == 1
        assert turn.consistency_maintained is True

    def test_turn_to_dict(self):
        """Test turn serialization."""
        turn = Turn(
            number=1,
            proposition="Socrates is mortal",
            response=ResponseType.CONCEDO,
            reasoning="Follows from positum",
            rule_applied=1,
            consistency_maintained=True,
        )
        data = turn.to_dict()
        assert data["number"] == 1
        assert data["response"] == "concedo"
        assert data["rule_applied"] == 1

    def test_turn_from_dict(self):
        """Test turn deserialization."""
        data = {
            "number": 1,
            "proposition": "Socrates is mortal",
            "response": "concedo",
            "reasoning": "Follows from positum",
            "rule_applied": 1,
            "consistency_maintained": True,
        }
        turn = Turn.from_dict(data)
        assert turn.number == 1
        assert turn.response == ResponseType.CONCEDO


class TestObligationesState:
    """Tests for ObligationesState."""

    def test_state_initialization(self):
        """Test creating a new state."""
        state = ObligationesState()
        assert state.positum is None
        assert len(state.concessa) == 0
        assert len(state.negata) == 0
        assert len(state.dubitata) == 0
        assert state.turn_count == 0
        assert state.disputation_active is False
        assert len(state.history) == 0

    def test_state_with_common_knowledge(self):
        """Test state with common knowledge."""
        knowledge = {"All men are mortal", "Socrates is a man"}
        state = ObligationesState(common_knowledge=knowledge)
        assert len(state.common_knowledge) == 2
        assert "All men are mortal" in state.common_knowledge

    def test_set_positum(self):
        """Test setting the positum."""
        state = ObligationesState()
        state.set_positum("Socrates is immortal")
        assert state.positum is not None
        assert state.positum.content == "Socrates is immortal"
        assert state.positum.status == PropositionStatus.POSITUM
        assert state.positum.turn_introduced == 0
        assert state.disputation_active is True
        assert state.metadata["started_at"] is not None

    def test_add_concedo_response(self):
        """Test adding a CONCEDO response."""
        state = ObligationesState()
        state.set_positum("Socrates is immortal")
        state.add_response(
            proposition="Socrates is a god",
            response=ResponseType.CONCEDO,
            reasoning="Follows from positum",
            rule_applied=1,
        )
        assert state.turn_count == 1
        assert len(state.concessa) == 1
        assert len(state.history) == 1
        assert "Socrates is a god" in state.get_all_commitments()

    def test_add_nego_response(self):
        """Test adding a NEGO response."""
        state = ObligationesState()
        state.set_positum("Socrates is immortal")
        state.add_response(
            proposition="Socrates is mortal",
            response=ResponseType.NEGO,
            reasoning="Incompatible with positum",
            rule_applied=2,
        )
        assert len(state.negata) == 1
        assert "Socrates is mortal" in state.get_all_negations()

    def test_add_dubito_response(self):
        """Test adding a DUBITO response."""
        state = ObligationesState()
        state.set_positum("Socrates is immortal")
        state.add_response(
            proposition="Plato is wise",
            response=ResponseType.DUBITO,
            reasoning="Irrelevant to commitments",
            rule_applied=5,
        )
        assert len(state.dubitata) == 1
        assert "Plato is wise" in state.get_all_doubts()

    def test_get_all_commitments(self):
        """Test retrieving all commitments."""
        state = ObligationesState()
        state.set_positum("Socrates is immortal")
        state.add_response("Socrates is a god", ResponseType.CONCEDO, "reason", 1)
        state.add_response("Gods are immortal", ResponseType.CONCEDO, "reason", 1)

        commitments = state.get_all_commitments()
        assert len(commitments) == 3  # positum + 2 concessa
        assert "Socrates is immortal" in commitments
        assert "Socrates is a god" in commitments
        assert "Gods are immortal" in commitments

    def test_history_tracking(self):
        """Test that history is properly maintained."""
        state = ObligationesState()
        state.set_positum("Socrates is immortal")
        state.add_response("Socrates is a god", ResponseType.CONCEDO, "reason 1", 1)
        state.add_response("Socrates is mortal", ResponseType.NEGO, "reason 2", 2)

        assert len(state.history) == 2
        assert state.history[0].number == 0
        assert state.history[0].response == ResponseType.CONCEDO
        assert state.history[1].number == 1
        assert state.history[1].response == ResponseType.NEGO

    def test_end_disputation(self):
        """Test ending a disputation."""
        state = ObligationesState()
        state.set_positum("Socrates is immortal")
        state.disputation_active = True

        state.end_disputation("OPPONENT", "Respondent contradicted themselves")

        assert state.disputation_active is False
        assert state.metadata["winner"] == "OPPONENT"
        assert state.metadata["reason"] == "Respondent contradicted themselves"
        assert state.metadata["ended_at"] is not None

    def test_state_serialization(self):
        """Test state serialization to dict."""
        state = ObligationesState(common_knowledge={"All men are mortal"})
        state.set_positum("Socrates is immortal")
        state.add_response("Socrates is a god", ResponseType.CONCEDO, "reason", 1)

        data = state.to_dict()
        assert data["positum"]["content"] == "Socrates is immortal"
        assert len(data["concessa"]) == 1
        assert data["turn_count"] == 1
        assert len(data["common_knowledge"]) == 1

    def test_state_deserialization(self):
        """Test state deserialization from dict."""
        original = ObligationesState(common_knowledge={"All men are mortal"})
        original.set_positum("Socrates is immortal")
        original.add_response("Socrates is a god", ResponseType.CONCEDO, "reason", 1)

        data = original.to_dict()
        restored = ObligationesState.from_dict(data)

        assert restored.positum.content == original.positum.content
        assert len(restored.concessa) == len(original.concessa)
        assert restored.turn_count == original.turn_count
        assert len(restored.common_knowledge) == len(original.common_knowledge)

    def test_state_json_serialization(self):
        """Test JSON serialization."""
        state = ObligationesState()
        state.set_positum("Socrates is immortal")
        state.add_response("Socrates is a god", ResponseType.CONCEDO, "reason", 1)

        json_str = state.to_json()
        assert isinstance(json_str, str)

        data = json.loads(json_str)
        assert data["positum"]["content"] == "Socrates is immortal"

    def test_state_json_deserialization(self):
        """Test JSON deserialization."""
        original = ObligationesState()
        original.set_positum("Socrates is immortal")
        original.add_response("Socrates is a god", ResponseType.CONCEDO, "reason", 1)

        json_str = original.to_json()
        restored = ObligationesState.from_json(json_str)

        assert restored.positum.content == "Socrates is immortal"
        assert len(restored.concessa) == 1
        assert restored.turn_count == 1

    def test_state_repr(self):
        """Test string representation."""
        state = ObligationesState()
        state.set_positum("Socrates is immortal")
        repr_str = repr(state)
        assert "Socrates is immortal" in repr_str
        assert "turn=0" in repr_str
        assert "active=True" in repr_str
