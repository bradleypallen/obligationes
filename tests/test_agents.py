"""
Unit tests for agent implementations.

Tests validate the behavior of Respondent and Opponent agents.
"""

from unittest.mock import Mock
from dotenv import load_dotenv

from obligationes.agents import (
    RespondentAgent,
    OpponentAgent,
    OpponentStrategy,
    RespondentResponse,
    OpponentProposal,
)
from obligationes.state import ObligationesState, ResponseType

# Load environment variables
load_dotenv()


class TestRespondentAgent:
    """Tests for RespondentAgent."""

    def test_initialization_default(self):
        """Test creating Respondent with default engines."""
        agent = RespondentAgent()
        assert agent.rules_engine is not None
        assert agent.inference_engine is not None

    def test_initialization_with_custom_engines(self):
        """Test creating Respondent with custom engines."""
        mock_rules = Mock()
        mock_inference = Mock()
        agent = RespondentAgent(
            rules_engine=mock_rules, inference_engine=mock_inference
        )
        assert agent.rules_engine == mock_rules
        assert agent.inference_engine == mock_inference

    def test_evaluate_proposition_follows_rules(self):
        """Test that Respondent follows rules strictly."""
        # Create mock rules engine
        mock_rules = Mock()
        mock_rules.evaluate_proposition.return_value = (
            ResponseType.CONCEDO,
            "Rule 1 applies",
            1,
        )

        # Create mock inference for trap detection
        mock_inference = Mock()
        mock_inference.check_consistency.return_value = (True, [], "Consistent")

        agent = RespondentAgent(
            rules_engine=mock_rules, inference_engine=mock_inference
        )
        state = ObligationesState()
        state.set_positum("Socrates is mortal")

        result = agent.evaluate_proposition("Socrates is a man", state)

        assert result["response"] == ResponseType.CONCEDO
        assert result["rule_applied"] == 1
        assert "Rule 1" in result["reasoning"]
        assert isinstance(result["trap_detected"], bool)

    def test_trap_detection(self):
        """Test that Respondent can detect traps but must follow rules."""
        # Mock rules to require CONCEDO
        mock_rules = Mock()
        mock_rules.evaluate_proposition.return_value = (
            ResponseType.CONCEDO,
            "Rule 1 applies",
            1,
        )

        # Mock inference to show this creates inconsistency
        mock_inference = Mock()
        mock_inference.check_consistency.return_value = (
            False,
            ["Socrates is mortal vs Socrates is immortal"],
            "Contradiction detected",
        )

        agent = RespondentAgent(
            rules_engine=mock_rules, inference_engine=mock_inference
        )
        state = ObligationesState()
        state.set_positum("Socrates is mortal")

        result = agent.evaluate_proposition("Socrates is immortal", state)

        # Agent detected trap
        assert result["trap_detected"] is True
        assert "TRAP DETECTED" in result["trap_analysis"]
        # But still follows rules
        assert result["response"] == ResponseType.CONCEDO


class TestOpponentAgent:
    """Tests for OpponentAgent."""

    def test_initialization_default(self):
        """Test creating Opponent with default strategy."""
        agent = OpponentAgent()
        assert agent.strategy == OpponentStrategy.BALANCED
        assert agent.inference_engine is not None

    def test_initialization_with_strategy(self):
        """Test creating Opponent with specific strategy."""
        agent = OpponentAgent(strategy=OpponentStrategy.AGGRESSIVE)
        assert agent.strategy == OpponentStrategy.AGGRESSIVE

    def test_initialization_strategies(self):
        """Test all strategy options can be initialized."""
        for strategy in OpponentStrategy:
            agent = OpponentAgent(strategy=strategy)
            assert agent.strategy == strategy

    def test_propose_proposition_returns_dict(self):
        """Test that propose_proposition returns proper structure."""
        agent = OpponentAgent()
        state = ObligationesState()
        state.set_positum("Socrates is mortal")

        result = agent.propose_proposition(state)

        assert "proposition" in result
        assert "strategy_note" in result
        assert "expected_response" in result
        assert isinstance(result["proposition"], str)
        assert len(result["proposition"]) > 0


class TestPydanticModels:
    """Tests for Pydantic models used by agents."""

    def test_respondent_response_model(self):
        """Test RespondentResponse model."""
        response = RespondentResponse(
            response="CONCEDO",
            reasoning="Test reasoning",
            rule_applied=1,
            trap_detected=False,
        )
        assert response.response == "CONCEDO"
        assert response.rule_applied == 1
        assert response.trap_detected is False

    def test_opponent_proposal_model(self):
        """Test OpponentProposal model."""
        proposal = OpponentProposal(
            proposition="Socrates is wise",
            strategy_note="Building toward contradiction",
            expected_response="CONCEDO",
        )
        assert proposal.proposition == "Socrates is wise"
        assert "contradiction" in proposal.strategy_note.lower()



class TestAgentInteraction:
    """Tests for interaction between agents."""

    def test_respondent_evaluates_opponent_proposal(self):
        """Test that Respondent can evaluate Opponent's proposals."""
        # Create agents
        opponent = OpponentAgent()
        respondent = RespondentAgent()

        # Set up state
        state = ObligationesState(common_knowledge={"All men are mortal"})
        state.set_positum("Socrates is a man")

        # Opponent proposes
        proposal = opponent.propose_proposition(state)

        # Respondent evaluates
        evaluation = respondent.evaluate_proposition(proposal["proposition"], state)

        # Should get a valid response
        assert evaluation["response"] in [
            ResponseType.CONCEDO,
            ResponseType.NEGO,
            ResponseType.DUBITO,
        ]
        assert isinstance(evaluation["rule_applied"], int)
        assert 1 <= evaluation["rule_applied"] <= 5

    def test_full_interaction_cycle(self):
        """Test a complete interaction: Opponent → Respondent."""
        # Create agents
        opponent = OpponentAgent(strategy=OpponentStrategy.BALANCED)
        respondent = RespondentAgent()

        # Initialize state
        state = ObligationesState(common_knowledge={"All men are mortal"})
        state.set_positum("Socrates is a man")

        # Run a few turns
        for _ in range(2):
            # Opponent proposes
            proposal = opponent.propose_proposition(state)

            # Respondent evaluates
            evaluation = respondent.evaluate_proposition(proposal["proposition"], state)

            # Update state
            state.add_response(
                proposal["proposition"],
                evaluation["response"],
                evaluation["reasoning"],
                evaluation["rule_applied"],
            )

        # Verify state was updated
        assert state.turn_count == 2  # 2 turns (positum doesn't increment turn_count via set_positum)
        assert len(state.history) == 2
        assert state.get_all_commitments()  # Has commitments


class TestStrategyBehavior:
    """Tests for different Opponent strategies."""

    def test_balanced_strategy_initialization(self):
        """Test balanced strategy agent."""
        agent = OpponentAgent(strategy=OpponentStrategy.BALANCED)
        assert agent.strategy == OpponentStrategy.BALANCED

    def test_aggressive_strategy_initialization(self):
        """Test aggressive strategy agent."""
        agent = OpponentAgent(strategy=OpponentStrategy.AGGRESSIVE)
        assert agent.strategy == OpponentStrategy.AGGRESSIVE

    def test_pedagogical_strategy_initialization(self):
        """Test pedagogical strategy agent."""
        agent = OpponentAgent(strategy=OpponentStrategy.PEDAGOGICAL)
        assert agent.strategy == OpponentStrategy.PEDAGOGICAL
