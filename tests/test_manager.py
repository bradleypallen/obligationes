"""
Unit tests for the Disputation Manager.

Tests validate the orchestration of complete disputations.
"""

import os
import tempfile
from dotenv import load_dotenv

from obligationes.manager import (
    DisputationManager,
    DisputationConfig,
    DisputationResult,
    create_disputation,
    DEFAULT_COMMON_KNOWLEDGE,
)
from obligationes.agents import OpponentStrategy

# Load environment variables
load_dotenv()


class TestDisputationConfig:
    """Tests for DisputationConfig."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = DisputationConfig()
        assert config.max_turns == 10
        assert config.opponent_strategy == OpponentStrategy.BALANCED
        assert config.verbose is True
        assert config.model_name == "gpt-4o-mini"  # Overridden by conftest.py fixture
        assert config.temperature == 0.0

    def test_config_custom_values(self):
        """Test custom configuration."""
        config = DisputationConfig(
            max_turns=5,
            opponent_strategy=OpponentStrategy.AGGRESSIVE,
            verbose=False,
            model_name="gpt-4-turbo",
        )
        assert config.max_turns == 5
        assert config.opponent_strategy == OpponentStrategy.AGGRESSIVE
        assert config.verbose is False
        assert config.model_name == "gpt-4-turbo"


class TestDisputationManager:
    """Tests for DisputationManager."""

    def test_initialization_default(self):
        """Test creating manager with defaults."""
        manager = DisputationManager()
        assert manager.state is not None
        assert manager.respondent is not None
        assert manager.opponent is not None
        assert len(manager.state.common_knowledge) > 0

    def test_initialization_custom_common_knowledge(self):
        """Test creating manager with custom common knowledge."""
        ck = {"Socrates is wise", "Plato is a philosopher"}
        manager = DisputationManager(common_knowledge=ck)
        assert manager.state.common_knowledge == ck

    def test_initialization_with_config(self):
        """Test creating manager with custom config."""
        config = DisputationConfig(
            max_turns=5, opponent_strategy=OpponentStrategy.PEDAGOGICAL, verbose=False
        )
        manager = DisputationManager(config=config)
        assert manager.config.max_turns == 5
        assert manager.config.opponent_strategy == OpponentStrategy.PEDAGOGICAL

    def test_default_common_knowledge(self):
        """Test that default common knowledge is reasonable."""
        assert len(DEFAULT_COMMON_KNOWLEDGE) > 0
        assert "All men are mortal" in DEFAULT_COMMON_KNOWLEDGE
        assert "Contradictions cannot be true" in DEFAULT_COMMON_KNOWLEDGE


class TestDisputationExecution:
    """Tests for running disputations."""

    def test_run_disputation_returns_result(self):
        """Test that run_disputation returns a valid result."""
        config = DisputationConfig(max_turns=2, verbose=False)
        manager = DisputationManager(config=config)

        result = manager.run_disputation("Socrates is immortal")

        assert isinstance(result, DisputationResult)
        assert result.winner in ["OPPONENT", "RESPONDENT"]
        assert result.positum == "Socrates is immortal"
        assert result.total_turns <= 3
        assert len(result.transcript) == result.total_turns
        assert isinstance(result.final_consistent, bool)

    def test_run_disputation_with_simple_positum(self):
        """Test disputation with a simple positum."""
        config = DisputationConfig(max_turns=2, verbose=False)
        manager = DisputationManager(config=config)

        result = manager.run_disputation("Socrates is a man")

        assert result.positum == "Socrates is a man"
        assert result.total_turns >= 1

    def test_step_execution(self):
        """Test single-step execution."""
        manager = DisputationManager()
        manager.state.set_positum("Socrates is mortal")

        step_result = manager.step()

        assert "proposition" in step_result
        assert "response" in step_result
        assert "reasoning" in step_result
        assert "rule_applied" in step_result
        assert "consistent" in step_result
        assert step_result["turn"] == 0

    def test_step_with_custom_proposition(self):
        """Test step with user-provided proposition."""
        manager = DisputationManager()
        manager.state.set_positum("Socrates is mortal")

        step_result = manager.step("All men are mortal")

        assert step_result["proposition"] == "All men are mortal"
        assert "response" in step_result

    def test_multiple_steps(self):
        """Test executing multiple steps."""
        config = DisputationConfig(verbose=False)
        manager = DisputationManager(config=config)
        manager.state.set_positum("Socrates is mortal")

        # Execute 3 steps
        for i in range(2):
            step_result = manager.step()
            assert step_result["turn"] == i

        assert manager.state.turn_count == 2


class TestDisputationStatus:
    """Tests for status tracking."""

    def test_get_status_initial(self):
        """Test status at initialization."""
        manager = DisputationManager()
        manager.state.set_positum("Socrates is mortal")

        status = manager.get_status()

        assert status["active"] is True
        assert status["positum"] == "Socrates is mortal"
        assert status["turn_count"] == 0
        assert isinstance(status["commitments"], list)
        assert "Socrates is mortal" in status["commitments"]

    def test_get_status_after_turns(self):
        """Test status after several turns."""
        config = DisputationConfig(verbose=False)
        manager = DisputationManager(config=config)
        manager.state.set_positum("Socrates is mortal")

        # Execute some turns
        manager.step()
        manager.step()

        status = manager.get_status()

        assert status["turn_count"] == 2
        assert len(status["commitments"]) >= 1


class TestTranscriptManagement:
    """Tests for saving and loading transcripts."""

    def test_save_transcript(self):
        """Test saving transcript to file."""
        config = DisputationConfig(max_turns=2, verbose=False)
        manager = DisputationManager(config=config)
        manager.run_disputation("Socrates is mortal")

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            manager.save_transcript(temp_path)
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_from_transcript(self):
        """Test loading disputation from transcript."""
        config = DisputationConfig(max_turns=2, verbose=False)
        manager = DisputationManager(config=config)
        result = manager.run_disputation("Socrates is mortal")

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            manager.save_transcript(temp_path)

            # Load it back
            loaded_manager = DisputationManager.from_transcript(temp_path)

            assert loaded_manager.state.positum.content == result.positum
            assert loaded_manager.state.turn_count == result.total_turns
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestConvenienceFunction:
    """Tests for the create_disputation convenience function."""

    def test_create_disputation_basic(self):
        """Test basic usage of create_disputation."""
        result = create_disputation(
            positum="Socrates is wise", max_turns=2, verbose=False
        )

        assert isinstance(result, DisputationResult)
        assert result.positum == "Socrates is wise"
        assert result.total_turns <= 3

    def test_create_disputation_with_strategy(self):
        """Test create_disputation with specific strategy."""
        result = create_disputation(
            positum="Socrates is mortal",
            max_turns=2,
            strategy=OpponentStrategy.AGGRESSIVE,
            verbose=False,
        )

        assert result.positum == "Socrates is mortal"

    def test_create_disputation_with_custom_ck(self):
        """Test create_disputation with custom common knowledge."""
        ck = {"Test fact 1", "Test fact 2"}
        result = create_disputation(
            positum="Some proposition", common_knowledge=ck, max_turns=1, verbose=False
        )

        # The result should complete
        assert result.total_turns >= 1


class TestDisputationResult:
    """Tests for DisputationResult."""

    def test_result_structure(self):
        """Test that result has all expected fields."""
        config = DisputationConfig(max_turns=2, verbose=False)
        manager = DisputationManager(config=config)
        result = manager.run_disputation("Socrates is mortal")

        # Check all fields exist
        assert hasattr(result, "winner")
        assert hasattr(result, "reason")
        assert hasattr(result, "positum")
        assert hasattr(result, "total_turns")
        assert hasattr(result, "final_consistent")
        assert hasattr(result, "transcript")
        assert hasattr(result, "judgment")
        assert hasattr(result, "state")
        assert hasattr(result, "started_at")
        assert hasattr(result, "ended_at")
        assert hasattr(result, "duration_seconds")

    def test_result_transcript_format(self):
        """Test that transcript is properly formatted."""
        config = DisputationConfig(max_turns=2, verbose=False)
        manager = DisputationManager(config=config)
        result = manager.run_disputation("Socrates is mortal")

        assert len(result.transcript) == result.total_turns

        for turn_data in result.transcript:
            assert "turn" in turn_data
            assert "proposition" in turn_data
            assert "response" in turn_data
            assert "reasoning" in turn_data
            assert "rule_applied" in turn_data


class TestStrategyIntegration:
    """Tests for different opponent strategies."""

    def test_balanced_strategy(self):
        """Test disputation with balanced strategy."""
        config = DisputationConfig(
            max_turns=2, opponent_strategy=OpponentStrategy.BALANCED, verbose=False
        )
        manager = DisputationManager(config=config)
        result = manager.run_disputation("Socrates is mortal")

        assert result.total_turns <= 3

    def test_aggressive_strategy(self):
        """Test disputation with aggressive strategy."""
        config = DisputationConfig(
            max_turns=2, opponent_strategy=OpponentStrategy.AGGRESSIVE, verbose=False
        )
        manager = DisputationManager(config=config)
        result = manager.run_disputation("Socrates is mortal")

        assert result.total_turns <= 3

    def test_pedagogical_strategy(self):
        """Test disputation with pedagogical strategy."""
        config = DisputationConfig(
            max_turns=2, opponent_strategy=OpponentStrategy.PEDAGOGICAL, verbose=False
        )
        manager = DisputationManager(config=config)
        result = manager.run_disputation("Socrates is mortal")

        assert result.total_turns <= 3
