"""
Unit tests for the LLM inference engine.

Note: These tests validate the Pydantic models and basic structure.
Tests require API keys to be set in .env file.
"""

import pytest
import os
from dotenv import load_dotenv
from obligationes.inference import (
    LLMInferenceEngine,
    FollowsFromResult,
    IncompatibleWithResult,
    ConsistencyResult,
)

# Load environment variables from .env
load_dotenv()


class TestLLMInferenceEngine:
    """Tests for LLMInferenceEngine initialization."""

    def test_initialization_default(self):
        """Test creating engine with default settings."""
        engine = LLMInferenceEngine()
        assert engine.temperature == 0.0
        assert engine.llm is not None

    def test_initialization_with_custom_model(self):
        """Test creating engine with custom model."""
        engine = LLMInferenceEngine(model_name="gpt-4-turbo", temperature=0.2)
        assert engine.temperature == 0.2

    def test_initialization_with_claude(self):
        """Test creating engine with Claude model."""
        # Skip if no Anthropic API key
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        engine = LLMInferenceEngine(model_name="claude-3-opus-20240229")
        assert engine.llm is not None

    def test_parsers_created(self):
        """Test that output parsers are created."""
        engine = LLMInferenceEngine()
        assert engine.follows_parser is not None
        assert engine.incompatible_parser is not None
        assert engine.consistency_parser is not None

    def test_chains_created(self):
        """Test that chains are created."""
        engine = LLMInferenceEngine()
        assert engine._follows_chain is not None
        assert engine._incompatible_chain is not None
        assert engine._consistency_chain is not None


class TestEmptyInputHandling:
    """Tests for handling empty or minimal inputs."""

    def test_follows_from_with_empty_premises(self):
        """Test follows_from with no premises."""
        engine = LLMInferenceEngine()
        follows, reasoning, confidence = engine.follows_from("P", set())

        assert follows is False
        assert "no premises" in reasoning.lower()
        assert confidence == 0.0

    def test_incompatible_with_empty_premises(self):
        """Test incompatible_with with no premises."""
        engine = LLMInferenceEngine()
        incompatible, reasoning, confidence = engine.incompatible_with("P", set())

        assert incompatible is False
        assert "no premises" in reasoning.lower()

    def test_check_consistency_with_single_proposition(self):
        """Test consistency with only one proposition (trivially consistent)."""
        engine = LLMInferenceEngine()
        consistent, contradictions, reasoning = engine.check_consistency({"P"})

        assert consistent is True
        assert len(contradictions) == 0
        assert "trivially consistent" in reasoning.lower()

    def test_check_consistency_with_empty_set(self):
        """Test consistency with empty set (trivially consistent)."""
        engine = LLMInferenceEngine()
        consistent, contradictions, reasoning = engine.check_consistency(set())

        assert consistent is True
        assert len(contradictions) == 0


class TestFallbackParsing:
    """Tests for fallback parsing strategies."""

    def test_fallback_parse_follows_with_markdown_json(self):
        """Test parsing JSON from markdown code blocks."""
        engine = LLMInferenceEngine()

        content = """```json
{
    "follows": true,
    "inference_type": "modus_ponens",
    "reasoning": "Valid inference",
    "confidence": 0.9
}
```"""
        result = engine._parse_follows_result(content)
        assert result.follows is True
        assert result.confidence == 0.9

    def test_fallback_parse_follows_with_keywords(self):
        """Test keyword-based fallback parsing."""
        engine = LLMInferenceEngine()

        content = "Yes, this follows by modus ponens with high confidence."
        result = engine._parse_follows_result(content)
        assert result.follows is True
        assert "modus_ponens" in result.inference_type  # Note: uses underscore in enum

    def test_fallback_parse_incompatible_with_keywords(self):
        """Test keyword-based parsing for incompatibility."""
        engine = LLMInferenceEngine()

        content = "These are incompatible due to direct contradiction."
        result = engine._parse_incompatible_result(content)
        assert result.incompatible is True
        assert "contradiction" in result.reasoning.lower()

    def test_fallback_parse_consistency_with_keywords(self):
        """Test keyword-based parsing for consistency."""
        engine = LLMInferenceEngine()

        content = "The set is consistent and has no contradictions."
        result = engine._parse_consistency_result(content)
        assert result.consistent is True


class TestPydanticModels:
    """Tests for Pydantic model validation."""

    def test_follows_from_result_validation(self):
        """Test FollowsFromResult model."""
        result = FollowsFromResult(
            follows=True,
            inference_type="modus_ponens",
            reasoning="Valid inference",
            confidence=0.95,
        )
        assert result.follows is True
        assert result.confidence == 0.95

    def test_follows_from_result_confidence_bounds(self):
        """Test confidence must be between 0 and 1."""
        with pytest.raises(Exception):  # Pydantic validation error
            FollowsFromResult(
                follows=True,
                inference_type="modus_ponens",
                reasoning="Test",
                confidence=1.5,  # Invalid
            )

    def test_incompatible_with_result_validation(self):
        """Test IncompatibleWithResult model."""
        result = IncompatibleWithResult(
            incompatible=True,
            contradiction_type="direct",
            reasoning="Contradiction detected",
            confidence=0.99,
        )
        assert result.incompatible is True
        assert result.contradiction_type == "direct"

    def test_consistency_result_validation(self):
        """Test ConsistencyResult model."""
        result = ConsistencyResult(
            consistent=False,
            contradictions=["P vs not P"],
            reasoning="Direct contradiction",
        )
        assert result.consistent is False
        assert len(result.contradictions) == 1
