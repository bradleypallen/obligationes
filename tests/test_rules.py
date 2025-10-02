"""
Unit tests for Burley's Rules Engine.

These tests validate that the rules are applied in correct precedence order
and that each rule behaves as specified.
"""

from unittest.mock import Mock
from dotenv import load_dotenv
from obligationes.rules import BurleyRulesEngine
from obligationes.state import ObligationesState, ResponseType

# Load environment variables
load_dotenv()


class TestBurleyRulesEngine:
    """Tests for BurleyRulesEngine initialization and basic functionality."""

    def test_initialization_default(self):
        """Test creating rules engine with default inference engine."""
        engine = BurleyRulesEngine()
        assert engine.inference_engine is not None

    def test_initialization_with_custom_engine(self):
        """Test creating rules engine with custom inference engine."""
        mock_inference = Mock()
        engine = BurleyRulesEngine(inference_engine=mock_inference)
        assert engine.inference_engine == mock_inference

    def test_explain_rules(self):
        """Test that rules explanation is provided."""
        engine = BurleyRulesEngine()
        explanation = engine.explain_rules()
        assert "Burley" in explanation
        assert "positum" in explanation.lower()
        assert "concessa" in explanation.lower()
        assert "1." in explanation and "5." in explanation  # All 5 rules present


class TestRulePrecedence:
    """Tests for rule precedence order."""

    def test_rule_1_before_rule_5(self):
        """Test that Rule 1 takes precedence over Rule 5."""
        # Create mock inference engine
        mock_inference = Mock()

        # Rule 1 should fire (follows from commitments)
        mock_inference.follows_from.return_value = (
            True,
            "Follows by modus ponens",
            0.95,
        )

        engine = BurleyRulesEngine(inference_engine=mock_inference)
        state = ObligationesState(common_knowledge={"All men are mortal"})
        state.set_positum("Socrates is a man")

        response, reasoning, rule = engine.evaluate_proposition(
            "Socrates is mortal", state
        )

        assert rule == 1
        assert response == ResponseType.CONCEDO
        assert "RULE 1" in reasoning

    def test_rule_2_before_rule_3(self):
        """Test that Rule 2 takes precedence over Rule 3."""
        # Create mock inference engine
        mock_inference = Mock()

        # Rule 1 doesn't apply (doesn't follow)
        # Rule 2 applies (incompatible)
        def follows_side_effect(prop, premises):
            return (False, "Doesn't follow", 0.1)

        def incompatible_side_effect(prop, premises):
            return (True, "Direct contradiction", 0.95)

        mock_inference.follows_from.side_effect = follows_side_effect
        mock_inference.incompatible_with.side_effect = incompatible_side_effect

        engine = BurleyRulesEngine(inference_engine=mock_inference)
        state = ObligationesState()
        state.set_positum("Socrates is mortal")

        response, reasoning, rule = engine.evaluate_proposition(
            "Socrates is immortal", state
        )

        assert rule == 2
        assert response == ResponseType.NEGO
        assert "RULE 2" in reasoning


class TestRule1:
    """Tests for Rule 1: Follows from commitments."""

    def test_rule_1_applies_with_high_confidence(self):
        """Test Rule 1 fires when proposition follows from commitments."""
        mock_inference = Mock()
        mock_inference.follows_from.return_value = (True, "Valid syllogism", 0.95)

        engine = BurleyRulesEngine(inference_engine=mock_inference)
        state = ObligationesState()
        state.set_positum("All men are mortal")
        state.add_response("Socrates is a man", ResponseType.CONCEDO, "test", 5)

        response, reasoning, rule = engine.evaluate_proposition(
            "Socrates is mortal", state
        )

        assert rule == 1
        assert response == ResponseType.CONCEDO
        assert "RULE 1" in reasoning

        # Verify inference engine was called with commitments only
        call_args = mock_inference.follows_from.call_args
        assert "Socrates is mortal" == call_args[0][0]
        premises = call_args[0][1]
        assert "All men are mortal" in premises
        assert "Socrates is a man" in premises

    def test_rule_1_not_applies_with_low_confidence(self):
        """Test Rule 1 doesn't fire with low confidence."""
        mock_inference = Mock()
        # Low confidence - should not trigger rule
        mock_inference.follows_from.return_value = (True, "Maybe follows", 0.5)
        mock_inference.incompatible_with.return_value = (False, "Not incompatible", 0.1)
        # Truth value for Rule 5
        mock_inference.evaluate_truth_value.return_value = ("unknown", 0.5, "Cannot determine")

        engine = BurleyRulesEngine(inference_engine=mock_inference)
        state = ObligationesState(common_knowledge={"Background fact"})
        state.set_positum("Positum")

        response, reasoning, rule = engine.evaluate_proposition(
            "Some proposition", state
        )

        # Should fall through to later rules
        assert rule != 1


class TestRule2:
    """Tests for Rule 2: Incompatible with commitments."""

    def test_rule_2_applies_with_contradiction(self):
        """Test Rule 2 fires when proposition contradicts commitments."""
        mock_inference = Mock()
        mock_inference.follows_from.return_value = (False, "Doesn't follow", 0.1)
        mock_inference.incompatible_with.return_value = (
            True,
            "Direct contradiction",
            0.95,
        )

        engine = BurleyRulesEngine(inference_engine=mock_inference)
        state = ObligationesState()
        state.set_positum("Socrates is mortal")

        response, reasoning, rule = engine.evaluate_proposition(
            "Socrates is immortal", state
        )

        assert rule == 2
        assert response == ResponseType.NEGO
        assert "RULE 2" in reasoning


class TestRule3:
    """Tests for Rule 3: Follows from commitments + common knowledge."""

    def test_rule_3_applies_with_common_knowledge(self):
        """Test Rule 3 fires when proposition follows from commitments + CK."""
        mock_inference = Mock()

        # Track calls to follows_from
        call_count = [0]

        def follows_side_effect(prop, premises):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call (Rule 1) - doesn't follow from commitments alone
                return (False, "Doesn't follow", 0.1)
            else:
                # Second call (Rule 3) - follows from commitments + CK
                return (True, "Follows with CK", 0.95)

        mock_inference.follows_from.side_effect = follows_side_effect
        mock_inference.incompatible_with.return_value = (False, "Not incompatible", 0.1)

        engine = BurleyRulesEngine(inference_engine=mock_inference)
        state = ObligationesState(common_knowledge={"All men are mortal"})
        state.set_positum("Socrates is a man")

        response, reasoning, rule = engine.evaluate_proposition(
            "Socrates is mortal", state
        )

        assert rule == 3
        assert response == ResponseType.CONCEDO
        assert "RULE 3" in reasoning


class TestRule4:
    """Tests for Rule 4: Incompatible with commitments + common knowledge."""

    def test_rule_4_applies_with_common_knowledge(self):
        """Test Rule 4 fires when proposition contradicts commitments + CK."""
        mock_inference = Mock()

        # Track calls
        follows_calls = [0]
        incompat_calls = [0]

        def follows_side_effect(prop, premises):
            follows_calls[0] += 1
            return (False, "Doesn't follow", 0.1)

        def incompatible_side_effect(prop, premises):
            incompat_calls[0] += 1
            if incompat_calls[0] == 1:
                # First call (Rule 2) - not incompatible with commitments alone
                return (False, "Not incompatible", 0.1)
            else:
                # Second call (Rule 4) - incompatible with commitments + CK
                return (True, "Incompatible with CK", 0.95)

        mock_inference.follows_from.side_effect = follows_side_effect
        mock_inference.incompatible_with.side_effect = incompatible_side_effect

        engine = BurleyRulesEngine(inference_engine=mock_inference)
        state = ObligationesState(common_knowledge={"The Earth is round"})
        state.set_positum("Socrates is a man")

        response, reasoning, rule = engine.evaluate_proposition(
            "The Earth is flat", state
        )

        assert rule == 4
        assert response == ResponseType.NEGO
        assert "RULE 4" in reasoning


class TestRule5:
    """Tests for Rule 5: Based on common knowledge alone."""

    def test_rule_5_concedo_when_true_in_ck(self):
        """Test Rule 5 grants propositions true in common knowledge."""
        mock_inference = Mock()

        # None of the first 4 rules apply
        mock_inference.follows_from.side_effect = [
            (False, "Doesn't follow", 0.1),  # Rule 1
            (False, "Doesn't follow", 0.1),  # Rule 3
            (True, "True in CK", 0.95),  # Rule 5 - follows from CK
        ]
        mock_inference.incompatible_with.return_value = (False, "Not incompatible", 0.1)

        engine = BurleyRulesEngine(inference_engine=mock_inference)
        state = ObligationesState(common_knowledge={"All men are mortal"})
        state.set_positum("Socrates is a man")

        response, reasoning, rule = engine.evaluate_proposition(
            "All men are mortal", state  # This is in CK
        )

        assert rule == 5
        assert response == ResponseType.CONCEDO
        assert "RULE 5" in reasoning

    def test_rule_5_nego_when_false_in_ck(self):
        """Test Rule 5 denies propositions false in common knowledge."""
        mock_inference = Mock()

        # None of the first 4 rules apply
        mock_inference.follows_from.return_value = (False, "Doesn't follow", 0.1)
        mock_inference.incompatible_with.side_effect = [
            (False, "Not incompatible", 0.1),  # Rule 2
            (False, "Not incompatible", 0.1),  # Rule 4
            (True, "False in CK", 0.95),  # Rule 5 - incompatible with CK
        ]

        engine = BurleyRulesEngine(inference_engine=mock_inference)
        state = ObligationesState(common_knowledge={"The Earth is round"})
        state.set_positum("Socrates is a man")

        response, reasoning, rule = engine.evaluate_proposition(
            "The Earth is flat", state
        )

        assert rule == 5
        assert response == ResponseType.NEGO
        assert "RULE 5" in reasoning

    def test_rule_5_dubito_when_irrelevant(self):
        """Test Rule 5 doubts irrelevant propositions."""
        mock_inference = Mock()

        # Nothing follows, nothing incompatible
        mock_inference.follows_from.return_value = (False, "Doesn't follow", 0.1)
        mock_inference.incompatible_with.return_value = (False, "Not incompatible", 0.1)
        # Truth value unknown for irrelevant proposition
        mock_inference.evaluate_truth_value.return_value = ("unknown", 0.5, "Cannot determine truth value")

        engine = BurleyRulesEngine(inference_engine=mock_inference)
        state = ObligationesState(common_knowledge={"All men are mortal"})
        state.set_positum("Socrates is a man")

        response, reasoning, rule = engine.evaluate_proposition(
            "Plato is wise", state  # Irrelevant to both commitments and CK
        )

        assert rule == 5
        assert response == ResponseType.DUBITO
        assert "RULE 5" in reasoning

    def test_rule_5_dubito_with_no_common_knowledge(self):
        """Test Rule 5 doubts when there's no common knowledge."""
        mock_inference = Mock()
        # Mock all the calls that will happen in rules 1-4
        mock_inference.follows_from.return_value = (False, "Doesn't follow", 0.1)
        mock_inference.incompatible_with.return_value = (False, "Not incompatible", 0.1)
        # Truth value unknown when no common knowledge
        mock_inference.evaluate_truth_value.return_value = ("unknown", 0.5, "Cannot determine truth value")

        engine = BurleyRulesEngine(inference_engine=mock_inference)
        state = ObligationesState()  # No common knowledge
        state.set_positum("Socrates is a man")

        response, reasoning, rule = engine.evaluate_proposition(
            "Some proposition", state
        )

        assert rule == 5
        assert response == ResponseType.DUBITO
        assert "RULE 5" in reasoning


class TestEmptyState:
    """Tests for edge cases with empty or minimal state."""

    def test_evaluate_with_no_commitments(self):
        """Test evaluation with only common knowledge."""
        mock_inference = Mock()
        # When there are no commitments, Rule 3 will fire with CK alone
        # Rule 1 check will be skipped (no commitments)
        # Rule 3 will find it follows from CK
        mock_inference.follows_from.return_value = (True, "True in CK", 0.95)
        mock_inference.incompatible_with.return_value = (False, "Not incompatible", 0.1)

        engine = BurleyRulesEngine(inference_engine=mock_inference)
        state = ObligationesState(common_knowledge={"All men are mortal"})
        # No positum set - empty commitments

        response, reasoning, rule = engine.evaluate_proposition(
            "All men are mortal", state
        )

        # With empty commitments but CK, Rule 3 will fire
        assert rule == 3  # Not Rule 5, because CK causes Rule 3 to apply
        assert response == ResponseType.CONCEDO

    def test_evaluate_with_no_common_knowledge_no_commitments(self):
        """Test evaluation with completely empty state."""
        mock_inference = Mock()
        # Truth value for Rule 5
        mock_inference.evaluate_truth_value.return_value = ("unknown", 0.5, "Cannot determine")

        engine = BurleyRulesEngine(inference_engine=mock_inference)
        state = ObligationesState()  # Completely empty

        response, reasoning, rule = engine.evaluate_proposition(
            "Some proposition", state
        )

        # Should go to Rule 5 and return DUBITO
        assert rule == 5
        assert response == ResponseType.DUBITO
