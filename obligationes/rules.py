"""
Burley's Rules Engine for obligationes.

This module implements Walter Burley's five rules for determining how a Respondent
must respond to propositions during an obligationes disputation.

The rules MUST be applied in strict order of precedence.
"""

from typing import Set, Tuple, Optional
from obligationes.state import ResponseType, ObligationesState
from obligationes.inference import LLMInferenceEngine


class BurleyRulesEngine:
    """
    Implementation of Walter Burley's evaluation rules for obligationes.

    Rules are applied in strict order of precedence:
    1. If proposition follows from (positum + concessa) → CONCEDO
    2. If proposition incompatible with (positum + concessa) → NEGO
    3. If proposition follows from (positum + concessa + common_knowledge) → CONCEDO
    4. If proposition incompatible with (positum + concessa + common_knowledge) → NEGO
    5. Otherwise, respond based on truth in common knowledge alone

    Attributes:
        inference_engine: LLM-based logical reasoning engine
    """

    def __init__(self, inference_engine: Optional[LLMInferenceEngine] = None):
        """
        Initialize the rules engine.

        Args:
            inference_engine: LLM inference engine (creates default if None)
        """
        self.inference_engine = inference_engine or LLMInferenceEngine()

    def evaluate_proposition(
        self, proposition: str, state: ObligationesState
    ) -> Tuple[ResponseType, str, int]:
        """
        Evaluate a proposition according to Burley's rules.

        Rules are checked in strict order. The first rule that applies determines
        the response. This implements the formal precedence structure of medieval
        obligationes.

        Args:
            proposition: The proposition to evaluate
            state: Current disputation state

        Returns:
            Tuple of (response_type, reasoning, rule_number)
            - response_type: CONCEDO, NEGO, or DUBITO
            - reasoning: Explanation of why this response is required
            - rule_number: Which rule was applied (1-5)
        """
        # Get current commitments (positum + concessa)
        commitments = state.get_all_commitments()
        common_knowledge = state.common_knowledge

        # Rule 1: Follows from commitments alone
        result = self._check_rule_1(proposition, commitments)
        if result is not None:
            return result

        # Rule 2: Incompatible with commitments alone
        result = self._check_rule_2(proposition, commitments)
        if result is not None:
            return result

        # Rule 3: Follows from commitments + common knowledge
        result = self._check_rule_3(proposition, commitments, common_knowledge)
        if result is not None:
            return result

        # Rule 4: Incompatible with commitments + common knowledge
        result = self._check_rule_4(proposition, commitments, common_knowledge)
        if result is not None:
            return result

        # Rule 5: Based on truth in common knowledge alone
        return self._check_rule_5(proposition, common_knowledge)

    def _check_rule_1(
        self, proposition: str, commitments: Set[str]
    ) -> Optional[Tuple[ResponseType, str, int]]:
        """
        Rule 1: If proposition follows from (positum + concessa) → CONCEDO

        This is the strongest rule. If the proposition is a logical consequence
        of what has already been granted, it MUST be granted.

        Args:
            proposition: Proposition to check
            commitments: Current commitments (positum + concessa)

        Returns:
            (CONCEDO, reasoning, 1) if rule applies, None otherwise
        """
        if not commitments:
            return None

        follows, reasoning, confidence = self.inference_engine.follows_from(
            proposition, commitments
        )

        if follows and confidence > 0.7:  # High confidence threshold
            explanation = (
                f"RULE 1: Proposition necessarily follows from commitments. "
                f"{reasoning}"
            )
            return (ResponseType.CONCEDO, explanation, 1)

        return None

    def _check_rule_2(
        self, proposition: str, commitments: Set[str]
    ) -> Optional[Tuple[ResponseType, str, int]]:
        """
        Rule 2: If proposition incompatible with (positum + concessa) → NEGO

        If granting the proposition would create a contradiction with existing
        commitments, it MUST be denied.

        Args:
            proposition: Proposition to check
            commitments: Current commitments (positum + concessa)

        Returns:
            (NEGO, reasoning, 2) if rule applies, None otherwise
        """
        if not commitments:
            return None

        incompatible, reasoning, confidence = self.inference_engine.incompatible_with(
            proposition, commitments
        )

        if incompatible and confidence > 0.7:  # High confidence threshold
            explanation = (
                f"RULE 2: Proposition incompatible with commitments. " f"{reasoning}"
            )
            return (ResponseType.NEGO, explanation, 2)

        return None

    def _check_rule_3(
        self, proposition: str, commitments: Set[str], common_knowledge: Set[str]
    ) -> Optional[Tuple[ResponseType, str, int]]:
        """
        Rule 3: If proposition follows from (positum + concessa + common_knowledge) → CONCEDO

        If the proposition follows from commitments combined with background knowledge,
        it must be granted.

        Args:
            proposition: Proposition to check
            commitments: Current commitments (positum + concessa)
            common_knowledge: Background facts

        Returns:
            (CONCEDO, reasoning, 3) if rule applies, None otherwise
        """
        # Combine commitments and common knowledge
        combined = commitments.union(common_knowledge)

        if not combined or combined == commitments:
            # No common knowledge to add, or already checked in Rule 1
            return None

        follows, reasoning, confidence = self.inference_engine.follows_from(
            proposition, combined
        )

        if follows and confidence > 0.7:
            explanation = (
                f"RULE 3: Proposition follows from commitments + common knowledge. "
                f"{reasoning}"
            )
            return (ResponseType.CONCEDO, explanation, 3)

        return None

    def _check_rule_4(
        self, proposition: str, commitments: Set[str], common_knowledge: Set[str]
    ) -> Optional[Tuple[ResponseType, str, int]]:
        """
        Rule 4: If proposition incompatible with (positum + concessa + common_knowledge) → NEGO

        If the proposition contradicts commitments when combined with background
        knowledge, it must be denied.

        Args:
            proposition: Proposition to check
            commitments: Current commitments (positum + concessa)
            common_knowledge: Background facts

        Returns:
            (NEGO, reasoning, 4) if rule applies, None otherwise
        """
        # Combine commitments and common knowledge
        combined = commitments.union(common_knowledge)

        if not combined or combined == commitments:
            # No common knowledge to add, or already checked in Rule 2
            return None

        incompatible, reasoning, confidence = self.inference_engine.incompatible_with(
            proposition, combined
        )

        if incompatible and confidence > 0.7:
            explanation = (
                f"RULE 4: Proposition incompatible with commitments + common knowledge. "
                f"{reasoning}"
            )
            return (ResponseType.NEGO, explanation, 4)

        return None

    def _check_rule_5(
        self, proposition: str, common_knowledge: Set[str]
    ) -> Tuple[ResponseType, str, int]:
        """
        Rule 5: Respond based on truth in common knowledge alone

        This is the default fallback rule. If none of the previous rules apply,
        the Respondent responds based on whether the proposition is true or false
        in common knowledge.

        If the proposition follows from common knowledge → CONCEDO
        If the proposition is incompatible with common knowledge → NEGO
        Otherwise → DUBITO (doubt/irrelevant)

        Args:
            proposition: Proposition to check
            common_knowledge: Background facts

        Returns:
            (ResponseType, reasoning, 5) - always returns a response
        """
        if not common_knowledge:
            # No common knowledge, so proposition is irrelevant
            explanation = (
                "RULE 5: No common knowledge available. "
                "Proposition is irrelevant to the disputation."
            )
            return (ResponseType.DUBITO, explanation, 5)

        # Check if it follows from common knowledge
        follows, follows_reasoning, follows_confidence = (
            self.inference_engine.follows_from(proposition, common_knowledge)
        )

        if follows and follows_confidence > 0.7:
            explanation = (
                f"RULE 5: Proposition follows from common knowledge (granted as true). "
                f"{follows_reasoning}"
            )
            return (ResponseType.CONCEDO, explanation, 5)

        # Check if it's incompatible with common knowledge
        incompatible, incompat_reasoning, incompat_confidence = (
            self.inference_engine.incompatible_with(proposition, common_knowledge)
        )

        if incompatible and incompat_confidence > 0.7:
            explanation = (
                f"RULE 5: Proposition incompatible with common knowledge (denied as false). "
                f"{incompat_reasoning}"
            )
            return (ResponseType.NEGO, explanation, 5)

        # Neither follows nor incompatible - doubt it
        explanation = (
            "RULE 5: Proposition neither follows from nor contradicts common knowledge. "
            "It is irrelevant to the disputation."
        )
        return (ResponseType.DUBITO, explanation, 5)

    def explain_rules(self) -> str:
        """
        Return a human-readable explanation of Burley's rules.

        Returns:
            String explaining all five rules
        """
        return """
Walter Burley's Five Rules for Obligationes (in order of precedence):

1. If the proposition NECESSARILY FOLLOWS from (positum + concessa):
   → Must respond CONCEDO (grant it)

2. If the proposition is INCOMPATIBLE with (positum + concessa):
   → Must respond NEGO (deny it)

3. If the proposition NECESSARILY FOLLOWS from (positum + concessa + common knowledge):
   → Must respond CONCEDO (grant it)

4. If the proposition is INCOMPATIBLE with (positum + concessa + common knowledge):
   → Must respond NEGO (deny it)

5. Otherwise, respond based on truth in common knowledge alone:
   → CONCEDO if true in common knowledge
   → NEGO if false in common knowledge
   → DUBITO if irrelevant (neither true nor false)

The Respondent MUST follow these rules in order and CANNOT deviate,
even if doing so would lead to contradiction.
        """.strip()
