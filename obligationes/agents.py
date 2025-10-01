"""
Agent implementations for obligationes disputations.

This module implements the three key participants in an obligationes disputation:
- RespondentAgent: Defends the positum by following Burley's rules strictly
- OpponentAgent: Proposes propositions strategically to force contradictions
- JudgeAgent: Evaluates the disputation and determines the winner
"""

from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from obligationes.state import ObligationesState, ResponseType, Turn
from obligationes.rules import BurleyRulesEngine
from obligationes.inference import LLMInferenceEngine


class OpponentStrategy(Enum):
    """Strategy options for the Opponent agent."""

    BALANCED = "balanced"  # Mix of direct and indirect approaches
    AGGRESSIVE = "aggressive"  # Focus on forcing contradictions quickly
    PEDAGOGICAL = "pedagogical"  # Demonstrate logical principles clearly


class RespondentResponse(BaseModel):
    """Response from the Respondent agent."""

    response: str = Field(description="The response: CONCEDO, NEGO, or DUBITO")
    reasoning: str = Field(description="Explanation for the response")
    rule_applied: int = Field(description="Which Burley rule was applied (1-5)")
    trap_detected: bool = Field(description="Whether a trap was detected")
    trap_analysis: str = Field(description="Analysis of any detected trap", default="")


class OpponentProposal(BaseModel):
    """Proposal from the Opponent agent."""

    proposition: str = Field(description="The proposition being proposed")
    strategy_note: str = Field(description="Explanation of the strategic intent")
    expected_response: str = Field(description="What response is anticipated")


class JudgmentResult(BaseModel):
    """Final judgment from the Judge agent."""

    winner: str = Field(description="OPPONENT or RESPONDENT")
    reason: str = Field(description="Detailed explanation of why this party won")
    rule_violations: List[str] = Field(
        description="Any rule violations found", default_factory=list
    )
    key_moments: List[str] = Field(
        description="Critical turns in the disputation", default_factory=list
    )
    overall_assessment: str = Field(
        description="Overall quality and fairness assessment"
    )


class RespondentAgent:
    """
    The Respondent agent in an obligationes disputation.

    The Respondent MUST defend the positum by following Burley's rules strictly.
    They cannot strategize or deviate from the rules, even if it leads to contradiction.

    This models the medieval obligation to maintain logical discipline regardless of outcome.

    Attributes:
        rules_engine: Burley's rules engine for evaluation
        inference_engine: LLM inference engine for trap detection
    """

    def __init__(
        self,
        rules_engine: Optional[BurleyRulesEngine] = None,
        inference_engine: Optional[LLMInferenceEngine] = None,
    ):
        """
        Initialize the Respondent agent.

        Args:
            rules_engine: Burley's rules engine (creates default if None)
            inference_engine: LLM inference engine (creates default if None)
        """
        self.rules_engine = rules_engine or BurleyRulesEngine()
        self.inference_engine = inference_engine or LLMInferenceEngine()

    def evaluate_proposition(
        self, proposition: str, state: ObligationesState
    ) -> Dict[str, Any]:
        """
        Evaluate a proposition according to Burley's rules.

        The Respondent MUST follow the rules and CANNOT deviate, even if
        they detect a trap that will lead to contradiction.

        Args:
            proposition: The proposition to evaluate
            state: Current disputation state

        Returns:
            Dictionary with response, reasoning, rule_applied, and trap analysis
        """
        # Apply Burley's rules to get required response
        response_type, reasoning, rule_applied = self.rules_engine.evaluate_proposition(
            proposition, state
        )

        # Detect if this is a trap (but cannot avoid it)
        trap_detected, trap_analysis = self._analyze_for_trap(
            proposition, response_type, state
        )

        return {
            "response": response_type,
            "reasoning": reasoning,
            "rule_applied": rule_applied,
            "trap_detected": trap_detected,
            "trap_analysis": trap_analysis,
        }

    def _analyze_for_trap(
        self,
        proposition: str,
        required_response: ResponseType,
        state: ObligationesState,
    ) -> Tuple[bool, str]:
        """
        Analyze if the required response will create a future contradiction.

        This is for informational purposes only - the Respondent cannot
        change their response based on this analysis.

        Args:
            proposition: The proposition being evaluated
            required_response: The response required by rules
            state: Current state

        Returns:
            Tuple of (is_trap, analysis_explanation)
        """
        # Simulate accepting this response
        simulated_commitments = state.get_all_commitments().copy()

        if required_response == ResponseType.CONCEDO:
            simulated_commitments.add(proposition)

        # Check if simulated state would be consistent
        if len(simulated_commitments) > 1:
            consistent, contradictions, reasoning = (
                self.inference_engine.check_consistency(simulated_commitments)
            )

            if not consistent:
                analysis = (
                    f"TRAP DETECTED: Responding {required_response.value.upper()} to "
                    f"'{proposition}' will create a contradiction. "
                    f"Contradictions: {contradictions}. "
                    f"However, Burley's rules require this response."
                )
                return True, analysis

        return False, "No trap detected in this move."


class OpponentAgent:
    """
    The Opponent agent in an obligationes disputation.

    The Opponent proposes propositions strategically to force the Respondent
    into a logical contradiction. Different strategies can be employed.

    Attributes:
        strategy: The strategy to use (balanced, aggressive, pedagogical)
        inference_engine: LLM inference engine for strategic analysis
        llm: Language model for generating proposals
    """

    def __init__(
        self,
        strategy: OpponentStrategy = OpponentStrategy.BALANCED,
        inference_engine: Optional[LLMInferenceEngine] = None,
    ):
        """
        Initialize the Opponent agent.

        Args:
            strategy: Strategy to employ
            inference_engine: LLM inference engine (creates default if None)
        """
        self.strategy = strategy
        self.inference_engine = inference_engine or LLMInferenceEngine()
        self.llm = self.inference_engine.llm

        # Create output parser
        self.parser = PydanticOutputParser(pydantic_object=OpponentProposal)

        # Create proposal chain
        self._proposal_chain = self._create_proposal_chain()

    def propose_proposition(self, state: ObligationesState) -> Dict[str, Any]:
        """
        Generate a strategic proposition to challenge the Respondent.

        Args:
            state: Current disputation state

        Returns:
            Dictionary with proposition and strategic analysis
        """
        # Get current commitments
        commitments = state.get_all_commitments()
        negations = state.get_all_negations()

        # Format state for LLM
        commitments_text = (
            "\n".join(f"- {c}" for c in commitments) if commitments else "None yet"
        )
        negations_text = (
            "\n".join(f"- {n}" for n in negations) if negations else "None yet"
        )

        # Generate proposal
        result = self._proposal_chain.invoke(
            {
                "strategy": self.strategy.value,
                "commitments": commitments_text,
                "negations": negations_text,
                "turn_count": state.turn_count,
                "format_instructions": self.parser.get_format_instructions(),
            }
        )

        # Parse result
        content = result.content if hasattr(result, "content") else str(result)
        proposal = self._parse_proposal(content)

        return {
            "proposition": proposal.proposition,
            "strategy_note": proposal.strategy_note,
            "expected_response": proposal.expected_response,
        }

    def _create_proposal_chain(self):
        """Create the LLM chain for generating proposals."""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are the Opponent in a medieval obligationes disputation.

Your goal is to propose propositions that will force the Respondent into a logical contradiction.

The Respondent has committed to defending a positum and must follow Burley's rules:
1. If your proposition follows from their commitments → they must CONCEDO
2. If your proposition contradicts their commitments → they must NEGO
3-5. Similar rules involving common knowledge

STRATEGY: {strategy}

Strategy guidelines:
- BALANCED: Mix direct and indirect approaches, build toward contradictions methodically
- AGGRESSIVE: Aim for quick contradictions, exploit obvious tensions immediately
- PEDAGOGICAL: Clearly demonstrate logical principles, make the reasoning transparent

Current state:
Turn: {turn_count}

Respondent's commitments (positum + concessa):
{commitments}

Respondent's denials (negata):
{negations}

Generate a proposition that advances your strategy. Consider:
- What would force them into a contradiction?
- What follows from their commitments that contradicts something else?
- What builds toward a multi-step trap?

{format_instructions}""",
                ),
                ("human", "Generate your next strategic proposition."),
            ]
        )

        return prompt | self.llm

    def _parse_proposal(self, content: str) -> OpponentProposal:
        """Parse LLM response with fallback strategies."""
        try:
            return self.parser.parse(content)
        except Exception:
            # Fallback: extract proposition manually
            import re
            import json

            # Try JSON extraction
            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL
            )
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                    return OpponentProposal(**data)
                except Exception:
                    pass

            # Ultimate fallback: use content as proposition
            return OpponentProposal(
                proposition=content[:200],  # First 200 chars
                strategy_note="Generated from fallback parsing",
                expected_response="UNKNOWN",
            )


class JudgeAgent:
    """
    The Judge agent evaluates an obligationes disputation.

    The Judge determines the winner by:
    - Checking if the Respondent maintained consistency
    - Verifying that Burley's rules were followed correctly
    - Assessing the quality of the disputation

    Attributes:
        inference_engine: LLM inference engine for analysis
        llm: Language model for generating judgments
    """

    def __init__(self, inference_engine: Optional[LLMInferenceEngine] = None):
        """
        Initialize the Judge agent.

        Args:
            inference_engine: LLM inference engine (creates default if None)
        """
        self.inference_engine = inference_engine or LLMInferenceEngine()
        self.llm = self.inference_engine.llm

        # Create output parser
        self.parser = PydanticOutputParser(pydantic_object=JudgmentResult)

        # Create judgment chain
        self._judgment_chain = self._create_judgment_chain()

    def judge_disputation(self, state: ObligationesState) -> Dict[str, Any]:
        """
        Judge a completed disputation and determine the winner.

        Args:
            state: Final disputation state with complete history

        Returns:
            Dictionary with winner, reason, and detailed analysis
        """
        # Check final consistency
        all_commitments = state.get_all_commitments()
        consistent, contradictions, consistency_reasoning = (
            self.inference_engine.check_consistency(all_commitments)
        )

        # Format transcript for analysis
        transcript_text = self._format_transcript(state.history)

        # Get LLM judgment
        result = self._judgment_chain.invoke(
            {
                "positum": state.positum.content if state.positum else "None",
                "transcript": transcript_text,
                "final_consistent": consistent,
                "contradictions": str(contradictions),
                "total_turns": state.turn_count,
                "format_instructions": self.parser.get_format_instructions(),
            }
        )

        # Parse result
        content = result.content if hasattr(result, "content") else str(result)
        judgment = self._parse_judgment(content)

        return {
            "winner": judgment.winner,
            "reason": judgment.reason,
            "rule_violations": judgment.rule_violations,
            "key_moments": judgment.key_moments,
            "overall_assessment": judgment.overall_assessment,
            "final_consistent": consistent,
        }

    def _create_judgment_chain(self):
        """Create the LLM chain for generating judgments."""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a Judge in a medieval obligationes disputation.

Your task is to determine the winner based on strict rules:

WINNER DETERMINATION:
- OPPONENT wins IF the Respondent falls into contradiction
- RESPONDENT wins IF they maintain consistency through all turns

EVALUATION CRITERIA:
1. Final consistency: Are all commitments logically compatible?
2. Rule compliance: Did the Respondent follow Burley's rules correctly?
3. Quality: Was the disputation intellectually rigorous?

Positum (what Respondent defends): {positum}

Transcript:
{transcript}

Final Consistency: {final_consistent}
Contradictions Found: {contradictions}
Total Turns: {total_turns}

Analyze the disputation and determine:
- Who won and why
- Any rule violations
- Key moments that determined the outcome
- Overall assessment of quality

{format_instructions}""",
                ),
                ("human", "Provide your judgment."),
            ]
        )

        return prompt | self.llm

    def _format_transcript(self, history: List[Turn]) -> str:
        """Format turn history for readability."""
        lines = []
        for turn in history:
            lines.append(
                f"Turn {turn.number}: "
                f"Proposition: '{turn.proposition}' → "
                f"Response: {turn.response.value.upper()} "
                f"(Rule {turn.rule_applied})"
            )
        return "\n".join(lines)

    def _parse_judgment(self, content: str) -> JudgmentResult:
        """Parse LLM judgment with fallback strategies."""
        try:
            return self.parser.parse(content)
        except Exception:
            # Fallback: basic extraction
            import re
            import json

            # Try JSON extraction
            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL
            )
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                    return JudgmentResult(**data)
                except Exception:
                    pass

            # Determine winner from keywords
            winner = "OPPONENT"
            if (
                "respondent wins" in content.lower()
                or "respondent maintained" in content.lower()
            ):
                winner = "RESPONDENT"

            return JudgmentResult(
                winner=winner,
                reason=content[:500],
                overall_assessment="Generated from fallback parsing",
            )
