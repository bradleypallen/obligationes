"""
Agent implementations for obligationes disputations.

This module implements the two key participants in an obligationes disputation:
- RespondentAgent: Defends the positum by following Burley's rules strictly
- OpponentAgent: Proposes propositions strategically to force contradictions
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

        # Create chains
        self._proposal_chain = self._create_proposal_chain()
        self._planning_chain = self._create_planning_chain()

        # Track strategic plan
        self.strategic_plan: Optional[str] = None

    def propose_proposition(self, state: ObligationesState) -> Dict[str, Any]:
        """
        Generate a strategic proposition to challenge the Respondent.

        Args:
            state: Current disputation state

        Returns:
            Dictionary with proposition and strategic analysis
        """
        # Create or update strategic plan
        if self.strategic_plan is None or state.turn_count == 1:
            # First turn: create initial plan
            self.strategic_plan = self._create_strategic_plan(state)

        # Get current commitments
        commitments = state.get_all_commitments()
        negations = state.get_all_negations()

        # Get all previously proposed propositions (to avoid repeats)
        proposed_already = set()
        if state.positum:
            proposed_already.add(state.positum.content)
        for turn in state.history:
            proposed_already.add(turn.proposition)

        # Format state for LLM
        commitments_text = (
            "\n".join(f"- {c}" for c in commitments) if commitments else "None yet"
        )
        negations_text = (
            "\n".join(f"- {n}" for n in negations) if negations else "None yet"
        )
        proposed_text = (
            "\n".join(f"- {p}" for p in proposed_already)
            if proposed_already
            else "None yet"
        )

        # Generate proposal using strategic plan
        result = self._proposal_chain.invoke(
            {
                "strategy": self.strategy.value,
                "strategic_plan": self.strategic_plan,
                "commitments": commitments_text,
                "negations": negations_text,
                "proposed_already": proposed_text,
                "turn_count": state.turn_count,
                "format_instructions": self.parser.get_format_instructions(),
            }
        )

        # Parse result
        content = result.content if hasattr(result, "content") else str(result)
        proposal = self._parse_proposal(content)

        # CRITICAL: Enforce no repeats programmatically using semantic equivalence
        # If LLM repeated a proposition, retry with stronger constraint
        max_retries = 3
        retry_count = 0

        def is_semantically_duplicate(new_prop: str, existing_props: set) -> bool:
            """Check if new proposition is semantically equivalent to any existing one."""
            # First, quick exact match check
            if new_prop in existing_props:
                return True
            # Then check semantic equivalence using LLM
            for existing_prop in existing_props:
                equivalent, _ = self.inference_engine.semantically_equivalent(
                    new_prop, existing_prop
                )
                if equivalent:
                    return True
            return False

        while is_semantically_duplicate(proposal.proposition, proposed_already) and retry_count < max_retries:
            retry_count += 1
            import sys
            print(
                f"⚠️  Opponent repeated proposition (attempt {retry_count}): '{proposal.proposition}'",
                file=sys.stderr
            )
            # Regenerate with explicit rejection of the repeated proposition
            result = self._proposal_chain.invoke(
                {
                    "strategy": self.strategy.value,
                    "strategic_plan": self.strategic_plan,
                    "commitments": commitments_text,
                    "negations": negations_text,
                    "proposed_already": proposed_text + f"\n- {proposal.proposition} (JUST ATTEMPTED - DO NOT USE)",
                    "turn_count": state.turn_count,
                    "format_instructions": self.parser.get_format_instructions(),
                }
            )
            content = result.content if hasattr(result, "content") else str(result)
            proposal = self._parse_proposal(content)

        # Final check: if still repeated after retries, add warning to strategy note
        if is_semantically_duplicate(proposal.proposition, proposed_already):
            import sys
            print(
                f"⚠️  WARNING: Opponent still repeated after {max_retries} retries: '{proposal.proposition}'",
                file=sys.stderr
            )
            proposal.strategy_note = (
                f"[WARNING: Repeated proposition after {max_retries} retries] "
                + proposal.strategy_note
            )

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

When the Respondent NEGOs a proposition, they commit to its NEGATION.
So if they NEGO "P", they are committed to "NOT P".

STRATEGY: {strategy}

Strategy guidelines:
- BALANCED: Mix direct and indirect approaches, build toward contradictions methodically
- AGGRESSIVE: Aim for quick contradictions, exploit obvious tensions immediately
- PEDAGOGICAL: Clearly demonstrate logical principles, make the reasoning transparent

ADVANCED TACTICS TO FORCE CONTRADICTIONS:

1. **Exploit Negations**: If they've NEGO'd propositions, find what follows from those negations combined with their commitments.

2. **Use Conditionals**: Propose "If A then B" where:
   - A is something they've committed to
   - B creates tension with their other commitments or negations

3. **Use Disjunctions**: Propose "A or B" where both disjuncts create problems:
   - If they CONCEDO, you can force them to commit to both sides later
   - If they NEGO, they commit to "NOT A AND NOT B" which may contradict commitments

4. **Set Multi-Step Traps**:
   - First, get them to commit to intermediate propositions
   - Then propose something that follows from those but contradicts the original positum

5. **Exploit Logical Equivalences**:
   - "A or B" is equivalent to "NOT A implies B"
   - Use this to force commitments that seem harmless but build toward contradiction

6. **Test Boundary Cases**: If positum is "A or B", test:
   - "A and NOT B"
   - "B and NOT A"
   - "A and B"
   - "NOT A and NOT B" (this should be NEGO'd)

STRATEGIC PLAN:
{strategic_plan}

Current state:
Turn: {turn_count}

Respondent's commitments (positum + concessa):
{commitments}

Respondent's denials (negata - they are committed to the NEGATION of each):
{negations}

Propositions already used (DO NOT REPEAT):
{proposed_already}

CRITICAL RULES:
1. You MUST propose a NEW proposition that has NOT been proposed before
2. Analyze what the Respondent is committed to (concessa + negation of negata)
3. Find logical tensions between these commitments
4. Propose something that exploits those tensions

Think strategically:
- What step of your strategic plan should you execute now?
- What logical consequences follow from their commitments AND negations together?
- What would force them to either contradict a commitment or contradict a negation?
- Can you use a conditional or disjunction to set a trap?

Your proposition should advance the strategic plan while adapting to the current state.

{format_instructions}""",
                ),
                ("human", "Generate your next strategic proposition."),
            ]
        )

        return prompt | self.llm

    def _create_planning_chain(self) -> Any:
        """Create the chain for strategic planning."""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a strategic planner for the Opponent in an obligationes disputation.

Your task is to analyze the positum and create a multi-turn plan to force the Respondent into contradiction.

The Respondent must follow Burley's rules mechanically:
- If a proposition follows from their commitments → CONCEDO
- If a proposition contradicts their commitments → NEGO
- When they NEGO, they commit to the negation

Your job: Create a 3-5 step plan that builds toward forcing an inevitable contradiction.

STRATEGY: {strategy}

POSITUM: {positum}

ANALYSIS STEPS:
1. **Identify Logical Structure**: What type of statement is the positum?
   - Disjunction (A or B)?
   - Conjunction (A and B)?
   - Conditional (If A then B)?
   - Universal (All X are Y)?
   - Simple atomic proposition?

2. **Find Vulnerabilities**: What logical tensions can be exploited?
   - For "A or B": Can you force commitment to NOT A, then NOT B?
   - For "A and B": Can you get them to deny either A or B?
   - For "If A then B": Can you force A and NOT B?
   - For "All X are Y": Can you find an X that is NOT Y?

3. **Plan the Trap**: Design a sequence of propositions:
   - Step 1: What should you propose first?
   - Step 2: Based on likely response, what next?
   - Step 3: How do you close the trap?
   - Fallback: If they respond unexpectedly, how do you adapt?

4. **Consider Their Commitments**: Remember that when they NEGO a proposition,
   they commit to its negation. Plan how to use these negations against them.

Create a concise strategic plan (3-5 sentences) that:
- Identifies the key vulnerability in the positum
- Outlines the 3-5 step sequence of propositions to force contradiction
- Explains why this sequence should work
""",
                ),
                ("human", "Analyze the positum and create a strategic plan to force contradiction."),
            ]
        )

        return prompt | self.llm

    def _create_strategic_plan(self, state: ObligationesState) -> str:
        """Create a strategic plan based on the positum."""
        positum = state.positum.content if state.positum else "None"

        result = self._planning_chain.invoke(
            {
                "strategy": self.strategy.value,
                "positum": positum,
            }
        )

        content = result.content if hasattr(result, "content") else str(result)
        return content

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


