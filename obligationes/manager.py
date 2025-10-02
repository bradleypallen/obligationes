"""
Disputation Manager for orchestrating obligationes disputations.

This module implements the main orchestration layer that manages the flow
of a complete disputation from initialization through judgment.
"""

from typing import Optional, Set, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from obligationes.state import ObligationesState
from obligationes.inference import LLMInferenceEngine
from obligationes.rules import BurleyRulesEngine
from obligationes.agents import (
    RespondentAgent,
    OpponentAgent,
    OpponentStrategy,
)


@dataclass
class DisputationConfig:
    """Configuration for a disputation."""

    max_turns: int = 10
    opponent_strategy: OpponentStrategy = OpponentStrategy.BALANCED
    verbose: bool = True
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    vendor: Optional[str] = None  # "openai", "anthropic", or None for auto-detect


@dataclass
class DisputationResult:
    """Complete result of a disputation."""

    winner: str
    reason: str
    positum: str
    total_turns: int
    final_consistent: bool
    transcript: List[Dict[str, Any]]
    judgment: Dict[str, Any]
    state: ObligationesState
    started_at: str
    ended_at: str
    duration_seconds: float


# Default medieval common knowledge
DEFAULT_COMMON_KNOWLEDGE = {
    "All men are mortal",
    "God is omnipotent",
    "The soul is immortal",
    "Fire rises naturally",
    "Heavy objects fall faster than light ones",
    "The Earth is at the center of the universe",
    "All effects have causes",
    "Contradictions cannot be true",
}


class DisputationManager:
    """
    Manages the complete flow of an obligationes disputation.

    The manager orchestrates:
    1. Initialization with positum and common knowledge
    2. Turn-by-turn exchanges between Opponent and Respondent
    3. Consistency checking after each turn
    4. Winner determination based on contradiction detection

    Attributes:
        state: Current disputation state
        respondent: Respondent agent
        opponent: Opponent agent
        config: Disputation configuration
    """

    def __init__(
        self,
        common_knowledge: Optional[Set[str]] = None,
        config: Optional[DisputationConfig] = None,
        inference_engine: Optional[LLMInferenceEngine] = None,
    ):
        """
        Initialize the disputation manager.

        Args:
            common_knowledge: Background facts both parties accept (uses default if None)
            config: Disputation configuration (uses default if None)
            inference_engine: Shared LLM inference engine (creates default if None)
        """
        self.config = config or DisputationConfig()

        # Use provided or default common knowledge
        if common_knowledge is None:
            common_knowledge = DEFAULT_COMMON_KNOWLEDGE.copy()

        # Initialize state
        self.state = ObligationesState(common_knowledge=common_knowledge)

        # Create or use provided inference engine
        self.inference_engine = inference_engine or LLMInferenceEngine(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            vendor=self.config.vendor,
        )

        # Create rules engine
        self.rules_engine = BurleyRulesEngine(inference_engine=self.inference_engine)

        # Create agents
        self.respondent = RespondentAgent(
            rules_engine=self.rules_engine, inference_engine=self.inference_engine
        )
        self.opponent = OpponentAgent(
            strategy=self.config.opponent_strategy,
            inference_engine=self.inference_engine,
        )

        # Track timing
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def run_disputation(self, positum: str) -> DisputationResult:
        """
        Run a complete disputation from start to finish.

        Args:
            positum: The initial position the Respondent must defend

        Returns:
            DisputationResult with complete outcome and analysis
        """
        # Start timing
        self.start_time = datetime.utcnow()

        # Initialize
        if self.config.verbose:
            print(f"\n{'='*70}")
            print("OBLIGATIONES DISPUTATION")
            print(f"{'='*70}")
            print(f"\nPositum: {positum}")
            print(f"Strategy: {self.config.opponent_strategy.value.upper()}")
            print(f"Max Turns: {self.config.max_turns}")
            print(f"\n{'='*70}\n")

        # Turn 0: Check if positum is self-contradictory (per Novaes 2005)
        # R(φ₀) = 0 iff φ₀ ⊢⊥ (reject if self-contradictory)
        # R(φ₀) = 1 iff φ₀ ⊬⊥ (accept if not self-contradictory)
        if self.config.verbose:
            print("\n--- Turn 0 (Positum) ---\n")
            print(f"Opponent proposes: {positum}")

        # Check if positum is self-contradictory
        positum_self_contradictory, contradiction_reasoning = (
            self.inference_engine.is_self_contradictory(positum)
        )

        from obligationes.state import ResponseType

        if positum_self_contradictory:
            # Positum is self-contradictory - reject it
            if self.config.verbose:
                print("\nRespondent: NEGO")
                print(
                    f"Reasoning: The positum is self-contradictory and cannot be defended. {contradiction_reasoning}"
                )
                print("Rule Applied: Positum rejection (self-contradictory)")
                print("\n❌ DISPUTATION CANNOT BEGIN - POSITUM REJECTED")

            # End immediately - no game starts
            self.end_time = datetime.utcnow()
            duration = (self.end_time - self.start_time).total_seconds()

            # Create result indicating no game - Respondent wins by correctly rejecting invalid positum
            return DisputationResult(
                winner="RESPONDENT",
                reason="Positum was self-contradictory and correctly rejected",
                positum=positum,
                total_turns=0,
                final_consistent=True,
                transcript=[],
                judgment={
                    "winner": "RESPONDENT",
                    "reason": "Positum was self-contradictory and correctly rejected",
                    "rule_violations": [],
                    "key_moments": ["Positum rejected as self-contradictory"],
                    "overall_assessment": "Disputation could not begin due to self-contradictory positum. Respondent correctly rejected it.",
                    "final_consistent": True,
                },
                state=self.state,
                started_at=self.start_time.isoformat(),
                ended_at=self.end_time.isoformat(),
                duration_seconds=duration,
            )

        # Positum is not self-contradictory - accept it
        if self.config.verbose:
            print("\nRespondent: CONCEDO")
            print(
                "Reasoning: The positum is not self-contradictory and is accepted by obligation. The Respondent commits to defend this position."
            )
            print("Rule Applied: Positum acceptance (non-contradictory)")

        # Set positum and record the CONCEDO response
        self.state.set_positum(positum)
        self.state.add_response(
            positum,
            ResponseType.CONCEDO,
            "The positum is not self-contradictory and is accepted by obligation. The Respondent commits to defend this position.",
            0,  # Rule 0 = positum acceptance
        )
        self.state.history[-1].consistency_maintained = True

        # Run disputation loop
        contradiction_found = False
        for turn in range(self.config.max_turns):
            if self.config.verbose:
                print(f"\n--- Turn {turn + 1} ---\n")

            # Opponent proposes
            proposal = self.opponent.propose_proposition(self.state)

            if self.config.verbose:
                print(f"Opponent: {proposal['proposition']}")
                print(f"Strategy: {proposal['strategy_note']}")

            # Respondent evaluates
            evaluation = self.respondent.evaluate_proposition(
                proposal["proposition"], self.state
            )

            if self.config.verbose:
                print(f"\nRespondent: {evaluation['response'].value.upper()}")
                print(f"Reasoning: {evaluation['reasoning']}")
                print(f"Rule Applied: {evaluation['rule_applied']}")

                if evaluation["trap_detected"]:
                    print(f"⚠️  {evaluation['trap_analysis']}")

            # Update state
            self.state.add_response(
                proposal["proposition"],
                evaluation["response"],
                evaluation["reasoning"],
                evaluation["rule_applied"],
            )

            # Check consistency
            # Build full commitment set: CONCEDO + negation of NEGO
            all_commitments = self.state.get_all_commitments()
            all_negations = self.state.get_all_negations()

            # Add negated versions of NEGO responses to the commitment set
            full_commitment_set = all_commitments.copy()
            for negated_prop in all_negations:
                full_commitment_set.add(f"NOT({negated_prop})")

            if len(full_commitment_set) > 1:
                consistent, contradictions, reasoning = (
                    self.inference_engine.check_consistency(full_commitment_set)
                )

                if not consistent:
                    contradiction_found = True
                    if self.config.verbose:
                        print("\n❌ CONTRADICTION DETECTED!")
                        print(f"Contradictions: {contradictions}")

                    # Mark in history
                    self.state.history[-1].consistency_maintained = False
                    break
                else:
                    self.state.history[-1].consistency_maintained = True

        # End timing
        self.end_time = datetime.utcnow()
        duration = (self.end_time - self.start_time).total_seconds()

        # Determine outcome
        if contradiction_found:
            winner = "OPPONENT"
            reason = "Respondent fell into contradiction"
        else:
            winner = "RESPONDENT"
            reason = "Maintained consistency through all turns"

        self.state.end_disputation(winner, reason)

        if self.config.verbose:
            print(f"\n{'='*70}")
            print("FINAL RESULT")
            print(f"{'='*70}")
            print(f"\nWinner: {winner}")
            print(f"Reason: {reason}")
            print(f"\n{'='*70}\n")

        # Create result
        return DisputationResult(
            winner=winner,
            reason=reason,
            positum=positum,
            total_turns=self.state.turn_count,
            final_consistent=not contradiction_found,
            transcript=self._format_transcript(),
            judgment={
                "winner": winner,
                "reason": reason,
                "final_consistent": not contradiction_found,
                "overall_assessment": f"Disputation completed with {self.state.turn_count} turns.",
            },
            state=self.state,
            started_at=self.start_time.isoformat(),
            ended_at=self.end_time.isoformat(),
            duration_seconds=duration,
        )

    def step(self, proposition: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a single step of the disputation.

        This allows for manual/interactive control of the disputation flow.

        Args:
            proposition: Optional proposition to evaluate. If None, Opponent generates one.

        Returns:
            Dictionary with step results
        """
        # Generate proposition if not provided
        if proposition is None:
            proposal = self.opponent.propose_proposition(self.state)
            proposition = proposal["proposition"]
            strategy_note = proposal["strategy_note"]
        else:
            strategy_note = "User provided"

        # Respondent evaluates
        evaluation = self.respondent.evaluate_proposition(proposition, self.state)

        # Update state
        self.state.add_response(
            proposition,
            evaluation["response"],
            evaluation["reasoning"],
            evaluation["rule_applied"],
        )

        # Check consistency
        all_commitments = self.state.get_all_commitments()
        consistent = True
        contradictions: list[str] = []

        if len(all_commitments) > 1:
            consistent, contradictions, _ = self.inference_engine.check_consistency(
                all_commitments
            )
            self.state.history[-1].consistency_maintained = consistent

        return {
            "proposition": proposition,
            "strategy_note": strategy_note,
            "response": evaluation["response"],
            "reasoning": evaluation["reasoning"],
            "rule_applied": evaluation["rule_applied"],
            "trap_detected": evaluation["trap_detected"],
            "trap_analysis": evaluation["trap_analysis"],
            "consistent": consistent,
            "contradictions": contradictions,
            "turn": self.state.turn_count - 1,
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the disputation.

        Returns:
            Dictionary with current state information
        """
        all_commitments = self.state.get_all_commitments()
        consistent = True

        if len(all_commitments) > 1:
            consistent, _, _ = self.inference_engine.check_consistency(all_commitments)

        return {
            "active": self.state.disputation_active,
            "positum": self.state.positum.content if self.state.positum else None,
            "turn_count": self.state.turn_count,
            "commitments": list(all_commitments),
            "negations": list(self.state.get_all_negations()),
            "doubts": list(self.state.get_all_doubts()),
            "consistent": consistent,
            "winner": self.state.metadata.get("winner"),
            "reason": self.state.metadata.get("reason"),
        }

    def _format_transcript(self) -> List[Dict[str, Any]]:
        """Format the complete transcript for output."""
        transcript = []
        for turn in self.state.history:
            transcript.append(
                {
                    "turn": turn.number,
                    "proposition": turn.proposition,
                    "response": turn.response.value,
                    "reasoning": turn.reasoning,
                    "rule_applied": turn.rule_applied,
                    "consistency_maintained": turn.consistency_maintained,
                }
            )
        return transcript

    def save_transcript(self, filepath: str) -> None:
        """
        Save the disputation transcript to a JSON file.

        Args:
            filepath: Path to save the transcript
        """
        self.state.to_json(filepath)

    @classmethod
    def from_transcript(cls, filepath: str) -> "DisputationManager":
        """
        Load a disputation from a saved transcript.

        Args:
            filepath: Path to the transcript file

        Returns:
            DisputationManager with restored state
        """
        state = ObligationesState.from_json(filepath)
        manager = cls(common_knowledge=state.common_knowledge)
        manager.state = state
        return manager


def create_disputation(
    positum: str,
    common_knowledge: Optional[Set[str]] = None,
    max_turns: int = 10,
    strategy: OpponentStrategy = OpponentStrategy.BALANCED,
    verbose: bool = True,
    model_name: str = "gpt-4",
) -> DisputationResult:
    """
    Convenience function to create and run a disputation.

    Args:
        positum: Initial position to defend
        common_knowledge: Background facts (uses default if None)
        max_turns: Maximum number of turns
        strategy: Opponent strategy
        verbose: Print progress
        model_name: LLM model to use

    Returns:
        DisputationResult with complete outcome
    """
    config = DisputationConfig(
        max_turns=max_turns,
        opponent_strategy=strategy,
        verbose=verbose,
        model_name=model_name,
    )

    manager = DisputationManager(common_knowledge=common_knowledge, config=config)

    return manager.run_disputation(positum)
