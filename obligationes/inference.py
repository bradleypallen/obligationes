"""
LLM-based logical inference engine for obligationes.

This module implements the core reasoning capabilities using Large Language Models
to determine logical relationships between propositions in natural language.
"""

from typing import Set, Any, Optional, Tuple, List
from enum import Enum
import json
import re
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


class InferenceType(Enum):
    """Types of logical inference patterns."""

    MODUS_PONENS = "modus_ponens"
    MODUS_TOLLENS = "modus_tollens"
    SYLLOGISM = "syllogism"
    CONJUNCTION = "conjunction"
    DISJUNCTION = "disjunction"
    CONTRADICTION = "contradiction"
    TRANSITIVE = "transitive"
    EQUIVALENCE = "equivalence"
    OTHER = "other"


class FollowsFromResult(BaseModel):
    """Result of checking if a proposition follows from premises."""

    follows: bool = Field(
        description="Whether the proposition necessarily follows from the premises"
    )
    inference_type: str = Field(
        description="Type of inference used (e.g., modus_ponens, syllogism)"
    )
    reasoning: str = Field(description="Step-by-step explanation of the inference")
    confidence: float = Field(description="Confidence level (0.0-1.0)", ge=0.0, le=1.0)


class IncompatibleWithResult(BaseModel):
    """Result of checking if a proposition is incompatible with premises."""

    incompatible: bool = Field(
        description="Whether the proposition is logically incompatible"
    )
    contradiction_type: str = Field(
        description="Type of incompatibility (direct, indirect, implicit)"
    )
    reasoning: str = Field(description="Explanation of the incompatibility")
    confidence: float = Field(description="Confidence level (0.0-1.0)", ge=0.0, le=1.0)


class ConsistencyResult(BaseModel):
    """Result of checking consistency of a set of propositions."""

    consistent: bool = Field(
        description="Whether the set of propositions is logically consistent"
    )
    contradictions: List[str] = Field(
        description="List of contradictory proposition pairs if inconsistent"
    )
    reasoning: str = Field(
        description="Explanation of consistency or contradictions found"
    )


class SemanticEquivalenceResult(BaseModel):
    """Result of checking semantic equivalence between two propositions."""

    equivalent: bool = Field(
        description="Whether the two propositions are semantically equivalent"
    )
    reasoning: str = Field(
        description="Explanation of why they are or are not equivalent"
    )


class SelfContradictionResult(BaseModel):
    """Result of checking if a single proposition is self-contradictory."""

    self_contradictory: bool = Field(
        description="Whether the proposition is internally self-contradictory"
    )
    reasoning: str = Field(
        description="Explanation of why it is or is not self-contradictory"
    )


class TruthValueResult(BaseModel):
    """Result of checking the truth value of a proposition in reality."""

    truth_value: str = Field(
        description="The truth value: 'true', 'false', or 'unknown'"
    )
    confidence: float = Field(
        description="Confidence in the assessment (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description="Explanation for the truth value assessment"
    )


class LLMInferenceEngine:
    """
    LLM-based logical reasoning engine.

    This class handles all logical inference operations using Large Language Models,
    enabling natural language reasoning while maintaining logical rigor.

    Attributes:
        llm: The language model instance (GPT-4, Claude, etc.)
        temperature: Temperature setting for LLM calls (0 for deterministic)
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        model_name: str = "gpt-4",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        vendor: Optional[str] = None,
    ):
        """
        Initialize the inference engine.

        Args:
            llm: Pre-configured LLM instance (if None, creates default)
            model_name: Name of the model to use (gpt-4, claude-3-opus-20240229, etc.)
            temperature: Temperature for LLM calls (0.0 for deterministic)
            api_key: API key for the LLM provider (if None, uses env vars)
            vendor: Vendor to use ("openai", "anthropic", or None for auto-detect)
        """
        import os

        self.temperature = temperature

        if llm is not None:
            self.llm = llm
        else:
            # Determine vendor
            if vendor is None or vendor == "auto":
                # Auto-detect from model name
                if "gpt" in model_name.lower() or "o1" in model_name.lower():
                    vendor = "openai"
                elif "claude" in model_name.lower():
                    vendor = "anthropic"
                else:
                    vendor = "openai"  # Default

            # Get API key from environment if not provided
            if api_key is None:
                if vendor == "openai":
                    api_key = os.getenv("OPENAI_API_KEY")
                elif vendor == "anthropic":
                    api_key = os.getenv("ANTHROPIC_API_KEY")

            # Create LLM instance
            kwargs = {"model": model_name, "temperature": temperature}
            if api_key is not None:
                kwargs["api_key"] = api_key

            if vendor == "anthropic":
                self.llm = ChatAnthropic(**kwargs)
            else:  # Default to OpenAI
                self.llm = ChatOpenAI(**kwargs)

        # Create output parsers
        self.follows_parser = PydanticOutputParser(pydantic_object=FollowsFromResult)
        self.incompatible_parser = PydanticOutputParser(
            pydantic_object=IncompatibleWithResult
        )
        self.consistency_parser = PydanticOutputParser(
            pydantic_object=ConsistencyResult
        )
        self.equivalence_parser = PydanticOutputParser(
            pydantic_object=SemanticEquivalenceResult
        )
        self.self_contradiction_parser = PydanticOutputParser(
            pydantic_object=SelfContradictionResult
        )
        self.truth_value_parser = PydanticOutputParser(
            pydantic_object=TruthValueResult
        )

        # Create chains
        self._follows_chain = self._create_follows_chain()
        self._incompatible_chain = self._create_incompatible_chain()
        self._consistency_chain = self._create_consistency_chain()
        self._equivalence_chain = self._create_equivalence_chain()
        self._self_contradiction_chain = self._create_self_contradiction_chain()
        self._truth_value_chain = self._create_truth_value_chain()

    def follows_from(
        self, proposition: str, premises: Set[str]
    ) -> Tuple[bool, str, float]:
        """
        Check if a proposition necessarily follows from premises.

        Args:
            proposition: The conclusion to check
            premises: Set of premise propositions

        Returns:
            Tuple of (follows, reasoning, confidence)
        """
        if not premises:
            return False, "No premises provided", 0.0

        try:
            premises_text = "\n".join(f"- {p}" for p in premises)

            result = self._follows_chain.invoke(
                {
                    "premises": premises_text,
                    "proposition": proposition,
                    "format_instructions": self.follows_parser.get_format_instructions(),
                }
            )

            # Parse the result - handle both AIMessage and string responses
            content = result.content if hasattr(result, "content") else str(result)
            parsed = self._parse_follows_result(content)
            return parsed.follows, parsed.reasoning, parsed.confidence

        except Exception as e:
            return False, f"Error in inference: {str(e)}", 0.0

    def incompatible_with(
        self, proposition: str, premises: Set[str]
    ) -> Tuple[bool, str, float]:
        """
        Check if a proposition is logically incompatible with premises.

        Args:
            proposition: The proposition to check
            premises: Set of premise propositions

        Returns:
            Tuple of (incompatible, reasoning, confidence)
        """
        if not premises:
            return False, "No premises provided", 0.0

        try:
            premises_text = "\n".join(f"- {p}" for p in premises)

            result = self._incompatible_chain.invoke(
                {
                    "premises": premises_text,
                    "proposition": proposition,
                    "format_instructions": self.incompatible_parser.get_format_instructions(),
                }
            )

            # Parse the result - handle both AIMessage and string responses
            content = result.content if hasattr(result, "content") else str(result)
            parsed = self._parse_incompatible_result(content)
            return parsed.incompatible, parsed.reasoning, parsed.confidence

        except Exception as e:
            return False, f"Error in incompatibility check: {str(e)}", 0.0

    def check_consistency(self, propositions: Set[str]) -> Tuple[bool, List[str], str]:
        """
        Check if a set of propositions is logically consistent.

        Args:
            propositions: Set of propositions to check

        Returns:
            Tuple of (consistent, contradictions, reasoning)
        """
        if len(propositions) < 2:
            return True, [], "Less than 2 propositions, trivially consistent"

        try:
            props_text = "\n".join(f"- {p}" for p in propositions)

            result = self._consistency_chain.invoke(
                {
                    "propositions": props_text,
                    "format_instructions": self.consistency_parser.get_format_instructions(),
                }
            )

            # Parse the result - handle both AIMessage and string responses
            content = result.content if hasattr(result, "content") else str(result)
            parsed = self._parse_consistency_result(content)
            return parsed.consistent, parsed.contradictions, parsed.reasoning

        except Exception as e:
            return True, [], f"Error in consistency check: {str(e)}"

    def is_self_contradictory(self, proposition: str) -> Tuple[bool, str]:
        """
        Check if a single proposition is internally self-contradictory.

        This checks whether the proposition itself contains a logical contradiction,
        such as "Socrates is both mortal and immortal" or "X is true and X is false".

        Args:
            proposition: The proposition to check

        Returns:
            Tuple of (self_contradictory, reasoning)
        """
        try:
            result = self._self_contradiction_chain.invoke(
                {
                    "proposition": proposition,
                    "format_instructions": self.self_contradiction_parser.get_format_instructions(),
                }
            )

            # Parse the result
            content = result.content if hasattr(result, "content") else str(result)
            parsed = self._parse_self_contradiction_result(content)
            return parsed.self_contradictory, parsed.reasoning

        except Exception as e:
            return False, f"Error in self-contradiction check: {str(e)}"

    def evaluate_truth_value(
        self, proposition: str, context: str = ""
    ) -> Tuple[str, float, str]:
        """
        Evaluate the truth value of a proposition based on common sense and reality.

        This is used for Burley's Rule 5 to determine if a proposition is true,
        false, or unknown in reality (not just in the explicit common knowledge set).

        Args:
            proposition: The proposition to evaluate
            context: Optional context about the disputation (e.g., "about medieval philosophy")

        Returns:
            Tuple of (truth_value, confidence, reasoning) where truth_value is
            'true', 'false', or 'unknown'
        """
        try:
            result = self._truth_value_chain.invoke(
                {
                    "proposition": proposition,
                    "context": context if context else "General common sense reasoning",
                    "format_instructions": self.truth_value_parser.get_format_instructions(),
                }
            )

            # Parse the result
            content = result.content if hasattr(result, "content") else str(result)
            parsed = self._parse_truth_value_result(content)
            return parsed.truth_value, parsed.confidence, parsed.reasoning

        except Exception as e:
            return "unknown", 0.0, f"Error in truth value evaluation: {str(e)}"

    def semantically_equivalent(self, prop1: str, prop2: str) -> Tuple[bool, str]:
        """
        Check if two propositions are semantically equivalent.

        This is used to detect when the Opponent proposes logically identical
        propositions with different wording (e.g., "A and B" vs "B and A").

        Args:
            prop1: First proposition
            prop2: Second proposition

        Returns:
            Tuple of (equivalent, reasoning)
        """
        # Quick check: if exactly the same, they're equivalent
        if prop1.strip().lower() == prop2.strip().lower():
            return True, "Propositions are identical"

        try:
            result = self._equivalence_chain.invoke(
                {
                    "prop1": prop1,
                    "prop2": prop2,
                    "format_instructions": self.equivalence_parser.get_format_instructions(),
                }
            )

            # Parse the result
            content = result.content if hasattr(result, "content") else str(result)
            parsed = self._parse_equivalence_result(content)
            return parsed.equivalent, parsed.reasoning

        except Exception as e:
            # On error, assume not equivalent (conservative)
            return False, f"Error in equivalence check: {str(e)}"

    def _create_follows_chain(self) -> Any:
        """Create the chain for checking logical inference."""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert in formal logic and medieval dialectical reasoning.

Your task is to determine if a proposition NECESSARILY FOLLOWS from a set of premises.

CRITICAL DEFINITION: "Necessarily follows" means:
- The conclusion MUST be true if ALL premises are true
- There is NO possible interpretation where premises are true and conclusion is false
- The inference is DEDUCTIVELY VALID (not merely plausible or probable)

Valid inference patterns include:
1. Modus Ponens: "If P then Q" + "P" ⊢ "Q"
2. Modus Tollens: "If P then Q" + "not Q" ⊢ "not P"
3. Universal Syllogism: "All A are B" + "X is A" ⊢ "X is B"
4. Conjunction: "P" + "Q" ⊢ "P and Q"
5. Disjunctive Syllogism: "P or Q" + "not P" ⊢ "Q"
6. Transitivity: "P implies Q" + "Q implies R" ⊢ "P implies R"

IMPORTANT:
- Distinguish between DEDUCTIVE VALIDITY and mere plausibility
- "Socrates is a man" + "All men are mortal" ⊢ "Socrates is mortal" (VALID)
- "Socrates is wise" does NOT follow from "Socrates is a man" (NOT VALID, even if plausible)
- Consider implicit logical structure in natural language

{format_instructions}""",
                ),
                (
                    "human",
                    """Premises:
{premises}

Proposition to evaluate:
{proposition}

Does this proposition NECESSARILY follow from the premises?""",
                ),
            ]
        )

        return prompt | self.llm

    def _create_incompatible_chain(self) -> Any:
        """Create the chain for checking logical incompatibility."""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert in formal logic and contradiction detection.

Your task is to determine if a proposition is LOGICALLY INCOMPATIBLE with a set of premises.

CRITICAL DEFINITION: "Logically incompatible" means:
- The proposition CANNOT be true if the premises are true
- Accepting both would create a LOGICAL CONTRADICTION
- There is NO consistent interpretation where both are true

Types of incompatibility:
1. Direct contradiction: "P" vs "not P"
2. Indirect contradiction: "All men are mortal" + "Socrates is a man" vs "Socrates is immortal"
3. Implicit contradiction: "X is larger than Y" + "Y is larger than Z" vs "Z is larger than X"

IMPORTANT:
- Distinguish between LOGICAL incompatibility and mere conflict
- "Socrates is mortal" is incompatible with "Socrates is immortal" (INCOMPATIBLE)
- "Socrates is wise" is NOT incompatible with "Socrates is a man" (COMPATIBLE)
- Consider multi-step inferences that reveal contradictions

{format_instructions}""",
                ),
                (
                    "human",
                    """Premises:
{premises}

Proposition to evaluate:
{proposition}

Is this proposition logically incompatible with the premises?""",
                ),
            ]
        )

        return prompt | self.llm

    def _create_consistency_chain(self) -> Any:
        """Create the chain for checking set consistency."""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert in formal logic and consistency analysis.

Your task is to determine if a SET of propositions is LOGICALLY CONSISTENT.

CRITICAL DEFINITION: "Logically consistent" means:
- There EXISTS at least one interpretation where ALL propositions are true
- NO logical contradiction can be derived from the set
- The propositions are mutually compatible

NOTATION:
- "NOT(P)" means the negation of proposition P
- If you see both "P" and "NOT(P)", that's a direct contradiction
- "NOT(A and B)" is logically equivalent to "NOT(A) or NOT(B)" (De Morgan's law)
- "NOT(A or B)" is logically equivalent to "NOT(A) and NOT(B)" (De Morgan's law)

CONDITIONAL STATEMENTS (CRITICAL):
- "If A then B" is logically equivalent to "NOT(A) or B"
- "NOT(A) → B" and "NOT(B) → A" are BOTH equivalent to "A or B"
- "NOT(A) → NOT(B)" and "NOT(B) → NOT(A)" are BOTH equivalent to "A or B"
- These are NOT contradictions - they're logically equivalent!

IMPORTANT LOGICAL EQUIVALENCES:
- "A or B" ≡ "If not A, then B" ≡ "If not B, then A"
- "NOT(A) → B" ≡ "NOT(B) → NOT(NOT(A))" ≡ "NOT(B) → A"
- Multiple conditionals can be consistent even if they form a logical chain

EXAMPLE OF CONSISTENT CONDITIONALS:
Given: "A or B" (exactly one is true)
These are ALL consistent with each other:
- "If not A, then B" (¬A → B)
- "If not B, then A" (¬B → A)
- "If A, then not B" (A → ¬B)
- "If B, then not A" (B → ¬A)

Check for:
1. Direct contradictions: "P" and "NOT(P)" in the set
2. Indirect contradictions: Derived through valid inference (modus ponens, etc.)
3. De Morgan contradictions: e.g., "NOT(A and B)" with "NOT(A or B)"
4. Circular contradictions: A chain of inferences leading to contradiction

IMPORTANT:
- ALL propositions must be considered together
- A set is inconsistent if ANY contradiction exists
- Report ALL contradictory pairs found
- Pay special attention to De Morgan's laws when analyzing NOT() propositions
- DO NOT flag logically equivalent conditionals as contradictions
- Check if there exists ANY truth assignment that makes ALL propositions true

{format_instructions}""",
                ),
                (
                    "human",
                    """Propositions:
{propositions}

Is this set of propositions logically consistent?""",
                ),
            ]
        )

        return prompt | self.llm

    def _create_equivalence_chain(self) -> Any:
        """Create the chain for checking semantic equivalence."""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert in formal logic and natural language semantics.

Your task is to determine if two propositions are SEMANTICALLY EQUIVALENT.

CRITICAL DEFINITION: "Semantically equivalent" means:
- Both propositions express the SAME logical content
- They have the SAME truth conditions
- One can be substituted for the other without changing logical meaning

Common equivalences to recognize:
1. Commutative: "A and B" ≡ "B and A"
2. Commutative: "A or B" ≡ "B or A"
3. Double negation: "not not P" ≡ "P"
4. Synonyms: "Socrates is mortal" ≡ "Socrates will die"
5. Rephrasing: "All men are mortal" ≡ "Every man is mortal"

NOT equivalent:
- Different logical structure: "A and B" ≠ "A or B"
- Different subjects: "Socrates is wise" ≠ "Plato is wise"
- Different predicates: "Socrates is mortal" ≠ "Socrates is wise"

{format_instructions}""",
                ),
                (
                    "human",
                    """Proposition 1: {prop1}

Proposition 2: {prop2}

Are these propositions semantically equivalent?""",
                ),
            ]
        )

        return prompt | self.llm

    def _create_self_contradiction_chain(self) -> Any:
        """Create the chain for checking self-contradiction."""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert in formal logic specializing in detecting contradictions.

Your task is to determine if a SINGLE proposition is INTERNALLY SELF-CONTRADICTORY.

CRITICAL DEFINITION: "Self-contradictory" means:
- The proposition asserts something AND its negation simultaneously
- The proposition cannot possibly be true under any circumstances
- The proposition contains mutually exclusive properties or states

Examples of SELF-CONTRADICTORY propositions:
1. "Socrates is both mortal and immortal" (mutually exclusive properties)
2. "X is true and X is false" (direct contradiction)
3. "The square circle exists" (contradictory definition)
4. "I am my own grandfather" (logical impossibility)
5. "All bachelors are married" (contradicts definition of bachelor)

Examples of NOT self-contradictory propositions:
1. "Socrates is mortal" (could be true or false, but not contradictory)
2. "The sky is green" (false but not contradictory)
3. "2+2=5" (false but not logically contradictory in itself)
4. "God exists" (controversial but not self-contradictory)

IMPORTANT: A proposition is only self-contradictory if it INTERNALLY contains a contradiction.
It is NOT self-contradictory merely because it contradicts external common knowledge or other propositions.

{format_instructions}""",
                ),
                (
                    "human",
                    """Proposition: {proposition}

Is this proposition internally self-contradictory?""",
                ),
            ]
        )

        return prompt | self.llm

    def _create_truth_value_chain(self) -> Any:
        """Create the chain for checking truth value in reality."""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert in evaluating the truth value of propositions based on common sense and general knowledge.

Your task is to determine if a proposition is TRUE, FALSE, or UNKNOWN in reality/common sense.

CRITICAL GUIDELINES:
- TRUE: The proposition is almost certainly true based on common sense, general knowledge, or obvious facts
- FALSE: The proposition is almost certainly false based on common sense, general knowledge, or obvious facts
- UNKNOWN: Cannot determine truth value without specific contextual information

Examples of TRUE propositions:
1. "The sky is blue" (generally true in common experience)
2. "Most people are not the Pope" (obvious statistical fact)
3. "Water is wet" (definitional truth)
4. "2+2=4" (mathematical truth)

Examples of FALSE propositions:
1. "The sky is green" (contradicts common experience)
2. "Most people are the Pope" (obviously false)
3. "Fire is cold" (contradicts common knowledge)

Examples of UNKNOWN propositions:
1. "Socrates is in Rome" (depends on specific context/time)
2. "The cat is on the mat" (depends on which cat, which mat, when)
3. "It will rain tomorrow" (depends on location, time)

IMPORTANT FOR CONTEXT-DEPENDENT PROPOSITIONS:
- For propositions about specific individuals (e.g., "You are the Pope"), use common sense
- Most people are NOT famous figures, kings, popes, etc.
- Assume ordinary circumstances unless otherwise indicated
- Use statistical likelihood: "You are the Pope" is almost certainly FALSE
- "You are in [specific city]" may be UNKNOWN without context

{format_instructions}""",
                ),
                (
                    "human",
                    """Proposition: {proposition}

Background context (if any): {context}

What is the truth value of this proposition in reality/common sense?""",
                ),
            ]
        )

        return prompt | self.llm

    def _parse_follows_result(self, content: str) -> FollowsFromResult:
        """Parse LLM response for follows_from check with fallback strategies."""
        try:
            # Try direct JSON parsing
            return self.follows_parser.parse(content)
        except Exception:
            # Fallback: regex extraction
            return self._fallback_parse_follows(content)

    def _parse_incompatible_result(self, content: str) -> IncompatibleWithResult:
        """Parse LLM response for incompatible_with check with fallback strategies."""
        try:
            # Try direct JSON parsing
            return self.incompatible_parser.parse(content)
        except Exception:
            # Fallback: regex extraction
            return self._fallback_parse_incompatible(content)

    def _parse_consistency_result(self, content: str) -> ConsistencyResult:
        """Parse LLM response for consistency check with fallback strategies."""
        try:
            # Try direct JSON parsing
            return self.consistency_parser.parse(content)
        except Exception:
            # Fallback: regex extraction
            return self._fallback_parse_consistency(content)

    def _fallback_parse_follows(self, content: str) -> FollowsFromResult:
        """Fallback parser for follows_from results."""
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return FollowsFromResult(**data)
            except Exception:
                pass

        # Keyword-based extraction
        follows = bool(
            re.search(r"\b(true|yes|follows|valid)\b", content, re.IGNORECASE)
        )
        if re.search(r"\b(false|no|does not follow|invalid)\b", content, re.IGNORECASE):
            follows = False

        # Extract inference type
        inference_type = "other"
        for itype in InferenceType:
            if itype.value.replace("_", " ") in content.lower():
                inference_type = itype.value
                break

        # Extract confidence
        confidence = 0.5
        conf_match = re.search(r"confidence[:\s]+([0-9.]+)", content, re.IGNORECASE)
        if conf_match:
            confidence = float(conf_match.group(1))

        return FollowsFromResult(
            follows=follows,
            inference_type=inference_type,
            reasoning=content[:500],  # First 500 chars as reasoning
            confidence=confidence,
        )

    def _fallback_parse_incompatible(self, content: str) -> IncompatibleWithResult:
        """Fallback parser for incompatible_with results."""
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return IncompatibleWithResult(**data)
            except Exception:
                pass

        # Keyword-based extraction
        incompatible = bool(
            re.search(
                r"\b(incompatible|contradiction|contradicts|inconsistent)\b",
                content,
                re.IGNORECASE,
            )
        )

        # Extract contradiction type
        contradiction_type = "unknown"
        if "direct" in content.lower():
            contradiction_type = "direct"
        elif "indirect" in content.lower():
            contradiction_type = "indirect"
        elif "implicit" in content.lower():
            contradiction_type = "implicit"

        # Extract confidence
        confidence = 0.5
        conf_match = re.search(r"confidence[:\s]+([0-9.]+)", content, re.IGNORECASE)
        if conf_match:
            confidence = float(conf_match.group(1))

        return IncompatibleWithResult(
            incompatible=incompatible,
            contradiction_type=contradiction_type,
            reasoning=content[:500],
            confidence=confidence,
        )

    def _fallback_parse_consistency(self, content: str) -> ConsistencyResult:
        """Fallback parser for consistency results."""
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return ConsistencyResult(**data)
            except Exception:
                pass

        # Keyword-based extraction
        consistent = not bool(
            re.search(
                r"\b(inconsistent|contradiction|contradicts)\b", content, re.IGNORECASE
            )
        )
        if re.search(r"\b(consistent|compatible)\b", content, re.IGNORECASE):
            consistent = True

        # Try to extract contradictions
        contradictions: list[str] = []
        # This is simplified - in practice, would need more sophisticated extraction

        return ConsistencyResult(
            consistent=consistent,
            contradictions=contradictions,
            reasoning=content[:500],
        )

    def _parse_equivalence_result(self, content: str) -> SemanticEquivalenceResult:
        """Parse LLM response for semantic equivalence check with fallback strategies."""
        try:
            # Try direct JSON parsing
            return self.equivalence_parser.parse(content)
        except Exception:
            # Fallback: regex extraction
            return self._fallback_parse_equivalence(content)

    def _fallback_parse_equivalence(self, content: str) -> SemanticEquivalenceResult:
        """Fallback parser for equivalence results."""
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return SemanticEquivalenceResult(**data)
            except Exception:
                pass

        # Keyword-based extraction
        equivalent = bool(
            re.search(
                r"\b(equivalent|same|identical|equal)\b",
                content,
                re.IGNORECASE,
            )
        )
        if re.search(
            r"\b(not equivalent|different|distinct)\b", content, re.IGNORECASE
        ):
            equivalent = False

        return SemanticEquivalenceResult(
            equivalent=equivalent,
            reasoning=content[:500],
        )

    def _parse_self_contradiction_result(self, content: str) -> SelfContradictionResult:
        """Parse LLM response for self-contradiction check with fallback strategies."""
        try:
            # Try direct JSON parsing
            return self.self_contradiction_parser.parse(content)
        except Exception:
            # Fallback: regex extraction
            return self._fallback_parse_self_contradiction(content)

    def _parse_truth_value_result(self, content: str) -> TruthValueResult:
        """Parse LLM response for truth value check with fallback strategies."""
        try:
            # Try direct JSON parsing
            return self.truth_value_parser.parse(content)
        except Exception:
            # Fallback: regex extraction
            return self._fallback_parse_truth_value(content)

    def _fallback_parse_self_contradiction(
        self, content: str
    ) -> SelfContradictionResult:
        """Fallback parser for self-contradiction results."""
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return SelfContradictionResult(**data)
            except Exception:
                pass

        # Keyword-based extraction
        self_contradictory = bool(
            re.search(
                r"\b(self-contradictory|contradictory|contradiction|mutually exclusive|impossible|cannot be true)\b",
                content,
                re.IGNORECASE,
            )
        )
        if re.search(
            r"\b(not self-contradictory|not contradictory|consistent|coherent|possible)\b",
            content,
            re.IGNORECASE,
        ):
            self_contradictory = False

        return SelfContradictionResult(
            self_contradictory=self_contradictory,
            reasoning=content[:500],
        )

    def _fallback_parse_truth_value(self, content: str) -> TruthValueResult:
        """Fallback parser for truth value results."""
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return TruthValueResult(**data)
            except Exception:
                pass

        # Keyword-based extraction
        content_lower = content.lower()

        # Check for truth value
        if re.search(r"\b(is true|true in reality|certainly true|obviously true)\b", content_lower):
            truth_value = "true"
            confidence = 0.8
        elif re.search(r"\b(is false|false in reality|certainly false|obviously false)\b", content_lower):
            truth_value = "false"
            confidence = 0.8
        else:
            truth_value = "unknown"
            confidence = 0.5

        return TruthValueResult(
            truth_value=truth_value,
            confidence=confidence,
            reasoning=content[:500],
        )
