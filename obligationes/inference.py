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
    ):
        """
        Initialize the inference engine.

        Args:
            llm: Pre-configured LLM instance (if None, creates default)
            model_name: Name of the model to use (gpt-4, claude-3-opus-20240229, etc.)
            temperature: Temperature for LLM calls (0.0 for deterministic)
            api_key: API key for the LLM provider
        """
        self.temperature = temperature

        if llm is not None:
            self.llm = llm
        else:
            # Auto-detect LLM type from model name
            if "gpt" in model_name.lower():
                kwargs = {"model": model_name, "temperature": temperature}
                if api_key is not None:
                    kwargs["api_key"] = api_key
                self.llm = ChatOpenAI(**kwargs)
            elif "claude" in model_name.lower():
                kwargs = {"model": model_name, "temperature": temperature}
                if api_key is not None:
                    kwargs["api_key"] = api_key
                self.llm = ChatAnthropic(**kwargs)
            else:
                # Default to GPT-4
                kwargs = {"model": "gpt-4", "temperature": temperature}
                if api_key is not None:
                    kwargs["api_key"] = api_key
                self.llm = ChatOpenAI(**kwargs)

        # Create output parsers
        self.follows_parser = PydanticOutputParser(pydantic_object=FollowsFromResult)
        self.incompatible_parser = PydanticOutputParser(
            pydantic_object=IncompatibleWithResult
        )
        self.consistency_parser = PydanticOutputParser(
            pydantic_object=ConsistencyResult
        )

        # Create chains
        self._follows_chain = self._create_follows_chain()
        self._incompatible_chain = self._create_incompatible_chain()
        self._consistency_chain = self._create_consistency_chain()

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

Check for:
1. Direct contradictions: "P" and "not P" in the set
2. Indirect contradictions: Derived through valid inference
3. Circular contradictions: A chain of inferences leading to contradiction

IMPORTANT:
- ALL propositions must be considered together
- A set is inconsistent if ANY contradiction exists
- Report ALL contradictory pairs found

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
