# Obligationes Disputation System: LLM-Based Implementation Guide

> **Implementation Status**: ✅ **COMPLETE** (All 8 phases implemented, 102 tests passing)
> **Last Updated**: October 2025

## Table of Contents
1. [Introduction](#1-introduction)
2. [Correspondence to Dutilh Novaes Formalization](#2-correspondence-to-dutilh-novaes-formalization)
3. [System Architecture](#3-system-architecture)
4. [Core Components](#4-core-components)
5. [Implementation Details](#5-implementation-details)
6. [API Reference](#6-api-reference)
7. [Usage Examples](#7-usage-examples)
8. [Testing and Validation](#8-testing-and-validation)
9. [Deployment Considerations](#9-deployment-considerations)
10. [Conclusion](#10-conclusion)

## 1. Introduction

### 1.1 Background

Obligationes were a form of structured dialectical disputation practiced in medieval universities, particularly during the 13th and 14th centuries. These highly formalized logical games served as training exercises for scholastic philosophers, testing their ability to maintain logical consistency while defending potentially false propositions.

This implementation follows Walter Burley's model of obligationes as formalized by Catarina Dutilh Novaes, who demonstrated that these medieval practices anticipated modern developments in formal logic, game theory, and discourse management.

### 1.2 Motivation

Traditional implementations of obligationes rely on symbolic logic engines that struggle with:
- Natural language complexity and ambiguity
- Contextual reasoning and implicit inference
- Multi-step logical derivations
- Recognition of subtle contradictions

By leveraging Large Language Models (LLMs), we can:
1. **Handle natural language** propositions without symbolic translation
2. **Perform sophisticated inference** including modal, counterfactual, and syllogistic reasoning
3. **Detect subtle contradictions** that require multiple inference steps
4. **Generate strategic moves** that exploit logical tensions
5. **Maintain formal rigor** while working with informal language

### 1.3 Design Philosophy

The architecture separates concerns into distinct layers:

```
┌─────────────────────────────────────┐
│         Game Management Layer        │
│  (Disputation flow, turn management) │
└─────────────────────────────────────┘
                    │
┌─────────────────────────────────────┐
│          Rules Engine Layer          │
│   (Burley's rules implementation)    │
└─────────────────────────────────────┘
                    │
┌─────────────────────────────────────┐
│        Logical Reasoning Layer       │
│    (LLM-based inference engine)      │
└─────────────────────────────────────┘
                    │
┌─────────────────────────────────────┐
│         State Management Layer       │
│   (Commitment tracking, history)     │
└─────────────────────────────────────┘
```

Each layer can be modified independently, allowing for:
- Different LLM backends (GPT-4, Claude, etc.)
- Alternative rule systems (Ockham, Swyneshed variations)
- Various game strategies
- Different persistence mechanisms

## 2. Correspondence to Dutilh Novaes Formalization

This section maps our implementation to the formal model presented in Catarina Dutilh Novaes' "Medieval Obligationes as Logical Games of Consistency Maintenance" (2005).

### 2.1 Core Formalization

Dutilh Novaes models obligationes as a game between two players with a response function **R** that maps propositions to {0, 1, ?}:

- **R(φ) = 1**: Accept the proposition (CONCEDO)
- **R(φ) = 0**: Reject the proposition (NEGO)
- **R(φ) = ?**: Neither accept nor reject (DUBITO)

**Implementation Mapping:**

```python
# state.py:15-24
class ResponseType(Enum):
    """Possible responses in obligationes disputation."""
    CONCEDO = "concedo"  # R(φ) = 1
    NEGO = "nego"        # R(φ) = 0
    DUBITO = "dubito"    # R(φ) = ?
```

### 2.2 Positum Acceptance Rule

**Novaes (p. 376)**: "R(φ₀) = 0 iff φ₀ ⊢⊥ (reject if and only if self-contradictory)"

The Respondent must reject a self-contradictory positum and accept a non-self-contradictory one.

**Implementation:**

```python
# manager.py:151-211
# Turn 0: Check if positum is self-contradictory (per Novaes 2005)
# R(φ₀) = 0 iff φ₀ ⊢⊥ (reject if self-contradictory)
# R(φ₀) = 1 iff φ₀ ⊬⊥ (accept if not self-contradictory)

positum_self_contradictory, contradiction_reasoning = (
    self.inference_engine.is_self_contradictory(positum)
)

if positum_self_contradictory:
    # Positum is self-contradictory - reject it
    # Respondent wins by correctly rejecting invalid positum
    return DisputationResult(
        winner="RESPONDENT",
        reason="Positum was self-contradictory and correctly rejected",
        ...
    )

# Positum is not self-contradictory - accept it
self.state.set_positum(positum)
self.state.add_response(
    positum,
    ResponseType.CONCEDO,
    "The positum is not self-contradictory and is accepted by obligation.",
    0,  # Rule 0 = positum acceptance
)
```

### 2.3 Commitment Set Updates

**Novaes (p. 376)**: The commitment set Γₙ is updated based on responses:

- **If R(φ) = 1 (CONCEDO)**: Γₙ = Γₙ₋₁ ∪ {φ}
- **If R(φ) = 0 (NEGO)**: Γₙ = Γₙ₋₁ ∪ {¬φ}
- **If R(φ) = ? (DUBITO)**: Γₙ = Γₙ₋₁

**Critical insight**: When NEGO is used, the Respondent commits to the **negation** of the proposition, not just to not accepting it.

**Implementation:**

```python
# state.py:192-234
def add_response(
    self,
    proposition: str,
    response: ResponseType,
    reasoning: str,
    rule_applied: int,
) -> None:
    """Record a response and update state accordingly."""
    prop = Proposition(
        content=proposition,
        status=self._response_to_status(response),
        turn_introduced=self.turn_count,
    )

    # Add to appropriate set
    if response == ResponseType.CONCEDO:
        self.concessa.add(prop)  # Γₙ = Γₙ₋₁ ∪ {φ}
    elif response == ResponseType.NEGO:
        self.negata.add(prop)    # Γₙ = Γₙ₋₁ ∪ {¬φ} - stored as negata
    elif response == ResponseType.DUBITO:
        self.dubitata.add(prop)  # Γₙ = Γₙ₋₁
```

```python
# state.py:236-256
def get_all_commitments(self) -> Set[str]:
    """Get all propositions the Respondent is committed to (positum + concessa)."""
    commitments = set()
    if self.positum:
        commitments.add(self.positum.content)
    commitments.update(p.content for p in self.concessa)
    return commitments

def get_all_negations(self) -> Set[str]:
    """Get all propositions the Respondent has denied (negata = ¬φ for each NEGO)."""
    return {p.content for p in self.negata}
```

### 2.4 Consistency Checking with Negations

**Novaes formalization**: The consistency check must account for both positive commitments and negated commitments from NEGO responses.

**Implementation:**

```python
# manager.py:248-273
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
        # Mark in history
        self.state.history[-1].consistency_maintained = False
        break
    else:
        self.state.history[-1].consistency_maintained = True
```

The LLM-based consistency checker explicitly handles NOT() notation:

```python
# inference.py:105-125 (consistency chain system prompt)
"""You are an expert in formal logic and consistency analysis.

NOTATION:
- "NOT(P)" means the negation of proposition P
- If you see both "P" and "NOT(P)", that's a direct contradiction
- "NOT(A and B)" is logically equivalent to "NOT(A) or NOT(B)" (De Morgan's law)
- "NOT(A or B)" is logically equivalent to "NOT(A) and NOT(B)" (De Morgan's law)

Check for:
1. Direct contradictions: "P" and "NOT(P)" in the set
2. Indirect contradictions: Derived through valid inference (modus ponens, etc.)
3. De Morgan contradictions: e.g., "NOT(A and B)" with "NOT(A or B)"
4. Circular contradictions: A chain of inferences leading to contradiction

Pay special attention to De Morgan's laws when analyzing NOT() propositions
"""
```

### 2.5 Common Knowledge (Kc)

**Novaes (p. 376)**: "Kc is the common state of knowledge of those present at the disputation. It is an incomplete model, in the sense that some propositions do not receive a truth-value."

Kc includes:
- Common sense knowledge
- Religious dogmas
- Circumstantially available information
- All participants agree on Kc

**Implementation:**

```python
# manager.py:52-62
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

# manager.py:86-102
def __init__(
    self,
    common_knowledge: Optional[Set[str]] = None,
    config: Optional[DisputationConfig] = None,
    inference_engine: Optional[LLMInferenceEngine] = None,
):
    """Initialize the disputation manager.

    Args:
        common_knowledge: Background facts both parties accept (uses default if None)
    """
    # Use provided or default common knowledge
    if common_knowledge is None:
        common_knowledge = DEFAULT_COMMON_KNOWLEDGE.copy()

    # Initialize state
    self.state = ObligationesState(common_knowledge=common_knowledge)
```

Common knowledge is used in Burley's Rules 3-5:

```python
# rules.py (conceptual - actual implementation uses LLM prompts)
# Rule 3: If φ follows from (positum + concessa + Kc) → CONCEDO
# Rule 4: If φ incompatible with (positum + concessa + Kc) → NEGO
# Rule 5: Otherwise, respond based on truth value in Kc
```

### 2.6 Burley's Rules as Response Function

**Novaes (p. 376-377)**: Burley's rules define the response function R in order of precedence.

**Implementation:**

```python
# rules.py:40-115 (BurleyRulesEngine.evaluate_proposition)
def evaluate_proposition(
    self, proposition: str, state: ObligationesState
) -> Tuple[ResponseType, str, int]:
    """
    Apply Burley's rules in strict order:

    1. If proposition follows from (positum + concessa) → CONCEDO
    2. If proposition incompatible with (positum + concessa) → NEGO
    3. If proposition follows from (positum + concessa + Kc) → CONCEDO
    4. If proposition incompatible with (positum + concessa + Kc) → NEGO
    5. Otherwise, respond based on truth in Kc

    Returns: (ResponseType, reasoning, rule_number)
    """

    # Get current commitments
    commitments = state.get_all_commitments()
    common_knowledge = state.common_knowledge

    # RULE 1: Check if follows from commitments alone
    if commitments:
        follows, reasoning = self.inference_engine.follows_from(
            proposition, commitments
        )
        if follows:
            return (ResponseType.CONCEDO, f"Rule 1: {reasoning}", 1)

    # RULE 2: Check if incompatible with commitments
    if commitments:
        incompatible, reasoning = self.inference_engine.incompatible_with(
            proposition, commitments
        )
        if incompatible:
            return (ResponseType.NEGO, f"Rule 2: {reasoning}", 2)

    # RULE 3: Check if follows from commitments + common knowledge
    if commitments and common_knowledge:
        combined = commitments.union(common_knowledge)
        follows, reasoning = self.inference_engine.follows_from(
            proposition, combined
        )
        if follows:
            return (ResponseType.CONCEDO, f"Rule 3: {reasoning}", 3)

    # RULE 4: Check if incompatible with commitments + common knowledge
    if commitments and common_knowledge:
        combined = commitments.union(common_knowledge)
        incompatible, reasoning = self.inference_engine.incompatible_with(
            proposition, combined
        )
        if incompatible:
            return (ResponseType.NEGO, f"Rule 4: {reasoning}", 4)

    # RULE 5: Respond based on truth in common knowledge
    is_true, reasoning = self.inference_engine.is_known(
        proposition, common_knowledge
    )
    if is_true:
        return (ResponseType.CONCEDO, f"Rule 5: {reasoning}", 5)
    else:
        return (ResponseType.NEGO, f"Rule 5: {reasoning}", 5)
```

### 2.7 Winner Determination

**Novaes**: The game ends when the Respondent is forced into contradiction (Γₙ ⊢⊥). The Opponent wins if a contradiction is reached, the Respondent wins if consistency is maintained.

**Implementation:**

```python
# manager.py:279-287
# Determine outcome
if contradiction_found:
    winner = "OPPONENT"
    reason = "Respondent fell into contradiction"
else:
    winner = "RESPONDENT"
    reason = "Maintained consistency through all turns"

self.state.end_disputation(winner, reason)
```

### 2.8 Self-Contradiction Detection

**Novaes**: A single proposition φ is self-contradictory if φ ⊢⊥ (it entails contradiction on its own).

**Implementation:**

```python
# inference.py:217-241
class SelfContradictionResult(BaseModel):
    """Result of checking if a single proposition is self-contradictory."""
    self_contradictory: bool = Field(
        description="Whether the proposition is internally self-contradictory"
    )
    reasoning: str = Field(
        description="Explanation of why it is or is not self-contradictory"
    )

def is_self_contradictory(self, proposition: str) -> Tuple[bool, str]:
    """
    Check if a single proposition is internally self-contradictory.

    A proposition is self-contradictory if it contains mutually exclusive claims:
    - "Socrates is both mortal and immortal"
    - "X is true and X is false"
    - "The square circle exists"

    This implements Novaes' positum rejection rule: R(φ₀) = 0 iff φ₀ ⊢⊥
    """
    result = self._self_contradiction_chain.invoke({
        "proposition": proposition,
        "format_instructions": self.self_contradiction_parser.get_format_instructions(),
    })
    content = result.content if hasattr(result, "content") else str(result)
    parsed = self._parse_self_contradiction_result(content)
    return parsed.self_contradictory, parsed.reasoning
```

### 2.9 Summary of Correspondence

| Novaes Formalization | Implementation |
|---------------------|----------------|
| R(φ) = 1 | `ResponseType.CONCEDO` |
| R(φ) = 0 | `ResponseType.NEGO` |
| R(φ) = ? | `ResponseType.DUBITO` |
| φ₀ (positum) | `state.positum` |
| Γₙ (commitment set) | `state.get_all_commitments() ∪ NOT(state.get_all_negations())` |
| Kc (common knowledge) | `state.common_knowledge` / `DEFAULT_COMMON_KNOWLEDGE` |
| R(φ₀) = 0 iff φ₀ ⊢⊥ | `is_self_contradictory()` check in Turn 0 |
| Γₙ = Γₙ₋₁ ∪ {φ} when R(φ) = 1 | `state.concessa.add(prop)` |
| Γₙ = Γₙ₋₁ ∪ {¬φ} when R(φ) = 0 | `state.negata.add(prop)` + consistency check with `NOT()` |
| Burley's Rules 1-5 | `BurleyRulesEngine.evaluate_proposition()` |
| Consistency check (Γₙ ⊬⊥) | `inference_engine.check_consistency()` with De Morgan handling |
| Opponent wins if Γₙ ⊢⊥ | `contradiction_found = True` → `winner = "OPPONENT"` |
| Respondent wins if Γₙ ⊬⊥ | No contradiction after max turns → `winner = "RESPONDENT"` |

### 2.10 Key Implementation Differences

While the implementation follows Novaes' formalization closely, there are some differences:

1. **LLM-based Logic**: Novaes uses formal logical consequence (⊢), we use LLM-based inference with natural language
2. **Strategic Planning**: Our Opponent uses multi-turn planning not discussed in Novaes
3. **Trap Detection**: Respondent can detect traps (for informational purposes) but cannot avoid them
4. **Explicit NOT() Notation**: We use `NOT(P)` string notation to track negations in consistency checking
5. **De Morgan Laws**: LLM explicitly handles De Morgan transformations in consistency checking
6. **Turn 0**: We formalize positum acceptance as "Turn 0" with explicit rule application

These differences adapt the medieval formal system to modern LLM capabilities while preserving the logical structure and game-theoretic properties described by Dutilh Novaes.

## 3. System Architecture

### 3.1 High-Level Architecture

```python
# Core system components and their relationships

ObligationesSystem
├── DisputationManager (orchestrates the game)
│   ├── OpponentAgent (proposes propositions)
│   └── RespondentAgent (evaluates and responds)
├── LogicalReasoningEngine (LLM-based inference)
│   ├── InferenceChain (checks logical consequence)
│   ├── ConsistencyChain (detects contradictions)
│   └── CompatibilityChain (checks incompatibilities)
├── RulesEngine (implements Burley's rules)
│   └── BurleyEvaluator (applies rules in order)
├── StateManager (tracks disputation state)
│   ├── CommitmentStore (tracks concessa/negata)
│   ├── PropositionHistory (maintains transcript)
│   └── ConsistencyMonitor (ongoing validation)
└── LangChainIntegration
    ├── CustomMemory (dialogue state persistence)
    ├── PromptTemplates (structured LLM prompts)
    └── OutputParsers (JSON response parsing)
```

### 3.2 Data Flow

1. **Initialization Phase**
   ```
   User → DisputationManager → StateManager
                           → Initialize common knowledge
                           → Set game parameters
   ```

2. **Positum Phase**
   ```
   Opponent → Propose positum → StateManager → Record
                                            → Set as foundation
   ```

3. **Disputation Loop**
   ```
   Opponent → Generate proposition → LogicalReasoningEngine → Strategic analysis
                                                           → Trap detection
           ↓
   Respondent → Evaluate via RulesEngine → Apply Burley's rules
                                        → Check consistency
                                        → Generate response
           ↓
   StateManager → Update commitments → Check for contradiction
                                   → Record in history
           ↓
   ConsistencyMonitor → Validate state → Continue or end game
   ```

4. **Outcome Determination Phase**
   ```
   DisputationManager → Check for contradiction → Determine winner mechanically
                                                → Record final state
   ```

### 3.3 Key Design Decisions

#### 3.3.1 LLM-Centric Logic
**Decision**: Use LLMs for all logical reasoning rather than symbolic engines.

**Rationale**:
- Natural language handling without lossy translation
- Sophisticated contextual reasoning
- Ability to recognize implicit inferences
- Flexibility with ambiguous propositions

**Trade-offs**:
- Non-deterministic (same input may yield different evaluations)
- Requires careful prompt engineering
- Higher computational cost
- Potential for hallucination

**Mitigation**:
- Temperature=0 for consistency
- Structured output formats (JSON)
- Multiple validation passes
- Explicit reasoning chains

#### 3.3.2 Separation of Roles
**Decision**: Implement Opponent and Respondent as separate agents with mechanical winner determination.

**Rationale**:
- Clear separation of concerns between strategic and rule-based agents
- Different prompting strategies per role (strategic vs. rule-following)
- Opponent focuses on finding contradictions, Respondent on rule compliance
- Winner determination is mechanical (no judgment needed)
- Easier testing and validation of each component independently

#### 3.3.3 State Immutability
**Decision**: Treat state updates as immutable transitions.

**Rationale**:
- Enables state replay and debugging
- Supports "what-if" analysis
- Prevents accidental state corruption
- Facilitates undo/redo functionality

## 4. Core Components

### 4.1 State Management

#### 4.1.1 Data Structures

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Set, List, Optional

class ResponseType(Enum):
    """Possible responses in obligationes"""
    CONCEDO = "concedo"    # Grant/accept
    NEGO = "nego"          # Deny
    DUBITO = "dubito"      # Doubt (neither grant nor deny)

class PropositionStatus(Enum):
    """Status of propositions in the disputation"""
    POSITUM = "positum"         # Initial position
    CONCESSA = "concessa"       # Granted
    NEGATA = "negata"          # Denied
    DUBITATA = "dubitata"      # Doubted
    IRRELEVANT = "irrelevant"  # Not yet addressed

@dataclass
class Proposition:
    """Represents a proposition in the disputation"""
    content: str                    # Natural language proposition
    status: PropositionStatus       # Current status
    turn_introduced: int            # When it was introduced
    follows_from: List[str] = field(default_factory=list)
    incompatible_with: List[str] = field(default_factory=list)
    inference_chain: Optional[str] = None  # How it was derived
```

#### 4.1.2 State Container

```python
class ObligationesState:
    """
    Maintains complete disputation state.
    This is the single source of truth for the game.
    """
    
    def __init__(self):
        self.positum: Optional[Proposition] = None
        self.concessa: Set[Proposition] = set()
        self.negata: Set[Proposition] = set()
        self.dubitata: Set[Proposition] = set()
        self.turn_count: int = 0
        self.disputation_active: bool = False
        self.history: List[Tuple[str, ResponseType, Dict]] = []
        
    def get_all_commitments(self) -> Set[str]:
        """Returns all propositions the Respondent is committed to"""
        commitments = set()
        if self.positum:
            commitments.add(self.positum.content)
        commitments.update(p.content for p in self.concessa)
        return commitments
```

### 4.2 Logical Reasoning Engine

#### 4.2.1 LLM Configuration

```python
class LLMInferenceEngine:
    """
    Handles all logical reasoning via LLM calls.
    Critical component - all logical decisions flow through here.
    """
    
    def __init__(self, llm=None):
        # Default to GPT-4 with temperature=0 for consistency
        self.llm = llm or ChatOpenAI(
            temperature=0,
            model="gpt-4",
            request_timeout=30,
            max_retries=3
        )
```

#### 4.2.2 Inference Chain

The inference chain determines if a proposition follows necessarily from premises:

```python
def _create_follows_chain(self) -> LLMChain:
    """
    Critical prompt for logical inference.
    Must be precise about what counts as "following from."
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a medieval logician expert.

TASK: Determine if a proposition NECESSARILY FOLLOWS from premises.

"Necessarily follows" means:
- The conclusion MUST be true if the premises are true
- No possible interpretation makes premises true and conclusion false
- The inference is DEDUCTIVELY VALID

Consider these inference patterns:
1. Modus Ponens: P→Q, P ⊢ Q
2. Modus Tollens: P→Q, ¬Q ⊢ ¬P
3. Syllogisms: All A are B, X is A ⊢ X is B
4. Conjunction: P, Q ⊢ P∧Q
5. Disjunction: P ⊢ P∨Q

Output JSON:
{
    "follows": true/false,
    "inference_type": "modus_ponens/syllogism/etc",
    "reasoning": "step by step derivation",
    "confidence": 0.0-1.0
}"""),
        ("human", "Premises:\n{premises}\n\nProposition:\n{proposition}")
    ])
```

### 4.3 Rules Engine

#### 4.3.1 Burley's Rules Implementation

```python
class BurleyRulesEngine:
    """
    Implements Walter Burley's evaluation rules.
    Rules must be applied in strict order of precedence.
    """

    def evaluate_proposition(self,
                            proposition: str,
                            state: ObligationesState) -> Tuple[ResponseType, str, int]:
        """
        Apply Burley's rules in order:

        1. If prop follows from (positum + concessa) → CONCEDO
        2. If prop incompatible with (positum + concessa) → NEGO
        3. If prop follows from (positum + concessa + common_knowledge) → CONCEDO
        4. If prop incompatible with (positum + concessa + common_knowledge) → NEGO
        5. Otherwise, respond based on truth in common knowledge

        Returns: (response_type, reasoning, rule_number_applied)

        Note: common_knowledge is stored in state.common_knowledge
        """
```

**Implementation Note**: The order of rules is crucial. Each rule must be evaluated completely before moving to the next. The LLM prompt must explicitly enforce this ordering.

### 4.4 Agent Implementations

#### 4.4.1 Respondent Agent

```python
class RespondentAgent:
    """
    The Respondent must follow rules strictly, even if it leads to contradiction.
    This models the medieval obligation to maintain logical discipline.
    """

    def evaluate_proposition(self, proposition: str, state: ObligationesState) -> Dict[str, Any]:
        # Step 1: Determine what rules require
        response, reasoning, rule_applied = self.rules_engine.evaluate_proposition(
            proposition, state
        )

        # Step 2: Check for traps (but cannot avoid them)
        trap_detected, trap_analysis = self._detect_trap(proposition, state, response)

        # Step 3: Respond according to rules (not strategy)
        # CRITICAL: Cannot deviate from rules even to avoid contradiction

        return {
            "response": response,
            "reasoning": reasoning,
            "rule_applied": rule_applied,
            "trap_detected": trap_detected,
            "trap_analysis": trap_analysis,
        }
```

#### 4.4.2 Opponent Agent

```python
class OpponentAgent:
    """
    The Opponent seeks to force contradiction through strategic proposition selection.
    Uses multi-step planning and trap sequences.
    """

    def propose_proposition(self, state: ObligationesState) -> Dict[str, str]:
        # Analyze current vulnerabilities using LLM
        # Plan multi-step trap sequences based on strategy
        # Select optimal next move

        return {
            "proposition": "All men are mortal",
            "strategy_note": "Exploit tension with positum",
            "expected_response": "NEGO"
        }
```

## 5. Implementation Details

### 5.1 LangChain Integration

#### 5.1.1 Custom Memory Class

```python
from langchain.memory import BaseMemory

class ObligationesMemory(BaseMemory):
    """
    Custom LangChain memory that maintains disputation state.
    Integrates with LangChain's conversation management.
    """
    
    @property
    def memory_variables(self) -> List[str]:
        """Variables exposed to prompts"""
        return [
            "disputation_state",    # Current game state
            "current_obligations",   # What must be done
            "consistency_status",    # Is state consistent?
            "required_response"      # What rules dictate
        ]
    
    def save_context(self, inputs: Dict, outputs: Dict) -> None:
        """
        Update state after each exchange.
        This is called automatically by LangChain after each interaction.
        """
        # Parse the response
        response = self._parse_response(outputs["output"])
        
        # Update commitments based on response
        self._update_commitments(inputs["proposition"], response)
        
        # Check consistency
        self._check_consistency()
        
        # Increment turn counter
        self.state.turn_count += 1
```

#### 5.1.2 Prompt Engineering

Prompts must be carefully structured for consistency:

```python
def create_respondent_prompt() -> ChatPromptTemplate:
    """
    Respondent prompt emphasizes rule-following over strategy.
    """
    return ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a medieval Respondent in obligationes.

CRITICAL RULES:
1. You MUST follow Burley's rules exactly
2. You CANNOT deviate even to avoid contradiction
3. You can only respond: CONCEDO, NEGO, or DUBITO

Your commitment: Defend the positum while maintaining consistency.
If the rules force you into contradiction, you lose - but you must still follow the rules."""),
        
        ("system", "{disputation_state}"),
        ("system", "Required by rules: {required_response}"),
        ("human", "Evaluate: {proposition}"),
    ])
```

### 5.2 Error Handling

#### 5.2.1 LLM Response Parsing

```python
def parse_llm_response(response: str) -> Dict:
    """
    Robust parsing with multiple fallback strategies.
    """
    # Try JSON parsing first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try to extract structured data
    patterns = {
        'response': r'(?:response|answer)[:\s]*([A-Z]+)',
        'reasoning': r'(?:reasoning|because)[:\s]*(.+?)(?:\n|$)',
        'follows': r'(?:follows|consequence)[:\s]*(true|false)'
    }
    
    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            result[key] = match.group(1)
    
    # Final fallback: keyword detection
    if not result.get('response'):
        if 'CONCEDO' in response.upper():
            result['response'] = 'CONCEDO'
        elif 'NEGO' in response.upper():
            result['response'] = 'NEGO'
        else:
            result['response'] = 'DUBITO'
    
    return result
```

#### 5.2.2 Consistency Validation

```python
def validate_response_consistency(response: ResponseType, 
                                 required: ResponseType,
                                 state: ObligationesState) -> Dict:
    """
    Ensures the system follows its own rules.
    """
    if response != required:
        return {
            "valid": False,
            "error": f"Response {response} violates rules (should be {required})",
            "severity": "CRITICAL"
        }
    
    # Simulate the response
    new_state = simulate_response(state, response)
    
    # Check for immediate contradiction
    consistency = check_consistency(new_state)
    
    return {
        "valid": True,
        "creates_contradiction": not consistency["consistent"],
        "warning": "Response follows rules but creates contradiction" 
                  if not consistency["consistent"] else None
    }
```

### 5.3 Performance Optimization

#### 5.3.1 Caching Strategy

```python
from functools import lru_cache
from hashlib import md5

class CachedInferenceEngine:
    """
    Cache LLM calls for identical logical queries.
    """
    
    @lru_cache(maxsize=1000)
    def _cached_inference(self, premises_hash: str, proposition: str) -> Dict:
        """
        Cache based on hash of premises + proposition.
        """
        return self._run_inference_chain(premises_hash, proposition)
    
    def follows_from(self, proposition: str, premises: Set[str]) -> bool:
        # Create deterministic hash of premises
        premises_sorted = sorted(premises)
        premises_str = "|".join(premises_sorted)
        premises_hash = md5(premises_str.encode()).hexdigest()
        
        # Use cached result if available
        result = self._cached_inference(premises_hash, proposition)
        return result["follows"]
```

#### 5.3.2 Batch Processing

```python
async def evaluate_propositions_batch(propositions: List[str], 
                                     state: ObligationesState) -> List[Dict]:
    """
    Evaluate multiple propositions in parallel.
    """
    import asyncio
    
    tasks = [
        evaluate_proposition_async(prop, state) 
        for prop in propositions
    ]
    
    results = await asyncio.gather(*tasks)
    return results
```

## 6. API Reference

### 6.1 Main Interface

```python
class DisputationManager:
    """Main entry point for obligationes disputations."""

    def __init__(self,
                 common_knowledge: Optional[Set[str]] = None,
                 config: Optional[DisputationConfig] = None):
        """
        Initialize disputation system.

        Args:
            common_knowledge: Background facts both parties accept
            config: Configuration for LLM settings and strategies
        """

    def run_disputation(self, positum: str) -> DisputationResult:
        """
        Run a complete disputation.

        Args:
            positum: Initial position to defend

        Returns:
            DisputationResult containing transcript, winner, and analysis

        Note: max_turns, verbose, and other settings are in config
        """

    def save_transcript(self, filepath: str) -> None:
        """Save disputation transcript to JSON file."""

    @staticmethod
    def from_transcript(filepath: str) -> "DisputationManager":
        """Load a disputation from a saved transcript."""
```

### 6.2 Result Types

```python
@dataclass
class DisputationResult:
    """Complete disputation outcome."""
    winner: str                          # "OPPONENT" or "RESPONDENT"
    reason: str                          # Why they won
    positum: str                         # The initial position
    total_turns: int                     # Number of turns completed
    final_consistent: bool               # Final consistency state
    transcript: List[Dict[str, Any]]     # Complete exchange history
    judgment: Dict[str, Any]             # Basic outcome summary
    state: ObligationesState             # Final game state
    started_at: str                      # ISO timestamp
    ended_at: str                        # ISO timestamp
    duration_seconds: float              # Duration

@dataclass
class Turn:
    """Single exchange in disputation (stored in state.history)."""
    number: int
    proposition: str
    response: ResponseType
    reasoning: str
    consistency_maintained: bool
    rule_applied: int
```

### 6.3 Configuration

```python
@dataclass
class DisputationConfig:
    """Configuration for disputation behavior."""

    max_turns: int = 10
    opponent_strategy: OpponentStrategy = OpponentStrategy.BALANCED
    verbose: bool = True
    model_name: str = "gpt-4"
    temperature: float = 0.0
```

## 7. Usage Examples

### 7.1 Basic Usage (Implemented)

```python
from obligationes.manager import DisputationManager, DisputationConfig, create_disputation
from obligationes.agents import OpponentStrategy

# Simple usage with convenience function
result = create_disputation(
    positum="Socrates is immortal",
    max_turns=8,
    verbose=True
)

# Check outcome
print(f"Winner: {result.winner}")
print(f"Reason: {result.reason}")

# Analyze specific turns
for turn in result.transcript:
    if not turn.get('consistency_maintained', True):
        print(f"Contradiction at turn {turn['turn']}: {turn['proposition']}")
```

### 7.2 Custom Configuration (Implemented)

```python
from obligationes.manager import DisputationManager, DisputationConfig
from obligationes.agents import OpponentStrategy

# Medieval common knowledge
common_knowledge = {
    "God is omnipotent",
    "All men are mortal",
    "The Earth is at the center of the universe",
    "Heavy objects fall faster than light ones",
    "Fire rises naturally",
    "The soul is immortal"
}

# Custom configuration
config = DisputationConfig(
    max_turns=10,
    opponent_strategy=OpponentStrategy.AGGRESSIVE,
    verbose=True,
    model_name="gpt-4",
    temperature=0.0
)

# Initialize with custom settings
manager = DisputationManager(
    common_knowledge=common_knowledge,
    config=config
)

# Run with specific positum
result = manager.run_disputation(positum="The Pope is not in Rome")
```

### 7.3 Interactive Mode

```python
class InteractiveDisputation:
    """Allow human to play as Respondent."""
    
    def __init__(self):
        self.manager = LLMDisputationManager()
        self.state = ObligationesState()
    
    def play(self):
        # Setup
        positum = input("Enter positum: ")
        self.state.positum = Proposition(positum, PropositionStatus.POSITUM, 0)
        
        print(f"\nYou must defend: {positum}")
        print("Respond with CONCEDO, NEGO, or DUBITO only.\n")
        
        # Game loop
        while self.state.turn_count < 10:
            # Opponent proposes
            proposition = self.manager.opponent.propose_proposition(self.state)
            print(f"Opponent: {proposition}")
            
            # Get human response
            response = input("Your response: ").upper()
            
            # Validate response
            required = self.manager.rules.evaluate_proposition(
                proposition, 
                self.state,
                self.manager.common_knowledge
            )
            
            if response != required[0].value:
                print(f"ERROR: Rules require {required[0].value}")
                print(f"Reason: {required[1]}")
            
            # Update state
            self.update_state(proposition, response)
            
            # Check consistency
            if not self.check_consistency():
                print("CONTRADICTION! You lose.")
                break
        else:
            print("Time's up! You maintained consistency and win!")
```

### 7.4 Analysis Tools

```python
class DisputationAnalyzer:
    """Analyze disputation patterns and strategies."""
    
    def analyze_transcript(self, result: DisputationResult) -> Analysis:
        """Generate detailed analysis of disputation."""
        
        return Analysis(
            total_moves=len(result.transcript),
            contradiction_points=self.find_contradictions(result),
            trap_sequences=self.identify_traps(result),
            rule_distribution=self.count_rule_applications(result),
            logical_depth=self.measure_inference_chains(result)
        )
    
    def compare_strategies(self, 
                          positum: str,
                          strategies: List[str],
                          trials: int = 10) -> StrategyComparison:
        """Compare effectiveness of different opponent strategies."""
        
        results = {}
        for strategy in strategies:
            wins = 0
            total_turns = 0
            
            for _ in range(trials):
                manager = LLMDisputationManager(strategy=strategy)
                result = manager.run_disputation(positum)
                
                if result.winner == "OPPONENT":
                    wins += 1
                total_turns += len(result.transcript)
            
            results[strategy] = {
                "win_rate": wins / trials,
                "avg_turns": total_turns / trials
            }
        
        return StrategyComparison(results)
```

## 8. Testing and Validation

### 8.1 Unit Tests

```python
import pytest
from unittest.mock import Mock, patch

class TestInferenceEngine:
    """Test logical inference capabilities."""
    
    def test_modus_ponens(self):
        """Test basic modus ponens inference."""
        engine = LLMInferenceEngine()
        
        premises = {
            "If Socrates is a man, then Socrates is mortal",
            "Socrates is a man"
        }
        conclusion = "Socrates is mortal"
        
        result = engine.follows_from(conclusion, premises)
        assert result[0] == True
        assert "modus ponens" in result[1].lower()
    
    def test_contradiction_detection(self):
        """Test detection of direct contradictions."""
        engine = LLMInferenceEngine()
        
        propositions = {
            "Socrates is mortal",
            "Socrates is not mortal"
        }
        
        consistent, contradictions = engine.check_consistency(propositions)
        assert consistent == False
        assert len(contradictions) > 0
    
    @patch('langchain.chat_models.ChatOpenAI')
    def test_llm_failure_handling(self, mock_llm):
        """Test graceful handling of LLM failures."""
        mock_llm.return_value.run.side_effect = Exception("API Error")
        
        engine = LLMInferenceEngine(mock_llm)
        result = engine.follows_from("P", {"Q"})
        
        assert result[0] == False  # Safe default
        assert "error" in result[1].lower()
```

### 8.2 Integration Tests

```python
class TestDisputation:
    """Test complete disputation flows."""
    
    def test_basic_disputation(self):
        """Test a simple disputation scenario."""
        manager = LLMDisputationManager()
        
        result = manager.run_disputation(
            positum="All birds can fly",
            max_turns=5
        )
        
        assert result.winner in ["OPPONENT", "RESPONDENT"]
        assert len(result.transcript) <= 5
        assert result.final_consistent in [True, False]
    
    def test_contradiction_detection(self):
        """Test that contradictions end the game."""
        manager = LLMDisputationManager()
        
        # Force a contradiction scenario
        manager.state.positum = Proposition("P", PropositionStatus.POSITUM, 0)
        manager.state.concessa.add(Proposition("not P", PropositionStatus.CONCESSA, 1))
        
        consistency = manager.consistency_monitor.check_consistency(manager.state)
        assert consistency["overall_consistent"] == False
    
    def test_rule_precedence(self):
        """Test that rules are applied in correct order."""
        rules = LLMBurleyRules()
        state = ObligationesState()
        state.positum = Proposition("Socrates is not a man", PropositionStatus.POSITUM, 0)
        
        # Should follow from positum (Rule 1)
        response, reasoning, rule_num = rules.evaluate_proposition(
            "Socrates is not a man",
            state,
            set()
        )
        
        assert response == ResponseType.CONCEDO
        assert rule_num == 1
```

### 8.3 Validation Suite

```python
class ValidationSuite:
    """Validate system behavior against formal specifications."""
    
    def validate_burley_compliance(self, num_trials: int = 100) -> ValidationReport:
        """Ensure system follows Burley's rules correctly."""
        
        violations = []
        
        for _ in range(num_trials):
            # Generate random scenario
            scenario = self.generate_scenario()
            
            # Get system response
            response = self.evaluate_scenario(scenario)
            
            # Check against formal rules
            expected = self.calculate_expected_response(scenario)
            
            if response != expected:
                violations.append({
                    "scenario": scenario,
                    "expected": expected,
                    "actual": response
                })
        
        return ValidationReport(
            total_trials=num_trials,
            violations=violations,
            compliance_rate=1 - (len(violations) / num_trials)
        )
```

## 9. Deployment Considerations

### 9.1 Production Configuration

```python
# production_config.py

import os
from dotenv import load_dotenv

load_dotenv()

class ProductionConfig:
    # API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE = 60
    MAX_TOKENS_PER_REQUEST = 2000
    
    # Caching
    REDIS_URL = os.getenv("REDIS_URL")
    CACHE_TTL = 3600  # 1 hour
    
    # Monitoring
    SENTRY_DSN = os.getenv("SENTRY_DSN")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Game Settings
    MAX_DISPUTATION_TURNS = 20
    DEFAULT_TIMEOUT = 300  # 5 minutes
    
    # Safety
    CONTENT_FILTER = True
    MAX_PROPOSITION_LENGTH = 500
```

### 9.2 API Deployment

```python
# api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Obligationes API")

class DisputationRequest(BaseModel):
    positum: str
    max_turns: int = 10
    strategy: str = "balanced"

class DisputationResponse(BaseModel):
    winner: str
    reason: str
    transcript: List[Dict]
    analysis: Dict

@app.post("/disputation", response_model=DisputationResponse)
async def run_disputation(request: DisputationRequest):
    """Run a complete disputation."""
    
    try:
        manager = LLMDisputationManager(strategy=request.strategy)
        result = manager.run_disputation(
            positum=request.positum,
            max_turns=request.max_turns
        )
        
        return DisputationResponse(
            winner=result.winner,
            reason=result.reason,
            transcript=[turn.dict() for turn in result.transcript],
            analysis={
                "winner": result.winner,
                "reason": result.reason,
                "final_consistent": result.final_consistent
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 9.3 Docker Deployment

```dockerfile
# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml

version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

### 9.4 Monitoring and Logging

```python
import logging
from datetime import datetime
import json

class DisputationLogger:
    """Structured logging for disputations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_disputation_start(self, positum: str, config: Dict):
        """Log disputation initialization."""
        self.logger.info(json.dumps({
            "event": "disputation_start",
            "timestamp": datetime.utcnow().isoformat(),
            "positum": positum,
            "config": config
        }))
    
    def log_turn(self, turn_num: int, proposition: str, response: str):
        """Log each turn of the disputation."""
        self.logger.info(json.dumps({
            "event": "turn",
            "turn_number": turn_num,
            "proposition": proposition,
            "response": response,
            "timestamp": datetime.utcnow().isoformat()
        }))
    
    def log_contradiction(self, turn_num: int, details: Dict):
        """Log contradiction detection."""
        self.logger.warning(json.dumps({
            "event": "contradiction",
            "turn_number": turn_num,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }))
    
    def log_disputation_end(self, winner: str, reason: str, duration: float):
        """Log disputation completion."""
        self.logger.info(json.dumps({
            "event": "disputation_end",
            "winner": winner,
            "reason": reason,
            "duration_seconds": duration,
            "timestamp": datetime.utcnow().isoformat()
        }))
```

### 9.5 Performance Metrics

```python
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class PerformanceMetrics:
    """Track system performance metrics."""
    
    avg_response_time: float      # Seconds per LLM call
    total_api_calls: int          # Number of LLM invocations
    cache_hit_rate: float         # Percentage of cached responses
    avg_disputation_length: float # Average number of turns
    contradiction_rate: float     # Percentage ending in contradiction
    
    @classmethod
    def calculate(cls, disputations: List[DisputationResult]) -> 'PerformanceMetrics':
        """Calculate metrics from disputation history."""
        
        response_times = []
        api_calls = 0
        cache_hits = 0
        lengths = []
        contradictions = 0
        
        for disp in disputations:
            response_times.extend(disp.response_times)
            api_calls += disp.api_call_count
            cache_hits += disp.cache_hit_count
            lengths.append(len(disp.transcript))
            if disp.winner == "OPPONENT":
                contradictions += 1
        
        return cls(
            avg_response_time=np.mean(response_times),
            total_api_calls=api_calls,
            cache_hit_rate=cache_hits / api_calls if api_calls > 0 else 0,
            avg_disputation_length=np.mean(lengths),
            contradiction_rate=contradictions / len(disputations)
        )
```

## 10. Conclusion

This implementation provides a sophisticated, LLM-based system for conducting medieval obligationes disputations. By leveraging modern AI capabilities while respecting the formal structure of medieval logic described by Dutilh Novaes, we create a system that is both historically faithful and practically powerful.

The implementation maintains strict correspondence to Novaes' formalization while adapting it to modern LLM capabilities. Key achievements include:

1. **Faithful Formalization**: Direct mapping of R(φ), Γₙ, and Kc to code structures
2. **Correct Semantics**: Proper handling of NEGO as commitment to negation (Γₙ = Γₙ₋₁ ∪ {¬φ})
3. **Strategic Depth**: Multi-turn planning for Opponent that exploits logical tensions
4. **Consistency Maintenance**: Full commitment set checking with De Morgan transformations
5. **Rule Compliance**: Mechanical application of Burley's 5 rules in strict precedence order

The architecture's separation of concerns, comprehensive error handling, and extensive testing infrastructure ensure a robust, maintainable system suitable for both research and educational applications.

### Future Enhancements

1. **Multi-agent tournaments**: Multiple AI agents competing
2. **Learning capabilities**: Agents that improve through self-play
3. **Historical variations**: Support for Ockham, Swyneshed, and other rule systems
4. **Natural language understanding**: Better handling of ambiguous propositions
5. **Explanation generation**: Detailed logical proofs for each inference
6. **Web interface**: Interactive UI for educational use

The system demonstrates how classical logical frameworks can be revitalized through modern AI, creating new opportunities for understanding historical philosophical practices and developing novel approaches to automated reasoning.
