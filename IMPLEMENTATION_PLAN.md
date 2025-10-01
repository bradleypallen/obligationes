# Implementation Plan for Obligationes System

## Phase 1: Foundation (Core Data Structures)

**1.1 State Management (`obligationes/state.py`)**
- Define enums: `ResponseType`, `PropositionStatus`
- Implement `Proposition` dataclass with content, status, turn number, and inference metadata
- Implement `ObligationesState` class:
  - Track positum, concessa, negata, dubitata sets
  - Maintain turn history as immutable list
  - Provide methods for querying commitments
  - Implement state serialization for debugging/replay

**Why this order?** Everything else depends on these data structures. Get them right first.

## Phase 2: LLM Inference Engine

**2.1 Basic Inference (`obligationes/inference.py`)**
- Create `LLMInferenceEngine` class with configurable LLM backend
- Implement three core chains:
  - **FollowsFromChain**: Determines if conclusion necessarily follows from premises
  - **IncompatibleWithChain**: Detects logical contradictions/incompatibilities
  - **ConsistencyChain**: Checks overall state consistency
- Use structured JSON output with Pydantic models for parsing
- Implement robust error handling with fallback parsing strategies

**2.2 Prompt Engineering**
- Design prompts that clearly define "necessarily follows" vs "compatible with" vs "true in common knowledge"
- Include examples of valid inference patterns (modus ponens, syllogisms, etc.)
- Ensure prompts distinguish between deductive validity and mere plausibility

**Critical design choice**: Start with a single LLM instance (GPT-4 or Claude) rather than multiple models. Keep prompts modular so they can be swapped easily.

## Phase 3: Rules Engine

**3.1 Burley's Rules (`obligationes/rules.py`)**
- Implement `BurleyRulesEngine` class
- Create `evaluate_proposition()` method that:
  - Applies 5 rules in strict order
  - Returns tuple: (ResponseType, reasoning, rule_number)
  - Uses InferenceEngine for logical checks at each rule
  - Short-circuits as soon as a rule fires

**3.2 Rule Implementation Strategy**
Each rule should be a separate method that returns `Optional[Tuple[ResponseType, str]]`:
- `_check_rule_1()`: follows from positum+concessa
- `_check_rule_2()`: incompatible with positum+concessa
- `_check_rule_3()`: follows from positum+concessa+common_knowledge
- `_check_rule_4()`: incompatible with positum+concessa+common_knowledge
- `_check_rule_5()`: truth in common knowledge alone

**Why separate methods?** Easier to test, debug, and modify individual rules.

## Phase 4: Agent Implementation

**4.1 Respondent Agent (`obligationes/agents.py`)**
- Implement `RespondentAgent` class
- Agent MUST use RulesEngine and cannot deviate
- Provide introspection: can detect traps but must follow rules anyway
- Return structured response with reasoning and detected vulnerabilities

**4.2 Opponent Agent (`obligationes/agents.py`)**
- Implement `OpponentAgent` class with strategy selection
- Strategies to implement:
  - **Balanced**: Mix of direct and indirect approaches
  - **Aggressive**: Focus on forcing contradictions quickly
  - **Pedagogical**: Demonstrate logical principles clearly
- Use LLM to analyze current state and generate strategic propositions
- Track multi-step trap sequences

**4.3 Judge Agent (`obligationes/agents.py`)**
- Implement `JudgeAgent` class
- Analyzes complete transcript
- Determines winner and provides detailed reasoning
- Validates rule compliance throughout

**Design consideration**: Each agent gets its own system prompt and chain. Don't try to make one LLM do everything with context switching.

## Phase 5: Disputation Management

**5.1 Core Manager (`obligationes/manager.py`)**
- Implement `DisputationManager` class
- Orchestrate the game loop:
  1. Initialize with positum and common knowledge
  2. Run turns until max_turns or contradiction
  3. Track state updates after each response
  4. Check consistency after each turn
  5. Invoke judge at completion
- Maintain immutable state transitions
- Provide step-by-step execution mode for debugging

**5.2 Common Knowledge Management**
- Create default medieval common knowledge set
- Allow custom knowledge sets
- Format knowledge as clear declarative statements

## Phase 6: CLI Interface

**6.1 Command-Line Interface (`obligationes/cli.py`)**
- Use `argparse` or `click` for CLI argument parsing
- Support commands:
  - `run`: Execute complete disputation
  - `interactive`: Human plays as Respondent (future enhancement)
- Arguments:
  - `--positum`: Initial position (required)
  - `--max-turns`: Maximum exchanges (default: 10)
  - `--strategy`: Opponent strategy (default: balanced)
  - `--verbose`: Detailed output
  - `--output`: Save transcript to file
  - `--model`: LLM model to use
  - `--common-knowledge`: Path to custom knowledge file

**6.2 Output Formatting**
- Clear turn-by-turn display
- Color coding for responses (CONCEDO=green, NEGO=red, DUBITO=yellow)
- Show reasoning at each step
- Final judgment with winner and explanation

## Phase 7: Testing

**7.1 Unit Tests**
- Test each rule in isolation with known inputs
- Test inference patterns (modus ponens, contradictions, etc.)
- Mock LLM responses for deterministic testing
- Test state transitions and immutability

**7.2 Integration Tests**
- Run complete disputations with known outcomes
- Test edge cases (immediate contradictions, max turns, etc.)
- Validate rule precedence
- Test different opponent strategies

**7.3 Validation**
- Create test suite with historical examples if available
- Verify Burley compliance manually on sample disputations
- Test with deliberately contradictory setups

## Phase 8: Polish & Documentation

**8.1 Error Handling**
- Graceful LLM API failures
- Retry logic with exponential backoff
- Clear error messages for users
- Logging for debugging

**8.2 Documentation**
- Docstrings for all classes and methods
- Usage examples in README
- Example disputations with analysis

---

## Implementation Order

**Phase 1: Foundation**
1. Set up project structure and dependencies
2. Implement state.py completely
3. Write unit tests for state management

**Phase 2: Logic Core**
4. Implement inference.py with all three chains
5. Test inference with manual examples
6. Implement rules.py with all 5 rules
7. Test rules in isolation

**Phase 3: Agents**
8. Implement Respondent agent (simpler, rule-bound)
9. Implement Opponent agent (more complex, strategic)
10. Implement Judge agent
11. Test agents independently

**Phase 4: Integration**
12. Implement DisputationManager
13. Implement CLI
14. Integration testing
15. Polish and documentation

---

## Key Design Decisions

**1. LangChain vs Direct LLM Calls**
- Use LangChain for chain management and structured output
- Use Pydantic for response validation
- Avoid heavy LangChain features like agents or tools—we're building our own agent architecture

**2. Synchronous vs Async**
- Start with synchronous implementation
- LLM calls are the bottleneck anyway
- Easier to debug and test

**3. Configuration Management**
- Use environment variables for API keys
- Use dataclasses for configuration objects
- Support both code-based and file-based config

**4. State Persistence**
- Keep state in memory during execution
- Support JSON serialization for saving transcripts
- No database needed for CLI version

**5. Prompt Strategy**
- Store prompts as templates in separate files or constants
- Use clear, explicit instructions over clever tricks
- Include few-shot examples in prompts for complex reasoning

---

## Risks & Mitigations

**Risk 1: LLM Non-Determinism**
- *Mitigation*: Temperature=0, structured output, multiple validation passes

**Risk 2: Complex Inference Failures**
- *Mitigation*: Start with simple test cases, gradually increase complexity

**Risk 3: Rule Precedence Bugs**
- *Mitigation*: Extensive unit tests for each rule, integration tests for precedence

**Risk 4: LLM Cost**
- *Mitigation*: Use shorter prompts, implement simple caching for identical queries during testing

---

## First Concrete Step

Start with `obligationes/state.py`:
```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Set, List, Optional, Dict, Any

class ResponseType(Enum):
    CONCEDO = "concedo"
    NEGO = "nego"
    DUBITO = "dubito"

class PropositionStatus(Enum):
    POSITUM = "positum"
    CONCESSA = "concessa"
    NEGATA = "negata"
    DUBITATA = "dubitata"
    IRRELEVANT = "irrelevant"

@dataclass
class Proposition:
    content: str
    status: PropositionStatus
    turn_introduced: int
    follows_from: List[str] = field(default_factory=list)
    incompatible_with: List[str] = field(default_factory=list)
    inference_chain: Optional[str] = None
```

This grounds everything in concrete, testable code from day one.
