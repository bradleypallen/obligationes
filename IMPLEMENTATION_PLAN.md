# Implementation Plan for Obligationes System

> **Status**: ✅ **COMPLETE** - All 8 phases implemented successfully
> **Final Test Results**: 97 tests passing in ~17 minutes (using gpt-4o-mini)
> **Last Updated**: October 2025

## Overview of Actual Implementation

This document originally outlined an 8-phase implementation plan. The actual implementation followed this plan closely with some notable deviations:

**Key Changes from Original Plan:**
1. **Judge Agent Removed**: Winner determination became mechanical (contradiction detected = Opponent wins) rather than requiring a Judge agent
2. **Semantic Duplicate Detection**: Added sophisticated duplicate proposition detection using LLM semantic equivalence checking
3. **NEGO Semantics**: Implemented explicit handling of NEGO as commitment to negation with NOT() notation
4. **Self-Contradiction Check**: Added positum self-contradiction detection per Novaes formalization (R(φ₀) = 0 iff φ₀ ⊢⊥)
5. **Strategic Planning**: Opponent uses multi-turn strategic planning with trap sequences
6. **Turn 0**: Formalized positum acceptance as "Turn 0" with explicit rule application

---

## Phase 1: Foundation (Core Data Structures) ✅

**1.1 State Management (`obligationes/state.py`)**
- Define enums: `ResponseType`, `PropositionStatus`
- Implement `Proposition` dataclass with content, status, turn number, and inference metadata
- Implement `ObligationesState` class:
  - Track positum, concessa, negata, dubitata sets
  - Maintain turn history as immutable list
  - Provide methods for querying commitments
  - Implement state serialization for debugging/replay

**Why this order?** Everything else depends on these data structures. Get them right first.

**Actual Implementation:**
- ✅ All planned features implemented
- ✅ Added `Turn` dataclass for history tracking
- ✅ Added `get_all_negations()` method for NEGO semantics
- ✅ State includes common_knowledge set
- ✅ Comprehensive state serialization via `to_dict()` and `from_dict()`

## Phase 2: LLM Inference Engine ✅

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

**Actual Implementation:**
- ✅ All three chains implemented as planned
- ✅ Added `is_self_contradictory()` chain for positum validation (not in original plan)
- ✅ Added `semantically_equivalent()` chain for duplicate detection (not in original plan)
- ✅ Added `is_known()` chain for Rule 5 evaluation
- ✅ Consistency chain includes De Morgan's law handling for NOT() notation
- ✅ Comprehensive error handling with fallback parsing
- ✅ Uses Pydantic models for structured output

## Phase 3: Rules Engine ✅

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

**Actual Implementation:**
- ✅ `BurleyRulesEngine` implemented with all 5 rules in strict precedence
- ✅ Uses single `evaluate_proposition()` method with sequential rule checking (not separate methods)
- ✅ Rules check commitments + common_knowledge as per Burley's formalization
- ✅ Returns (ResponseType, reasoning, rule_number) tuple
- ✅ Properly handles empty commitment sets

## Phase 4: Agent Implementation ✅

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

**4.3 Judge Agent (`obligationes/agents.py`)** ❌ **REMOVED**
- ~~Implement `JudgeAgent` class~~
- ~~Analyzes complete transcript~~
- ~~Determines winner and provides detailed reasoning~~
- ~~Validates rule compliance throughout~~

**Design consideration**: Each agent gets its own system prompt and chain. Don't try to make one LLM do everything with context switching.

**Actual Implementation:**
- ✅ `RespondentAgent` implemented exactly as planned
- ✅ Trap detection via `_analyze_for_trap()` method
- ✅ `OpponentAgent` implemented with three strategies (BALANCED, AGGRESSIVE, PEDAGOGICAL)
- ✅ Strategic planning with `_create_strategic_plan()` method
- ✅ **Multi-turn planning**: Opponent creates strategic plan and adapts based on state
- ✅ **Semantic duplicate prevention**: Added sophisticated duplicate detection with retry logic
- ❌ **Judge Agent removed**: Winner determination is mechanical (contradiction = Opponent wins)
- ✅ Uses Pydantic models: `RespondentResponse`, `OpponentProposal`

## Phase 5: Disputation Management ✅

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

**Actual Implementation:**
- ✅ `DisputationManager` orchestrates complete game loop
- ✅ **Turn 0 implementation**: Positum self-contradiction check per Novaes (R(φ₀) = 0 iff φ₀ ⊢⊥)
- ✅ **Consistency checking**: Full commitment set includes NOT() of all negations
- ✅ **Mechanical winner determination**: No Judge needed, contradiction detection is deterministic
- ✅ `DisputationResult` dataclass with comprehensive outcome data
- ✅ `DisputationConfig` dataclass for configuration
- ✅ `DEFAULT_COMMON_KNOWLEDGE` with medieval facts
- ✅ `create_disputation()` convenience function
- ✅ Transcript save/load with JSON serialization
- ✅ `from_transcript()` class method for replay

## Phase 6: CLI Interface ✅

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

**Actual Implementation:**
- ✅ Uses `argparse` for CLI parsing
- ✅ Commands implemented:
  - `run`: Execute complete disputation ✅
  - `info`: Display Burley's rules ✅
  - `list-strategies`: Show available strategies ✅
  - `replay`: Replay saved transcript ✅
- ✅ All planned arguments implemented: `--positum`, `--max-turns`, `--strategy`, `--verbose`, `--output`, `--model`, `--vendor`
- ✅ Color-coded output using `colorama` (CONCEDO=green, NEGO=red)
- ✅ Turn-by-turn display with reasoning
- ✅ Final judgment display without separate Judge agent

## Phase 7: Testing ✅

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

**Actual Implementation:**
- ✅ Comprehensive test suite: **97 tests, all passing**
- ✅ Test files:
  - `test_state.py`: State management and data structures
  - `test_inference.py`: LLM inference chains
  - `test_rules.py`: Burley's rules engine
  - `test_agents.py`: Agent behavior (Respondent, Opponent)
  - `test_manager.py`: Disputation orchestration
  - `test_cli.py`: Command-line interface
- ✅ Uses `conftest.py` to override model to `gpt-4o-mini` for fast/cheap testing
- ✅ Mock LLM responses where appropriate
- ✅ Integration tests for complete disputation flows
- ✅ **Test runtime**: ~17 minutes with gpt-4o-mini
- ✅ **Optimized**: Reduced turns from 3→2 in tests for performance

## Phase 8: Polish & Documentation ✅

**8.1 Error Handling**
- Graceful LLM API failures
- Retry logic with exponential backoff
- Clear error messages for users
- Logging for debugging

**8.2 Documentation**
- Docstrings for all classes and methods
- Usage examples in README
- Example disputations with analysis

**Actual Implementation:**
- ✅ Comprehensive docstrings throughout codebase
- ✅ `README.md` with:
  - Installation instructions (5-step setup)
  - API key configuration for OpenAI/Anthropic
  - Usage examples and CLI commands
  - Troubleshooting section
  - Performance notes
- ✅ `DESIGN_DOCUMENT.md` with:
  - Complete technical design specification
  - Correspondence to Dutilh Novaes formalization (Section 2)
  - System architecture and data flow
  - Implementation details and examples
- ✅ `CLAUDE.md` for AI assistant guidance
- ✅ `QUICKSTART.md` for new users
- ✅ Code quality checks: black, mypy, ruff (all passing)
- ✅ Error handling with fallback strategies for LLM parsing

---

## Actual Implementation Order

**Phase 1: Foundation** ✅
1. ✅ Set up project structure and dependencies
2. ✅ Implement state.py completely
3. ✅ Write unit tests for state management

**Phase 2: Logic Core** ✅
4. ✅ Implement inference.py with all five chains (added self-contradiction & semantic equivalence)
5. ✅ Test inference with manual examples
6. ✅ Implement rules.py with all 5 rules
7. ✅ Test rules in isolation

**Phase 3: Agents** ✅
8. ✅ Implement Respondent agent (simpler, rule-bound)
9. ✅ Implement Opponent agent (more complex, strategic with planning)
10. ❌ ~~Implement Judge agent~~ → Removed (mechanical winner determination)
11. ✅ Test agents independently

**Phase 4: Integration** ✅
12. ✅ Implement DisputationManager with Turn 0 and consistency checking
13. ✅ Implement CLI with run/info/list-strategies/replay commands
14. ✅ Integration testing (97 tests passing)
15. ✅ Polish and documentation

**Additional Work Completed:**
- ✅ Added NEGO semantics with NOT() notation
- ✅ Added semantic duplicate detection for Opponent
- ✅ Added Dutilh Novaes formalization mapping to design doc
- ✅ Optimized tests for performance (~17 min runtime)
- ✅ Comprehensive README with installation/troubleshooting

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

## Risks & Mitigations (Retrospective)

**Risk 1: LLM Non-Determinism** ✅ **Mitigated**
- *Original Mitigation*: Temperature=0, structured output, multiple validation passes
- *Actual Result*: Successfully achieved consistent behavior with temperature=0 and Pydantic models

**Risk 2: Complex Inference Failures** ✅ **Mitigated**
- *Original Mitigation*: Start with simple test cases, gradually increase complexity
- *Actual Result*: LLM handles complex multi-step inferences well; De Morgan transformations work correctly

**Risk 3: Rule Precedence Bugs** ✅ **Mitigated**
- *Original Mitigation*: Extensive unit tests for each rule, integration tests for precedence
- *Actual Result*: 97 tests ensure correct rule precedence; no bugs found in final implementation

**Risk 4: LLM Cost** ✅ **Mitigated**
- *Original Mitigation*: Use shorter prompts, implement simple caching for identical queries during testing
- *Actual Result*: Used gpt-4o-mini for testing (~17 min, $0.50/run); production uses gpt-4 or claude

**New Risks Discovered:**
- **Opponent Repeating Propositions**: Mitigated with semantic equivalence checking and retry logic
- **NEGO Semantics**: Required explicit NOT() notation and De Morgan handling in consistency checker
- **Test Performance**: Mitigated by reducing turns (3→2) and using fast model (gpt-4o-mini)

---

## Lessons Learned

**What Went Well:**
1. **Phased Approach**: Building from state → inference → rules → agents → manager worked perfectly
2. **Test-Driven Development**: Writing tests alongside implementation caught bugs early
3. **LLM-based Logic**: LLMs handle natural language reasoning better than expected
4. **Structured Output**: Pydantic models + temperature=0 provide reliable consistency
5. **Modular Design**: Each component can be modified independently

**What Required Adjustment:**
1. **Judge Agent Unnecessary**: Contradiction detection is mechanical, no judgment needed
2. **NEGO Semantics**: Required explicit negation tracking and consistency checking
3. **Duplicate Prevention**: Needed semantic equivalence, not just string matching
4. **Turn 0**: Formalized positum acceptance as explicit rule application
5. **Test Performance**: Had to optimize (reduce turns, use fast model) for practical runtime

**Key Technical Insights:**
1. **De Morgan Laws**: LLM needed explicit guidance on NOT() transformations
2. **Strategic Planning**: Opponent benefits greatly from multi-turn planning
3. **Temperature=0**: Essential for deterministic behavior in formal logic
4. **Fallback Parsing**: LLM output isn't always perfect JSON; need robust parsing
5. **Conftest Override**: Pytest fixtures can override defaults for faster testing

**Recommendations for Future Work:**
1. Consider caching LLM calls for repeated propositions
2. Add more opponent strategies (e.g., Socratic, defensive)
3. Implement alternative rule systems (Ockham, Swyneshed)
4. Build web interface for educational use
5. Add learning capabilities for Opponent strategy improvement

---

## Final Statistics

- **Total Implementation Time**: ~8 phases completed
- **Lines of Code**: ~2,500 (excluding tests)
- **Test Coverage**: 97 tests, all passing
- **Test Runtime**: ~17 minutes (gpt-4o-mini)
- **Files Created**:
  - Core: 6 (state.py, inference.py, rules.py, agents.py, manager.py, cli.py)
  - Tests: 6 (test_*.py files)
  - Docs: 4 (README.md, DESIGN_DOCUMENT.md, IMPLEMENTATION_PLAN.md, CLAUDE.md)
- **Documentation**: ~3,000 lines of technical documentation
- **API Endpoints**: CLI with 4 commands (run, info, list-strategies, replay)
- **Supported LLMs**: OpenAI (GPT-4, GPT-4o-mini), Anthropic (Claude)
