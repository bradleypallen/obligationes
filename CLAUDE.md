# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM-based implementation of **Obligationes**, a medieval dialectical disputation system. The project implements Walter Burley's model of obligationes (formalized by Catarina Dutilh Novaes) where two parties engage in structured logical debates to test logical consistency.

The implementation is a **CLI application** for running disputations between AI agents.

## Architecture

The system uses a **layered architecture** with clear separation of concerns:

1. **Game Management Layer**: Orchestrates disputation flow and turn management
2. **Rules Engine Layer**: Implements Burley's obligationes rules in strict order
3. **Logical Reasoning Layer**: LLM-based inference engine for natural language logic
4. **State Management Layer**: Tracks commitments (concessa/negata), maintains history

### Core Components

- **DisputationManager**: Main orchestrator with two agents (Opponent, Respondent)
- **LogicalReasoningEngine**: Uses LLMs for inference, consistency checking, and compatibility analysis
- **RulesEngine (BurleyEvaluator)**: Applies Burley's 5 rules in strict precedence order
- **StateManager**: Maintains immutable state transitions with CommitmentStore, PropositionHistory, and ConsistencyMonitor
- **LangChain Integration**: Custom memory, prompt templates, and output parsers

### Key Design Decisions

- **LLM-Centric Logic**: All logical reasoning uses LLMs (GPT-4/Claude) rather than symbolic engines, enabling sophisticated natural language inference at the cost of non-determinism
- **Separate Agents**: Opponent and Respondent are distinct agents with different prompting strategies (strategic vs. rule-following)
- **State Immutability**: State updates are treated as immutable transitions for debugging, replay, and analysis
- **Temperature=0**: Used throughout for consistency and determinism
- **Structured Output**: All LLM responses use JSON format for reliable parsing

## Development Setup

**Implementation Status**: ✅ **COMPLETE** - All phases implemented and tested.

### Installation
```bash
# Clone and setup
git clone https://github.com/yourusername/obligationes.git
cd obligationes
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Dependencies
- Python 3.11+
- **LLM**: langchain, langchain-openai, langchain-anthropic
- **CLI**: click
- **Testing**: pytest, pytest-cov
- **Data**: pydantic
- **Environment**: python-dotenv

### Environment Variables
Create a `.env` file:
```bash
OPENAI_API_KEY=your_key_here
# OR
ANTHROPIC_API_KEY=your_key_here
```

## Burley's Rules (Critical Implementation Detail)

Rules MUST be applied in strict order of precedence:

1. If proposition follows from (positum + concessa) → CONCEDO
2. If proposition incompatible with (positum + concessa) → NEGO
3. If proposition follows from (positum + concessa + common_knowledge) → CONCEDO
4. If proposition incompatible with (positum + concessa + common_knowledge) → NEGO
5. Otherwise, respond based on truth in common knowledge

The Respondent **must** follow these rules even if they lead to contradiction—this is the core constraint of the game.

## Data Structures

### Response Types
- `CONCEDO`: Grant/accept
- `NEGO`: Deny
- `DUBITO`: Doubt (neither grant nor deny)

### Proposition Statuses
- `POSITUM`: Initial position to defend
- `CONCESSA`: Granted propositions
- `NEGATA`: Denied propositions
- `DUBITATA`: Doubted propositions
- `IRRELEVANT`: Not yet addressed

### State Container
The `ObligationesState` maintains:
- Current positum, concessa set, negata set, dubitata set
- Turn count and active status
- Complete history of exchanges with inference chains

## LLM Integration Details

### Inference Types Required
- **Modus Ponens**: P→Q, P ⊢ Q
- **Modus Tollens**: P→Q, ¬Q ⊢ ¬P
- **Syllogisms**: All A are B, X is A ⊢ X is B
- **Conjunction/Disjunction**: Logical combination rules

### Prompt Engineering Constraints
- System prompts must emphasize rule-following over strategy for Respondent
- Opponent prompts should focus on finding multi-step trap sequences
- Winner determination is mechanical (contradiction detected = Opponent wins)
- All prompts use temperature=0 for consistency
- JSON output format enforced via structured prompts

## Error Handling

- **LLM Response Parsing**: Multi-layer fallback strategy (JSON → regex → keyword detection)
- **Consistency Validation**: Simulate responses before committing to check for contradictions
- **LLM Failures**: Safe defaults (return False for inference, log errors)
- **State Corruption**: Prevented via immutability

## File Organization

**Implemented Structure:**
- `obligationes/state.py`: State management data structures (✅ complete)
- `obligationes/inference.py`: LLM-based logical reasoning engine (✅ complete)
- `obligationes/rules.py`: Burley's rules implementation (✅ complete)
- `obligationes/agents.py`: Opponent and Respondent agents (✅ complete)
- `obligationes/manager.py`: DisputationManager orchestrator (✅ complete)
- `obligationes/cli.py`: Command-line interface with colored output (✅ complete)
- `obligationes/__main__.py`: Package entry point (✅ complete)
- `tests/`: Unit and integration tests with pytest (✅ 102 tests, all passing)
- `tests/conftest.py`: Test configuration (uses gpt-4o-mini for faster testing)

## CLI Usage

**Available Commands:**
```bash
# Run a disputation
python -m obligationes run "Socrates is immortal"
python -m obligationes run "Socrates is immortal" --max-turns 5 --strategy aggressive

# Save transcript
python -m obligationes run "God exists" --output transcript.json

# Replay saved disputation
python -m obligationes replay transcript.json

# Get help
python -m obligationes info                    # Show Burley's rules
python -m obligationes list-strategies         # Show opponent strategies
python -m obligationes --help                  # Show all commands
```

**Options:**
- `--max-turns, -t`: Maximum number of turns (default: 10)
- `--strategy, -s`: Opponent strategy: balanced, aggressive, pedagogical (default: balanced)
- `--model, -m`: LLM model to use (default: gpt-4)
- `--output, -o`: Save transcript to JSON file
- `--quiet, -q`: Suppress verbose output (only show final judgment)
- `--no-color`: Disable colored output

## Testing Strategy

**Implemented Test Suite** (102 tests, all passing):
- **Unit Tests**: Individual inference patterns, rule precedence, state management
- **Integration Tests**: Complete disputation flows with both agents
- **Agent Tests**: Respondent rule-following, Opponent strategy and planning
- **Manager Tests**: Full orchestration, transcript save/load, status tracking, mechanical winner determination
- **Real LLM Testing**: All tests use real API calls (gpt-4o-mini for speed)
- **Code Quality**: black, mypy, ruff all passing

## Medieval Context

The implementation follows **13th-14th century scholastic disputation practices**. Common knowledge sets should reflect medieval worldviews for historical accuracy (e.g., geocentrism, four elements, Aristotelian physics).

## References

See `DESIGN_DOCUMENT.md` for complete technical specifications, including:
- Detailed component diagrams
- Full code examples for each component
- Usage patterns and examples
- Whenever a phase of development is complete, remember to update the project status in the README.md file.
- Before pushing the repository to GitHub, run basic code quality tests (black, mypy, ruff) and all pytest tests, and fix any problems uncovered.