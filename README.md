# Obligationes

An LLM-based implementation of medieval dialectical disputation following Walter Burley's model of obligationes.

## Overview

Obligationes were formalized logical games practiced in medieval universities (13th-14th centuries) where two parties engaged in structured debates to test logical consistency. This implementation uses Large Language Models to handle natural language reasoning while maintaining the strict formal rules of the game.

In an obligationes disputation:
- The **Respondent** defends an initial position (the *positum*)
- The **Opponent** proposes subsequent propositions
- The **Respondent** must respond with CONCEDO (grant), NEGO (deny), or DUBITO (doubt) according to Burley's rules
- The **Opponent** wins if they force the Respondent into a logical contradiction

## Features

- **LLM-powered logical reasoning** for natural language propositions
- **Strict rule enforcement** following Burley's 5-rule precedence system
- **Strategic opponent AI** with multiple difficulty levels
- **Complete state tracking** with serialization for replay and analysis
- **CLI interface** for running disputations

## Quick Start

**New to Obligationes?** See the [QUICKSTART.md](QUICKSTART.md) guide for a 5-minute introduction!

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/obligationes.git
cd obligationes

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with your LLM API key:

```bash
OPENAI_API_KEY=your_key_here
# OR
ANTHROPIC_API_KEY=your_key_here
```

## Usage

### Running a Disputation

```bash
# Basic usage
python -m obligationes run "Socrates is immortal"

# Specify opponent strategy (balanced, aggressive, pedagogical)
python -m obligationes run "All ravens are black" --strategy aggressive

# Limit number of turns
python -m obligationes run "The Earth is flat" --max-turns 5

# Save transcript to file
python -m obligationes run "God exists" --output transcript.json

# Quiet mode (only show final judgment)
python -m obligationes run "Socrates is wise" --quiet

# Disable colored output
python -m obligationes run "Fire is cold" --no-color
```

### Other Commands

```bash
# Display information about Burley's rules
python -m obligationes info

# List available opponent strategies
python -m obligationes list-strategies

# Replay a saved disputation
python -m obligationes replay transcript.json

# Show help
python -m obligationes --help
```

## Development Status

- [x] **Phase 1**: Core state management and data structures
- [x] **Phase 2**: LLM inference engine
- [x] **Phase 3**: Rules engine (Burley's rules)
- [x] **Phase 4**: Agent implementation (Opponent, Respondent)
- [x] **Phase 5**: Disputation manager
- [x] **Phase 6**: CLI interface
- [x] **Phase 7**: Testing and validation
- [ ] **Phase 8**: Documentation and polish

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=obligationes

# Run specific test file
pytest tests/test_state.py -v
```

## Project Structure

```
obligationes/
├── obligationes/          # Main package
│   ├── state.py          # State management (✓ complete)
│   ├── inference.py      # LLM inference engine (✓ complete)
│   ├── rules.py          # Burley's rules (✓ complete)
│   ├── agents.py         # Opponent/Respondent agents (✓ complete)
│   ├── manager.py        # Disputation orchestration (✓ complete)
│   ├── cli.py            # Command-line interface (✓ complete)
│   └── __main__.py       # Package entry point (✓ complete)
├── tests/                 # Test suite
├── DESIGN_DOCUMENT.md    # Detailed technical design
├── IMPLEMENTATION_PLAN.md # Development roadmap
└── CLAUDE.md             # AI assistant guidance
```

## Burley's Rules

The system implements Walter Burley's 5 rules in strict precedence order:

1. If proposition follows from (positum + concessa) → CONCEDO
2. If proposition incompatible with (positum + concessa) → NEGO
3. If proposition follows from (positum + concessa + common knowledge) → CONCEDO
4. If proposition incompatible with (positum + concessa + common knowledge) → NEGO
5. Otherwise, respond based on truth in common knowledge

## References

- Catarina Dutilh Novaes, "Formalizing Medieval Logical Theories"
- Walter Burley's treatise on obligations
- Medieval logic and scholastic disputation practices

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Performance Notes

The system makes extensive use of LLM API calls:
- A typical 10-turn disputation makes ~20-30 API calls
- Uses temperature=0 for deterministic responses
- Recommend using `gpt-4o-mini` for faster, cheaper testing
- The test suite (102 tests) takes ~15 minutes with `gpt-4o-mini`

## Contributing

Contributions are welcome! Areas for improvement:

- Additional opponent strategies
- Support for other medieval rule systems (Ockham, Swyneshed)
- Web interface for interactive use
- Performance optimizations
- Additional test scenarios
- Documentation improvements

Please open an issue to discuss major changes before submitting a PR.

## Acknowledgments

- **Walter Burley** for the original obligationes rules (14th century)
- **Catarina Dutilh Novaes** for formalizing medieval logic in modern terms
- The **LangChain** team for excellent LLM orchestration tools
- **OpenAI** and **Anthropic** for powerful language models
