# Obligationes Quickstart Guide

Get started with medieval dialectical disputations in 5 minutes!

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

Create a `.env` file in the project root:

```bash
# For OpenAI (recommended)
OPENAI_API_KEY=your_openai_api_key_here

# OR for Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Your First Disputation

Run a simple disputation:

```bash
python -m obligationes run "Socrates is immortal" --max-turns 3
```

You'll see:
1. **Opponent** proposes challenging propositions
2. **Respondent** responds following Burley's rules (CONCEDO/NEGO/DUBITO)
3. **Judge** determines the winner based on logical consistency

## Example Output

```
======================================================================
                     OBLIGATIONES DISPUTATION
======================================================================

Positum: Socrates is immortal
Strategy: BALANCED
Max Turns: 3

======================================================================

--- Turn 1 ---

Opponent: All men are mortal
Strategy: Testing consistency with common knowledge

Respondent: NEGO
Reasoning: RULE 2: Proposition incompatible with positum...
Rule Applied: Rule 2

--- Turn 2 ---

...

======================================================================
                             JUDGMENT
======================================================================

Winner: RESPONDENT
Reason: Maintained consistency through all turns
...
```

## Common Commands

### Run with Different Strategies

```bash
# Balanced (default): Mix of direct and indirect approaches
python -m obligationes run "All ravens are black" --strategy balanced

# Aggressive: Quick contradictions
python -m obligationes run "Fire is cold" --strategy aggressive

# Pedagogical: Educational, transparent reasoning
python -m obligationes run "God exists" --strategy pedagogical
```

### Save and Replay

```bash
# Save transcript
python -m obligationes run "The Earth is flat" --output my_disputation.json

# Replay later
python -m obligationes replay my_disputation.json
```

### Quiet Mode

```bash
# Only show final judgment
python -m obligationes run "Socrates is wise" --quiet
```

## Understanding the Output

### Response Types

- **CONCEDO** (Grant): Respondent accepts the proposition
- **NEGO** (Deny): Respondent rejects the proposition
- **DUBITO** (Doubt): Respondent neither accepts nor rejects

### Burley's Rules

The Respondent follows these rules in strict order:

1. **Rule 1**: If proposition follows from commitments → CONCEDO
2. **Rule 2**: If proposition contradicts commitments → NEGO
3. **Rule 3**: If follows from commitments + common knowledge → CONCEDO
4. **Rule 4**: If contradicts commitments + common knowledge → NEGO
5. **Rule 5**: Otherwise, respond based on common knowledge truth

### Winning Conditions

- **Opponent wins**: If they force the Respondent into a contradiction
- **Respondent wins**: If they maintain consistency through all turns

## Advanced Usage

### Custom Number of Turns

```bash
python -m obligationes run "Socrates is mortal" --max-turns 10
```

### Different LLM Models

```bash
# Use GPT-4 (default)
python -m obligationes run "Truth exists" --model gpt-4

# Use GPT-4 Turbo (faster)
python -m obligationes run "Truth exists" --model gpt-4-turbo

# Use GPT-4o-mini (fastest, cheapest)
python -m obligationes run "Truth exists" --model gpt-4o-mini
```

### Disable Colors

```bash
python -m obligationes run "Logic is useful" --no-color
```

## Learning More

### Get Help

```bash
# Show all commands
python -m obligationes --help

# Show Burley's rules explanation
python -m obligationes info

# List opponent strategies
python -m obligationes list-strategies
```

### Example Positions (Posita)

Try these interesting positions:

- **Classic paradoxes**: "This statement is false"
- **Medieval questions**: "The soul is immortal", "Angels can occupy space"
- **Aristotelian physics**: "Heavy objects fall faster", "Fire rises naturally"
- **Theological**: "God is omnipotent", "Free will exists"
- **Modern**: "Artificial intelligence can think", "Time travel is possible"

## Troubleshooting

### API Key Not Found

```
Error: The api_key client option must be set...
```

**Solution**: Make sure your `.env` file is in the project root with your API key.

### Tests Timing Out

If running tests with `pytest`, they may take 15-20 minutes due to real LLM calls. This is normal.

### Import Errors

```
ModuleNotFoundError: No module named 'obligationes'
```

**Solution**: Make sure you activated the virtual environment and installed dependencies:
```bash
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Next Steps

- Read the full [README.md](README.md) for project overview
- Check [DESIGN_DOCUMENT.md](DESIGN_DOCUMENT.md) for technical details
- Explore the code in `obligationes/` directory
- Run the test suite: `pytest tests/`
- Contribute improvements via pull requests!

## Example Session

Here's a complete example session:

```bash
# Activate environment
source venv/bin/activate

# Run a disputation
python -m obligationes run "Socrates is mortal" \\
  --max-turns 5 \\
  --strategy balanced \\
  --output session1.json

# Review the rules
python -m obligationes info

# Try an aggressive opponent
python -m obligationes run "The Earth is the center of the universe" \\
  --strategy aggressive \\
  --max-turns 3

# Replay your first session
python -m obligationes replay session1.json
```

Happy disputing! 🏛️
