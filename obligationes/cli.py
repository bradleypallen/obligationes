"""
Command-line interface for obligationes disputations.

This module provides a CLI for running medieval dialectical disputations
following Walter Burley's obligationes rules.
"""

import sys
import click
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from obligationes.manager import (  # noqa: E402
    DisputationManager,
    DisputationConfig,
)
from obligationes.agents import OpponentStrategy  # noqa: E402


# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


def print_header(text: str) -> None:
    """Print a formatted header."""
    click.echo(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    click.echo(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.RESET}")
    click.echo(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")


def print_turn(turn_num) -> None:
    """Print a turn header."""
    click.echo(f"\n{Colors.BOLD}{Colors.YELLOW}--- Turn {turn_num} ---{Colors.RESET}\n")


def print_opponent(proposition: str, strategy_note: str) -> None:
    """Print opponent's proposal."""
    click.echo(f"{Colors.BOLD}{Colors.RED}Opponent:{Colors.RESET} {proposition}")
    click.echo(f"{Colors.MAGENTA}Strategy:{Colors.RESET} {strategy_note}")


def print_respondent(response: str, reasoning: str, rule: int) -> None:
    """Print respondent's response."""
    # Color based on response type
    color = (
        Colors.GREEN
        if response == "CONCEDO"
        else Colors.RED if response == "NEGO" else Colors.YELLOW
    )

    click.echo(
        f"\n{Colors.BOLD}{Colors.BLUE}Respondent:{Colors.RESET} {color}{Colors.BOLD}{response}{Colors.RESET}"
    )
    click.echo(f"{Colors.CYAN}Reasoning:{Colors.RESET} {reasoning}")
    click.echo(f"{Colors.CYAN}Rule Applied:{Colors.RESET} Rule {rule}")


def print_trap(trap_analysis: str) -> None:
    """Print trap detection warning."""
    click.echo(f"{Colors.BOLD}{Colors.RED}⚠️  {trap_analysis}{Colors.RESET}")


def print_contradiction() -> None:
    """Print contradiction detection."""
    click.echo(f"\n{Colors.BOLD}{Colors.RED}❌ CONTRADICTION DETECTED!{Colors.RESET}")


def print_judgment(judgment: dict) -> None:
    """Print final judgment."""
    print_header("JUDGMENT")

    winner_color = Colors.GREEN if judgment["winner"] == "RESPONDENT" else Colors.RED
    click.echo(
        f"{Colors.BOLD}Winner:{Colors.RESET} {winner_color}{Colors.BOLD}{judgment['winner']}{Colors.RESET}"
    )
    click.echo(f"{Colors.BOLD}Reason:{Colors.RESET} {judgment['reason']}")
    click.echo(f"\n{Colors.BOLD}Overall Assessment:{Colors.RESET}")
    click.echo(judgment["overall_assessment"])

    if judgment.get("key_moments"):
        click.echo(f"\n{Colors.BOLD}Key Moments:{Colors.RESET}")
        for moment in judgment["key_moments"]:
            click.echo(f"  • {moment}")

    if judgment.get("rule_violations"):
        click.echo(f"\n{Colors.BOLD}{Colors.RED}Rule Violations:{Colors.RESET}")
        for violation in judgment["rule_violations"]:
            click.echo(f"  • {violation}")

    click.echo(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}\n")


@click.group()
@click.version_option(version="1.0.0", prog_name="obligationes")
def cli():
    """
    Obligationes - Medieval Dialectical Disputation System

    An LLM-based implementation of Walter Burley's obligationes rules
    for structured logical debates.
    """
    pass


@cli.command()
@click.argument("positum")
@click.option(
    "--max-turns",
    "-t",
    default=10,
    type=int,
    help="Maximum number of turns (default: 10)",
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["balanced", "aggressive", "pedagogical"], case_sensitive=False),
    default="balanced",
    help="Opponent strategy (default: balanced)",
)
@click.option(
    "--model",
    "-m",
    default="gpt-4o-mini",
    help="LLM model to use (default: gpt-4o-mini). Examples: gpt-4, gpt-4o-mini, claude-3-5-sonnet-20241022",
)
@click.option(
    "--vendor",
    "-v",
    type=click.Choice(["openai", "anthropic", "auto"], case_sensitive=False),
    default="auto",
    help="LLM vendor (default: auto-detect from model name)",
)
@click.option("--output", "-o", type=click.Path(), help="Save transcript to JSON file")
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress verbose output (only show final judgment)",
)
@click.option("--no-color", is_flag=True, help="Disable colored output")
def run(
    positum: str,
    max_turns: int,
    strategy: str,
    model: str,
    vendor: str,
    output: Optional[str],
    quiet: bool,
    no_color: bool,
):
    """
    Run a complete obligationes disputation.

    POSITUM is the initial proposition the Respondent must defend.

    Example:
        obligationes run "Socrates is immortal" --max-turns 5 --strategy aggressive
    """
    # Disable colors if requested
    if no_color:
        for attr in dir(Colors):
            if not attr.startswith("_"):
                setattr(Colors, attr, "")

    # Convert strategy string to enum
    strategy_map = {
        "balanced": OpponentStrategy.BALANCED,
        "aggressive": OpponentStrategy.AGGRESSIVE,
        "pedagogical": OpponentStrategy.PEDAGOGICAL,
    }
    opponent_strategy = strategy_map[strategy.lower()]

    # Create configuration
    config = DisputationConfig(
        max_turns=max_turns,
        opponent_strategy=opponent_strategy,
        verbose=not quiet,
        model_name=model,
        temperature=0.0,
        vendor=None if vendor == "auto" else vendor,
    )

    # Create manager
    manager = DisputationManager(config=config)

    # Print header (unless quiet)
    if not quiet:
        print_header("OBLIGATIONES DISPUTATION")
        click.echo(f"{Colors.BOLD}Positum:{Colors.RESET} {positum}")
        click.echo(f"{Colors.BOLD}Strategy:{Colors.RESET} {strategy.upper()}")
        click.echo(f"{Colors.BOLD}Max Turns:{Colors.RESET} {max_turns}")
        click.echo(f"{Colors.BOLD}Model:{Colors.RESET} {model}")
        click.echo(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")

    # Set up custom verbose output if not quiet
    if not quiet:
        # Override manager's verbose output with our colored version
        manager.config.verbose = False  # Disable default output

        # Start timing
        manager.start_time = __import__("datetime").datetime.utcnow()

        # Turn 0: Check if positum is self-contradictory (per Novaes 2005)
        # R(φ₀) = 0 iff φ₀ ⊢⊥ (reject if self-contradictory)
        # R(φ₀) = 1 iff φ₀ ⊬⊥ (accept if not self-contradictory)
        print_turn("0 (Positum)")
        click.echo(f"{Colors.BOLD}{Colors.RED}Opponent proposes:{Colors.RESET} {positum}")

        # Check if positum is self-contradictory
        positum_self_contradictory, contradiction_reasoning = (
            manager.inference_engine.is_self_contradictory(positum)
        )

        from obligationes.state import ResponseType

        if positum_self_contradictory:
            # Positum is self-contradictory - reject it
            print_respondent(
                "NEGO",
                f"The positum is self-contradictory and cannot be defended. {contradiction_reasoning}",
                0,  # Rule 0 = positum rejection
            )
            print_contradiction()
            click.echo(f"\n{Colors.BOLD}{Colors.RED}DISPUTATION CANNOT BEGIN - POSITUM REJECTED{Colors.RESET}")

            # End timing
            manager.end_time = __import__("datetime").datetime.utcnow()

            # Show rejection result - Respondent wins by correctly rejecting invalid positum
            print_header("FINAL RESULT")
            click.echo(f"{Colors.BOLD}Winner:{Colors.RESET} {Colors.GREEN}{Colors.BOLD}RESPONDENT{Colors.RESET}")
            click.echo(f"{Colors.BOLD}Reason:{Colors.RESET} Positum was self-contradictory and correctly rejected")
            click.echo(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}\n")

            # Save transcript if requested
            if output:
                output_path = Path(output)
                manager.save_transcript(str(output_path))
                click.echo(f"\n{Colors.GREEN}✓{Colors.RESET} Transcript saved to {output_path}")

            return

        # Positum is not self-contradictory - accept it
        print_respondent(
            "CONCEDO",
            "The positum is not self-contradictory and is accepted by obligation. The Respondent commits to defend this position.",
            0,  # Rule 0 = positum acceptance
        )

        # Set positum and record the CONCEDO response
        manager.state.set_positum(positum)
        manager.state.add_response(
            positum,
            ResponseType.CONCEDO,
            "The positum is not self-contradictory and is accepted by obligation. The Respondent commits to defend this position.",
            0,
        )
        manager.state.history[-1].consistency_maintained = True

        contradiction_found = False

        # Run disputation loop manually for colored output (if not already contradicted)
        if not contradiction_found:
            for turn in range(max_turns):
                print_turn(turn + 1)

                # Opponent proposes
                proposal = manager.opponent.propose_proposition(manager.state)
                print_opponent(proposal["proposition"], proposal["strategy_note"])

                # Respondent evaluates
                evaluation = manager.respondent.evaluate_proposition(
                    proposal["proposition"], manager.state
                )
                print_respondent(
                    evaluation["response"].value.upper(),
                    evaluation["reasoning"],
                    evaluation["rule_applied"],
                )

                if evaluation["trap_detected"]:
                    print_trap(evaluation["trap_analysis"])

                # Update state
                manager.state.add_response(
                    proposal["proposition"],
                    evaluation["response"],
                    evaluation["reasoning"],
                    evaluation["rule_applied"],
                )

                # Check consistency
                # Build full commitment set: CONCEDO + negation of NEGO
                all_commitments = manager.state.get_all_commitments()
                all_negations = manager.state.get_all_negations()

                # Add negated versions of NEGO responses to the commitment set
                full_commitment_set = all_commitments.copy()
                for negated_prop in all_negations:
                    full_commitment_set.add(f"NOT({negated_prop})")

                if len(full_commitment_set) > 1:
                    consistent, contradictions, reasoning = (
                        manager.inference_engine.check_consistency(full_commitment_set)
                    )

                    if not consistent:
                        contradiction_found = True
                        print_contradiction()
                        click.echo(
                            f"{Colors.RED}Contradictions:{Colors.RESET} {contradictions}"
                        )
                        manager.state.history[-1].consistency_maintained = False
                        break
                    else:
                        manager.state.history[-1].consistency_maintained = True

        # End timing
        manager.end_time = __import__("datetime").datetime.utcnow()

        # Determine outcome
        if contradiction_found:
            winner = "OPPONENT"
            reason = "Respondent fell into contradiction"
        else:
            winner = "RESPONDENT"
            reason = "Maintained consistency through all turns"

        manager.state.end_disputation(winner, reason)

        # Print final result
        print_judgment({
            "winner": winner,
            "reason": reason,
            "overall_assessment": f"Disputation completed with {manager.state.turn_count} turns.",
            "key_moments": [],
            "rule_violations": [],
        })

        # Show timing
        duration = (manager.end_time - manager.start_time).total_seconds()
        click.echo(f"{Colors.CYAN}Duration:{Colors.RESET} {duration:.2f} seconds")
        click.echo(
            f"{Colors.CYAN}Total Turns:{Colors.RESET} {manager.state.turn_count}"
        )
    else:
        # Run in quiet mode - just show judgment
        result = manager.run_disputation(positum)
        print_judgment(result.judgment)

    # Save transcript if requested
    if output:
        output_path = Path(output)
        manager.save_transcript(str(output_path))
        click.echo(f"\n{Colors.GREEN}✓{Colors.RESET} Transcript saved to {output_path}")


@cli.command()
@click.argument("transcript_file", type=click.Path(exists=True))
def replay(transcript_file: str):
    """
    Replay a saved disputation from a transcript file.

    TRANSCRIPT_FILE is the path to a saved JSON transcript.

    Example:
        obligationes replay transcript.json
    """
    # Load the disputation
    manager = DisputationManager.from_transcript(transcript_file)

    print_header("REPLAY: OBLIGATIONES DISPUTATION")
    positum_content = manager.state.positum.content if manager.state.positum else "None"
    click.echo(f"{Colors.BOLD}Positum:{Colors.RESET} {positum_content}")
    click.echo(f"{Colors.BOLD}Total Turns:{Colors.RESET} {manager.state.turn_count}")
    click.echo(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")

    # Replay each turn
    for turn in manager.state.history:
        print_turn(turn.number + 1)
        click.echo(
            f"{Colors.BOLD}{Colors.RED}Opponent:{Colors.RESET} {turn.proposition}"
        )

        # Color based on response
        color = (
            Colors.GREEN
            if turn.response.value == "concedo"
            else Colors.RED if turn.response.value == "nego" else Colors.YELLOW
        )
        click.echo(
            f"\n{Colors.BOLD}{Colors.BLUE}Respondent:{Colors.RESET} {color}{Colors.BOLD}{turn.response.value.upper()}{Colors.RESET}"
        )
        click.echo(f"{Colors.CYAN}Reasoning:{Colors.RESET} {turn.reasoning}")
        click.echo(f"{Colors.CYAN}Rule Applied:{Colors.RESET} Rule {turn.rule_applied}")

        if not turn.consistency_maintained:
            print_contradiction()

    # Show final outcome
    print_header("FINAL OUTCOME")
    winner = manager.state.metadata.get("winner", "UNKNOWN")
    reason = manager.state.metadata.get("reason", "No reason recorded")

    winner_color = Colors.GREEN if winner == "RESPONDENT" else Colors.RED
    click.echo(
        f"{Colors.BOLD}Winner:{Colors.RESET} {winner_color}{Colors.BOLD}{winner}{Colors.RESET}"
    )
    click.echo(f"{Colors.BOLD}Reason:{Colors.RESET} {reason}")
    click.echo(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}\n")


@cli.command()
def info():
    """
    Display information about Burley's rules and the obligationes system.
    """
    print_header("BURLEY'S OBLIGATIONES RULES")

    click.echo(f"{Colors.BOLD}Overview:{Colors.RESET}")
    click.echo(
        "Obligationes were formalized logical games practiced in medieval universities"
    )
    click.echo(
        "(13th-14th centuries) where two parties engaged in structured debates to test"
    )
    click.echo("logical consistency.\n")

    click.echo(f"{Colors.BOLD}Participants:{Colors.RESET}")
    click.echo(
        f"  {Colors.BLUE}• Respondent:{Colors.RESET} Defends the positum (initial position)"
    )
    click.echo(
        f"  {Colors.RED}• Opponent:{Colors.RESET} Proposes propositions to force contradictions\n"
    )

    click.echo(f"{Colors.BOLD}Response Types:{Colors.RESET}")
    click.echo(f"  {Colors.GREEN}• CONCEDO:{Colors.RESET} Grant/accept the proposition")
    click.echo(f"  {Colors.RED}• NEGO:{Colors.RESET} Deny the proposition")
    click.echo(
        f"  {Colors.YELLOW}• DUBITO:{Colors.RESET} Doubt (neither grant nor deny)\n"
    )

    click.echo(
        f"{Colors.BOLD}Burley's Five Rules (in strict precedence order):{Colors.RESET}\n"
    )
    click.echo(
        f"  {Colors.CYAN}Rule 1:{Colors.RESET} If proposition follows from (positum + concessa) → CONCEDO"
    )
    click.echo(
        f"  {Colors.CYAN}Rule 2:{Colors.RESET} If proposition incompatible with (positum + concessa) → NEGO"
    )
    click.echo(
        f"  {Colors.CYAN}Rule 3:{Colors.RESET} If proposition follows from (positum + concessa + common knowledge) → CONCEDO"
    )
    click.echo(
        f"  {Colors.CYAN}Rule 4:{Colors.RESET} If proposition incompatible with (positum + concessa + common knowledge) → NEGO"
    )
    click.echo(
        f"  {Colors.CYAN}Rule 5:{Colors.RESET} Otherwise, respond based on truth in common knowledge\n"
    )

    click.echo(f"{Colors.BOLD}Winning Conditions:{Colors.RESET}")
    click.echo(
        f"  {Colors.RED}• Opponent wins:{Colors.RESET} If the Respondent falls into a contradiction"
    )
    click.echo(
        f"  {Colors.GREEN}• Respondent wins:{Colors.RESET} If they maintain consistency through all turns\n"
    )

    click.echo(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")


@cli.command()
def list_strategies():
    """
    List available opponent strategies.
    """
    print_header("OPPONENT STRATEGIES")

    click.echo(f"{Colors.BOLD}{Colors.CYAN}BALANCED{Colors.RESET}")
    click.echo("Mix of direct and indirect approaches. Builds toward contradictions")
    click.echo("methodically while maintaining intellectual rigor.\n")

    click.echo(f"{Colors.BOLD}{Colors.RED}AGGRESSIVE{Colors.RESET}")
    click.echo("Focus on forcing contradictions quickly. Exploits obvious tensions")
    click.echo("immediately and aims for fast victories.\n")

    click.echo(f"{Colors.BOLD}{Colors.MAGENTA}PEDAGOGICAL{Colors.RESET}")
    click.echo("Demonstrate logical principles clearly. Makes reasoning transparent")
    click.echo("and focuses on educational value over winning.\n")

    click.echo(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")


def main():
    """Entry point for the CLI."""
    try:
        cli()
    except Exception as e:
        click.echo(
            f"\n{Colors.BOLD}{Colors.RED}Error:{Colors.RESET} {str(e)}", err=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
