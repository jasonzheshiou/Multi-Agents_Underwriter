#!/usr/bin/env python3
"""CLI entry point for running test questionnaires.

Loads a YAML questionnaire definition, runs the multi-agent underwriting
pipeline, logs all decisions to JSONL, generates a console summary,
and saves results to a JSON file.

Usage:
    python demo/test_questionnaire_cli.py --questionnaire data/test_questionnaires/standard.yaml
    python demo/test_questionnaire_cli.py --questionnaire data/test_questionnaires/high_risk.yaml --agents MedicalAgent,ComplianceAgent
    python demo/test_questionnaire_cli.py --help
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Ensure the src directory is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from underwriting.test_questionnaire.engine import TestQuestionnaireEngine

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("test_questionnaire_cli")

# Suppress noisy agent-level debug logs during demo
logging.getLogger("underwriting.agents").setLevel(logging.WARNING)
logging.getLogger("underwriting.rules").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        prog="test_questionnaire",
        description=(
            "Run the multi-agent underwriting pipeline on a "
            "test questionnaire."
        ),
        epilog=(
            "Examples:\n"
            "  python demo/test_questionnaire_cli.py --questionnaire data/test_questionnaires/standard.yaml\n"
            "  python demo/test_questionnaire_cli.py --questionnaire data/test_questionnaires/high_risk.yaml --agents MedicalAgent,ComplianceAgent\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--questionnaire",
        type=str,
        required=True,
        help="Path to the YAML questionnaire file.",
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=None,
        help=(
            "Comma-separated list of agent names to run. "
            "Valid: MedicalAgent, FinancialAgent, ComplianceAgent. "
            "If not specified, runs all agents."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./audit_logs",
        help="Directory for JSONL audit logs (default: ./audit_logs)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="./audit_reports",
        help="Directory for audit reports (default: ./audit_reports)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/test_questionnaires/results",
        help="Directory for JSON result files (default: data/test_questionnaires/results)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Entry point for the test questionnaire CLI.

    Parses CLI arguments, loads the questionnaire, runs the full pipeline,
    and prints a console summary.

    Args:
        argv: Optional argument list for testing.
    """
    args = parse_args(argv)

    # Parse agent names
    agent_names = None
    if args.agents:
        agent_names = [a.strip() for a in args.agents.split(",") if a.strip()]

    print(f"[LOAD] Loading questionnaire from: {args.questionnaire}")

    # Initialize engine
    engine = TestQuestionnaireEngine(
        questionnaire_path=args.questionnaire,
        log_dir=args.log_dir,
        report_dir=args.report_dir,
    )

    # Load and display questionnaire info
    qd = engine.load()
    benefit_values = [bt.value if hasattr(bt, "value") else str(bt) for bt in qd.benefit_types]
    print(f"[OK] Questionnaire: {qd.name}")
    print(f"     Applicant: {qd.full_name}")
    print(f"     Benefits: {', '.join(benefit_values)}")

    if agent_names:
        print(f"     Agents: {', '.join(agent_names)}")
    else:
        print("     Agents: MedicalAgent, FinancialAgent, ComplianceAgent (all)")

    # Run pipeline
    print("\n[RUN] Running underwriting pipeline...")
    results = engine.run(agent_names=agent_names)
    print("[OK] Pipeline complete.")

    # Get and print console summary
    app = engine.load().to_application()
    summary = engine.get_console_summary(app, results)
    print(summary)

    # Print result file path
    print(f"[RPT] Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
