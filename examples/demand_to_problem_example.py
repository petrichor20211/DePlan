"""Example: Generate PDDL problem file from natural language demand."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm import AsyncLLM
from agents.deplan.prompts import demand_to_pddl_prompt
from utils.common import extract_domain_name, parse_pddl_from_response


async def main():
    """Example: Generate PDDL problem from demand."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate PDDL problem from demand")
    parser.add_argument("--domain", default="barman", help="Domain name")
    parser.add_argument("--problem", default="01", help="Problem number")
    parser.add_argument("--profile", default="deepseek", help="LLM profile")
    parser.add_argument("--save", action="store_true", help="Save generated problem to file")
    args = parser.parse_args()

    # Construct file paths
    base_path = Path(__file__).parent.parent / "envs" / "pddl" / "domains" / args.domain
    demand_file = base_path / f"p{args.problem}.dm"
    domain_file = base_path / "domain.pddl"
    domain_nl_file = base_path / "domain.nl"

    # Check if demand file exists
    if not demand_file.exists():
        print(f"Error: Demand file not found: {demand_file}")
        print(f"Please run demand_to_pddl_example.py with --save first to generate the demand file.")
        sys.exit(1)

    # Load demand
    with open(demand_file, 'r', encoding='utf-8') as f:
        demand = f.read().strip()

    # Load domain PDDL
    with open(domain_file, 'r', encoding='utf-8') as f:
        domain_pddl = f.read()

    # Load domain explanation if available
    domain_nl = ""
    if domain_nl_file.exists():
        with open(domain_nl_file, 'r', encoding='utf-8') as f:
            domain_nl = f.read().strip()

    # Extract domain name
    domain_name = extract_domain_name(domain_pddl)

    print(f"Domain: {domain_name}")
    print(f"Demand: \"{demand}\"\n")

    # Create LLM client
    llm = AsyncLLM(args.profile)

    # Generate prompt
    prompt = demand_to_pddl_prompt(demand, domain_pddl, domain_nl, domain_name)

    # Query LLM
    print("Generating PDDL problem file...")
    response, cost = await llm(prompt)

    # Parse PDDL from response
    problem_pddl = parse_pddl_from_response(response)

    print(f"\nGenerated Problem PDDL:\n{'='*60}")
    print(problem_pddl)
    print(f"{'='*60}\n")

    print(f"LLM Cost: ${cost:.4f}")

    # Save to file if requested
    if args.save:
        problem_file = base_path / f"p{args.problem}_generated.pddl"
        with open(problem_file, 'w', encoding='utf-8') as f:
            f.write(problem_pddl)
        print(f"\n✓ Problem PDDL saved to: {problem_file}")


if __name__ == "__main__":
    asyncio.run(main())
