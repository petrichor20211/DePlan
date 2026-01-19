"""Example: Generate natural language demand from PDDL files."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.demand_generator import parse_pddl_files, generate_demand


async def main():
    """Example: Generate demand from PDDL files."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate NL demand from PDDL")
    parser.add_argument("--domain", default="barman")
    parser.add_argument("--problem", default="01")
    parser.add_argument("--profile", default="deepseek")
    parser.add_argument("--save", action="store_true", help="Save demand to .dm file")
    args = parser.parse_args()

    base_path = Path(__file__).parent.parent / "envs" / "pddl" / "domains" / args.domain
    domain_file = base_path / "domain.pddl"
    problem_file = base_path / f"p{args.problem}.pddl"

    # Parse PDDL
    pddl_info = parse_pddl_files(domain_file, problem_file)

    print(f"Domain: {pddl_info['domain_name']}")
    print(f"Goal: {pddl_info['goal']}...\n")

    # Generate demand
    demand = await generate_demand(
        pddl_info['domain_pddl'],
        pddl_info['problem_pddl'],
        args.profile
    )

    print(f"Generated Demand: \"{demand}\"")

    # Save demand to file
    if args.save:
        demand_file = base_path / f"p{args.problem}.dm"
        with open(demand_file, 'w', encoding='utf-8') as f:
            f.write(demand)
        print(f"\n✓ Demand saved to: {demand_file}")


if __name__ == "__main__":
    asyncio.run(main())
