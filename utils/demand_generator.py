"""Utility functions for generating natural language demands from PDDL."""

import re
from pathlib import Path
from utils.llm import AsyncLLM


def parse_pddl_files(domain_path, problem_path):
    """Read and parse PDDL files."""
    with open(domain_path, 'r') as f:
        domain_pddl = f.read()
    with open(problem_path, 'r') as f:
        problem_pddl = f.read()

    domain_match = re.search(r'\(domain\s+(\w+)\)', domain_pddl)
    domain_name = domain_match.group(1) if domain_match else "unknown"

    goal_match = re.search(r'\(:goal\s*\(([^)]+)\)', problem_pddl, re.DOTALL)
    goal = goal_match.group(1).strip() if goal_match else ""

    return {
        "domain_name": domain_name,
        "domain_pddl": domain_pddl,
        "problem_pddl": problem_pddl,
        "goal": goal
    }


async def generate_demand(domain_pddl, problem_pddl, profile):
    """Generate natural language demand from PDDL using LLM.

    Args:
        domain_pddl: Domain PDDL content
        problem_pddl: Problem PDDL content
        profile: LLM profile name

    Returns:
        Natural language demand string
    """
    prompt = f"""Generate a HIGH-LEVEL ABSTRACT natural language demand
for this PDDL planning problem. The demand should be simple, like "I'm thirsty".

=== DOMAIN ===
{domain_pddl[:1000]}

=== PROBLEM ===
{problem_pddl[:1000]}

"""

    llm = AsyncLLM(profile)
    response, cost = await llm(prompt)
    return response.strip().strip('"').strip("'")
