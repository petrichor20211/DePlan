"""Utility functions for PDDL environment."""


def postprocess(text: str) -> str:
    """Remove leading/trailing whitespace from text."""
    return text.strip()


def extract_plan_cost(plan_line: str) -> float:
    """Extract plan cost from the last line of a plan file.
    
    Args:
        plan_line: Last line containing cost info (e.g., "; cost = 10 (unit cost)")
        
    Returns:
        Plan cost, or 1e5 if not found.
    """
    tokens = plan_line.split()
    try:
        cost_idx = tokens.index("cost")
        return float(tokens[cost_idx + 2])
    except (ValueError, IndexError):
        return 1e5


def extract_solver_errors(stdout: str, stderr: str) -> list[str]:
    """Extract key error hints from solver output.
    
    Args:
        stdout: Solver standard output
        stderr: Solver standard error
        
    Returns:
        List of error hint strings
    """
    hints = []
    combined = f"{stdout.lower()} {stderr.lower()}"
    
    if "unsolvable" in combined:
        hints.append("Problem appears to be UNSOLVABLE")
    if "parse error" in combined:
        hints.append("PARSE ERROR detected")
    if "time limit" in combined:
        hints.append("TIME LIMIT exceeded")
    
    return hints

