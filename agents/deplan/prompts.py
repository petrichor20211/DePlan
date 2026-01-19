"""Prompt builders for DePlan PDDL generation."""


def pddl_nl_desc_to_pddl_prompt(
    task_nl: str,
    domain_pddl: str,
    domain_nl: str,
    domain_name: str
) -> str:
    """Build prompt for PDDL generation without context.

    Args:
        task_nl: Natural language task description
        domain_pddl: PDDL domain file content
        domain_nl: Natural language domain explanation
        domain_name: Domain name extracted from PDDL

    Returns:
        Prompt string
    """
    return (
        f"Here is the complete PDDL domain definition:\n\n"
        f"{domain_pddl}\n\n"
        f"Domain explanation: {domain_nl}\n\n"
        f"Now consider a planning problem. "
        f"The problem description is:\n{task_nl}\n\n"
        f"Provide me with the problem PDDL file that describes "
        f"the planning problem directly without further explanations. "
        f"You MUST use the exact domain name '{domain_name}' "
        f"and the exact predicate names as defined in the domain PDDL above. "
        f"Only return the PDDL file. Do not return anything else."
    )


def demand_to_pddl_prompt(
    demand: str,
    domain_pddl: str,
    domain_nl: str,
    domain_name: str
) -> str:
    """Build prompt for demand-to-PDDL problem conversion.

    Args:
        demand: Natural language demand (high-level abstract user need)
        domain_pddl: PDDL domain file content
        domain_nl: Natural language domain explanation
        domain_name: Domain name extracted from PDDL

    Returns:
        Prompt string for LLM
    """
    return (
        f"Here is the complete PDDL domain definition:\n\n"
        f"{domain_pddl}\n\n"
        f"Domain explanation: {domain_nl}\n\n"
        f"Now consider a user demand: \"{demand}\"\n\n"
        f"Provide me with the problem PDDL file that corresponds to "
        f"this user demand. The demand is a high-level abstract need, "
        f"so you need to translate it into a concrete planning problem "
        f"with appropriate objects, initial state, and goal.\n\n"
        f"You MUST use the exact domain name '{domain_name}' "
        f"and the exact predicate names as defined in the domain PDDL above.\n\n"
        f"Only return the PDDL file. Do not return anything else."
    )


def prepare_incontext_example(
    context_nl: str,
    context_pddl: str,
    context_sol: str
) -> str:
    # TODO: Implement formatting logic for in-context example
    return ""
