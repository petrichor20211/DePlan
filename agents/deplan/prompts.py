"""Prompt builders for DePlan PDDL generation."""

from typing import Optional, Tuple


def build_llm_pddl_prompt(
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


def build_llm_ic_pddl_prompt(
    task_nl: str,
    domain_pddl: str,
    domain_name: str,
    context: Optional[Tuple[str, str, str]] = None
) -> str:
    """Build prompt for PDDL generation with in-context example.
    
    Args:
        task_nl: Natural language task description
        domain_pddl: PDDL domain file content
        domain_name: Domain name extracted from PDDL
        context: Optional tuple of (context_nl, context_pddl, context_sol)
        
    Returns:
        Prompt string
    """
    if not context:
        # Fallback to no-context version if context not provided
        return build_llm_pddl_prompt(task_nl, domain_pddl, "", domain_name)
        
    context_nl, context_pddl, context_sol = context
    
    return (
        f"Here is the complete PDDL domain definition:\n\n"
        f"{domain_pddl}\n\n"
        f"I want you to solve planning problems. "
        f"An example planning problem is:\n{context_nl}\n\n"
        f"The problem PDDL file to this problem is:\n{context_pddl}\n\n"
        f"Now I have a new planning problem and its description is:\n{task_nl}\n\n"
        f"Provide me with the problem PDDL file that describes "
        f"the new planning problem directly without further explanations. "
        f"You MUST use the exact domain name '{domain_name}' "
        f"and the exact predicate names as defined in the domain PDDL above. "
        f"Only return the PDDL file. Do not return anything else."
    )

