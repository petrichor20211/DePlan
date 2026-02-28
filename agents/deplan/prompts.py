"""Prompt builders for DePlan PDDL generation."""

def demand_to_pddl_prompt(
    demand: str,
    context: str,
    domain_pddl: str
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
    return ( f"""Here is the complete PDDL domain definition:

{domain_pddl}

Context information:
{context}

The task demand is: {demand}

Provide me with a CLEAN, MINIMAL, SOLVABLE PDDL problem file that describes this planning problem.

Hard constraints:
- You MUST use the exact domain name and ONLY the exact predicate names as defined in the domain PDDL above (same argument order/types).
- Output ONLY the PDDL problem file. No explanations, no extra text.
- Keep :objects minimal: include ONLY objects that are required to achieve the goal for this specific task. Do NOT include unrelated items/appliances/surfaces.

CRITICAL - DO NOT make these common mistakes:
- NEVER re-declare domain constants as objects. If the domain has (:constants X Y Z - room), do NOT add X Y Z to your :objects section.
- NEVER add redundant equality assertions like (= X X) in the :init section. These are useless and cause solver issues.
- Only declare NEW objects in :objects that are NOT already defined as constants in the domain.

COMMENT FORMAT REQUIREMENTS:
- :init section - Add inline comments. Format: ;; brief explanation of what this predicate does. The explanation should describe the PDDL predicate's purpose in natural language, NOT copy the context text verbatim. Example: (at-robot robby livingroom) ;; The robot starts in the living room
- :goal section - Add comment above using ;; brief explanation of what the goal achieves. Example: ;; The elder receives their requested medicine
- :objects section - NO comments needed, keep it clean

Only return the PDDL file. Do not return anything else."""
    )
