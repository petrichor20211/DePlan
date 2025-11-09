"""LLM PDDL Planner Agent.

Translates natural language task descriptions into PDDL problem files
using LLM, then relies on environment to solve using classical planner.
"""

from typing import Dict, List, Optional, Tuple

from base.agent import Agent
from utils.llm import AsyncLLM


class LLMPDDLAgent(Agent):
    """Agent that uses LLM to generate PDDL problem files.
    
    Supports two modes:
    - llm_pddl: Direct NL -> PDDL translation without context
    - llm_ic_pddl: Translation with in-context example
    """
    
    def __init__(self, use_context: bool = False, logger=None):
        """Initialize LLM PDDL agent.
        
        Args:
            use_context: If True, use in-context learning (llm_ic_pddl)
            logger: Optional logger instance
        """
        self.use_context = use_context
        self.logger = logger
        
        # LLM client (initialized on reset with profile)
        self.llm_client: Optional[AsyncLLM] = None
        
        # Domain info (set during reset)
        self.domain_nl: Optional[str] = None
        self.domain_pddl: Optional[str] = None
        self.context: Optional[Tuple[str, str, str]] = None  # (nl, pddl, sol)
        
        # Statistics
        self.num_queries = 0
        self.total_cost = 0.0
        
    def reset(self, running_config: dict, init_info: dict = None) -> None:
        """Reset agent for new task.
        
        Args:
            running_config: Config dict with 'profile' for LLM
            init_info: Environment info with domain_nl, domain_pddl, context
        """
        # Initialize LLM client
        profile = running_config.get("profile", "default")
        self.llm_client = AsyncLLM(profile)
        
        # Load domain info from environment
        if init_info:
            self.domain_nl = init_info.get("domain_nl", "")
            self.domain_pddl = init_info.get("domain_pddl", "")
            self.context = init_info.get("context")
            
        # Reset stats
        self.num_queries = 0
        self.total_cost = 0.0
        
        if self.logger:
            mode = "with context" if self.use_context else "without context"
            self.logger.info(f"LLMPDDLAgent reset (mode: {mode}, profile: {profile})")
            
    async def act(self, observations: List[str]) -> List[str]:
        """Generate PDDL problem file from task description.
        
        Args:
            observations: List with natural language task description
            
        Returns:
            List with generated PDDL problem file content
        """
        if not observations:
            if self.logger:
                self.logger.warning("Empty observations received")
            return [""]
            
        task_nl = observations[0]
        
        try:
            # Build prompt
            if self.use_context and self.context:
                prompt = self._build_llm_ic_pddl_prompt(task_nl)
            else:
                prompt = self._build_llm_pddl_prompt(task_nl)
                
            if self.logger:
                self.logger.info(f"Querying LLM for PDDL generation...")
                
            # Query LLM (AsyncLLM uses __call__)
            response, cost = await self.llm_client(prompt)
            self.num_queries += 1
            self.total_cost += cost
                
            # Parse PDDL from response
            pddl_problem = self._parse_pddl(response)
            
            if self.logger:
                self.logger.info(f"Generated PDDL ({len(pddl_problem)} chars, cost: ${cost:.4f})")
                
            return [pddl_problem]
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating PDDL: {e}")
            # Return empty PDDL on error
            return ["(define (problem error) (:domain error) (:objects) (:init) (:goal (and)))"]
        
    def _build_llm_pddl_prompt(self, task_nl: str) -> str:
        """Build prompt for PDDL generation without context.
        
        Args:
            task_nl: Natural language task description
            
        Returns:
            Prompt string
        """
        prompt = (
            f"{self.domain_nl}\n\n"
            f"Now consider a planning problem. "
            f"The problem description is:\n{task_nl}\n\n"
            f"Provide me with the problem PDDL file that describes "
            f"the planning problem directly without further explanations. "
            f"Keep the domain name consistent in the problem PDDL. "
            f"Only return the PDDL file. Do not return anything else."
        )
        return prompt
        
    def _build_llm_ic_pddl_prompt(self, task_nl: str) -> str:
        """Build prompt for PDDL generation with in-context example.
        
        Args:
            task_nl: Natural language task description
            
        Returns:
            Prompt string
        """
        if not self.context:
            # Fallback to no-context version
            return self._build_llm_pddl_prompt(task_nl)
            
        context_nl, context_pddl, context_sol = self.context
        
        prompt = (
            f"I want you to solve planning problems. "
            f"An example planning problem is:\n{context_nl}\n\n"
            f"The problem PDDL file to this problem is:\n{context_pddl}\n\n"
            f"Now I have a new planning problem and its description is:\n{task_nl}\n\n"
            f"Provide me with the problem PDDL file that describes "
            f"the new planning problem directly without further explanations? "
            f"Only return the PDDL file. Do not return anything else."
        )
        return prompt
        
    def _parse_pddl(self, response: str) -> str:
        """Parse PDDL content from LLM response.
        
        Handles cases where LLM wraps PDDL in markdown code blocks.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Extracted PDDL string
        """
        # Try to extract from markdown code block
        if "```" in response:
            # Find content between triple backticks
            parts = response.split("```")
            for i, part in enumerate(parts):
                # Look for PDDL content (skip language tags like "pddl")
                if i % 2 == 1:  # Odd indices are inside code blocks
                    # Remove language identifier if present
                    lines = part.strip().split('\n')
                    if lines and lines[0].lower() in ['pddl', 'lisp', 'scheme']:
                        return '\n'.join(lines[1:])
                    return part.strip()
                    
        # Return as-is if no code block found
        return response.strip()
        
    def report(self) -> dict:
        """Generate report with agent statistics.
        
        Returns:
            Dict with query count and cost
        """
        return {
            "llm_queries": self.num_queries,
            "llm_cost": self.total_cost,
        }

