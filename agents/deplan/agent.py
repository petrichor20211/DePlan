"""DePlan Agent.

Translates natural language task descriptions into PDDL problem files
using LLM, then relies on environment to solve using classical planner.
"""

from typing import Any, Dict, List, Optional, Tuple

from base.agent import Agent
from utils.llm import AsyncLLM
from utils.common import parse_pddl_from_response
from agents.deplan.prompts import demand_to_pddl_prompt


class DePlanAgent(Agent):
    """Agent that uses LLM to generate PDDL problem files."""

    def __init__(self, logger=None):
        self.logger = logger
        
        # LLM client (initialized on reset with profile)
        self.llm_client: Optional[AsyncLLM] = None
        
        # Domain info (set during reset)
        self.domain_pddl: Optional[str] = None
        
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
            self.domain_pddl = init_info.get("domain_pddl", "")
            
        # Reset stats
        self.num_queries = 0
        self.total_cost = 0.0
        
        if self.logger:
            self.logger.info(f"DePlanAgent reset (profile: {profile})")
            
    async def act(self, observations: List[Any]) -> List[str]:
        """Generate PDDL problem file from task description.
        
        Args:
            observations: List with natural language task description as first element
            
        Returns:
            List with generated PDDL problem file content
        """
        if not observations:
            if self.logger:
                self.logger.warning("Empty observations received")
            return [""]
            
        task_nl = observations[0]
        
        try:
            # Build prompt using appropriate strategy
            prompt = self._build_prompt(task_nl)
                
            if self.logger:
                self.logger.info("Querying LLM for PDDL generation...")
                
            # Query LLM (AsyncLLM uses __call__)
            response, cost = await self.llm_client(prompt)
            self.num_queries += 1
            self.total_cost += cost
                
            # Parse PDDL from response
            pddl_problem = parse_pddl_from_response(response)
            
            if self.logger:
                self.logger.info(f"Generated PDDL ({len(pddl_problem)} chars, cost: ${cost:.4f})")
                
            return [pddl_problem]
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating PDDL: {e}")
            return ["(define (problem error) (:domain error) (:objects) (:init) (:goal (and)))"]
        
    def _split_demand_context(self, text: str) -> Tuple[str, str]:
        if "Demand:" in text and "Context:" in text:
            demand_part = text.split("Demand:", 1)[1]
            demand, context = demand_part.split("Context:", 1)
            return demand.strip(), context.strip()
        return text.strip(), ""

    def _build_prompt(self, task_text: str) -> str:
        demand, context = self._split_demand_context(task_text)
        return demand_to_pddl_prompt(
            demand=demand,
            context=context,
            domain_pddl=self.domain_pddl or "",
        )
        
    def report(self) -> dict:
        """Generate report with agent statistics.
        
        Returns:
            Dict with query count and cost
        """
        return {
            "llm_queries": self.num_queries,
            "llm_cost": self.total_cost,
        }
