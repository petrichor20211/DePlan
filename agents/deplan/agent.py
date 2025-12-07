"""DePlan Agent.

Translates natural language task descriptions into PDDL problem files
using LLM, then relies on environment to solve using classical planner.
"""

from typing import Any, Dict, List, Optional, Tuple

from base.agent import Agent
from utils.llm import AsyncLLM
from utils.common import extract_domain_name, parse_pddl_from_response
from agents.deplan.prompts import build_llm_pddl_prompt, build_llm_ic_pddl_prompt


class DePlanAgent(Agent):
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
        self.domain_name: Optional[str] = None  
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
            self.domain_name = extract_domain_name(self.domain_pddl)
            self.context = init_info.get("context")
            
        # Reset stats
        self.num_queries = 0
        self.total_cost = 0.0
        
        if self.logger:
            mode = "with context" if self.use_context else "without context"
            self.logger.info(f"DePlanAgent reset (mode: {mode}, profile: {profile})")
            
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
        
    def _build_prompt(self, task_nl: str) -> str:
        """Build prompt based on agent mode.
        
        Args:
            task_nl: Natural language task description
            
        Returns:
            Prompt string
        """
        if self.use_context:
            return build_llm_ic_pddl_prompt(
                task_nl=task_nl,
                domain_pddl=self.domain_pddl,
                domain_name=self.domain_name,
                context=self.context
            )
        else:
            return build_llm_pddl_prompt(
                task_nl=task_nl,
                domain_pddl=self.domain_pddl,
                domain_nl=self.domain_nl,
                domain_name=self.domain_name
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
