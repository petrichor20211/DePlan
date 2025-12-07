"""Abstract base class for all environments in the system."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class Env(ABC):
    """Abstract base class for all environments.
    
    Enforces common interface for reset, run, report cycle.
    """
    
    id: str
    _step_count: int = 0
    _done: bool = False
    _success: bool = False
    
    @abstractmethod
    async def _run(self, action: str) -> Any:
        """Execute a single action in the environment.
        
        Args:
            action: Action string to execute.
            
        Returns:
            Observation from the environment.
        """
        pass
    
    async def run(self, action: List[str]) -> List[str]:
        """Execute a list of actions sequentially.
        
        Args:
            action: Single action string or list of action strings.
            
        Returns:
            List of observations from executed actions.
        """
        if isinstance(action, str):
            action = [action]
        
        if not action:
            return []
        
        observations: List[Any] = []
        for single_action in action:
            observations.append(await self._run(single_action))
            if self.is_success():
                self._done = True
            if self.is_done():
                break
        
        return observations
    
    def is_done(self) -> bool:
        """Check if the environment episode is done."""
        return self._done
    
    def is_success(self) -> bool:
        """Check if the task was completed successfully."""
        return self._success
    
    def get_step_count(self) -> int:
        """Get the number of steps taken in current episode."""
        return self._step_count
    
    @abstractmethod
    def reset(self, running_config: dict, id: Optional[str] = None) -> dict:
        """Reset environment to initial state.
        
        Args:
            running_config: Runtime configuration dictionary.
            id: Optional environment identifier.
            
        Returns:
            Initial observation dictionary.
        """
        pass
    
    @abstractmethod
    def report(self) -> dict:
        """Report environment statistics and state.
        
        Returns:
            Dictionary containing environment metrics.
        """
        pass
    
    async def close(self) -> None:
        """Clean up environment resources.
        
        Override if environment needs cleanup.
        """
        pass
