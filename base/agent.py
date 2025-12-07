from abc import ABC, abstractmethod
from typing import Any, List

class Agent(ABC):
    """Abstract base class for all agents in the system."""

    @abstractmethod
    async def act(self, observations: List[Any]) -> List[str]:
        """Execute actions based on observations.

        Args:
            observations: List of observations from the environment.

        Returns:
            List of action strings to be executed.
        """
        pass

    @abstractmethod
    def reset(self, running_config: dict, init_info: dict=None) -> None:
        """Reset agent state with configuration.

        Args:
            running_config: Runtime configuration dictionary.
            init_info: Optional initialization information.
        """
        pass

    @abstractmethod
    def report(self) -> dict:
        """Report agent statistics and state.

        Returns:
            Dictionary containing agent metrics and status.
        """
        pass