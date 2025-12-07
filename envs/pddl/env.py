"""PDDL Environment for LLM+P planning.

This environment loads PDDL domains, manages planning tasks, calls fast-downward
solver, and evaluates plan quality.
"""

from pathlib import Path
from typing import List, Optional

from base.environment import Env
from utils.pddl_loader import DomainData, DomainLoader
from utils.pddl_solver import FastDownwardSolver, PlanResult


class PDDLEnv(Env):
    """PDDL planning environment.
    
    Manages PDDL domain files, planning tasks, and fast-downward solver calls.
    Agent provides PDDL problem files, environment validates and solves them.
    """
    
    def __init__(self, domain_name: str = "barman", logger=None):
        """Initialize PDDL environment.
        
        Args:
            domain_name: Name of PDDL domain (e.g., "barman", "blocksworld")
            logger: Optional logger instance
        """
        self.domain_name = domain_name.lower()
        self.logger = logger
        
        # State tracking
        self._step_count = 0
        self._done = False
        self._success = False
        
        # Paths
        base_path = Path(__file__).parent
        domain_path = base_path / "domains" / self.domain_name
        solver_path = base_path / ".." / ".." / "support" / "downward-release-22.06.1"
        
        # Load domain data
        loader = DomainLoader(domain_path)
        self.domain: DomainData = loader.load()
        self.domain_path = domain_path
        
        # Initialize solver
        self.solver = FastDownwardSolver(solver_path)
        
        # Current task state
        self.current_task_id: Optional[int] = None
        self.current_task_nl: Optional[str] = None
        self.current_task_pddl: Optional[str] = None
        
        # Planning results
        self.last_result: Optional[PlanResult] = None
        
        self._log_init()
        
    def _log_init(self) -> None:
        """Log initialization info."""
        if self.logger:
            count = len(self.domain.task_files)
            self.logger.info(
                f"Loaded domain '{self.domain_name}' with {count} tasks"
            )
    
    def reset(self, running_config: dict, id: Optional[str] = None) -> dict:
        """Reset environment with a new task.
        
        Args:
            running_config: Configuration dict (unused currently)
            id: Optional task id (as string)
            
        Returns:
            Dict with observations, domain info, and context
        """
        self._step_count = 0
        self._done = False
        self._success = False
        self.last_result = None
        
        # Determine task id
        task_id = self._parse_task_id(id)
        self._validate_task_id(task_id)
        
        # Load task files
        self.current_task_id = task_id
        self._load_current_task(task_id)
        
        self.id = f"pddl-{self.domain_name}-t{task_id}"
        
        if self.logger:
            nl_file = self.domain.task_files[task_id][0]
            self.logger.info(f"Reset: domain={self.domain_name}, task={task_id} ({nl_file})")
        
        return self._build_observation()
    
    def _parse_task_id(self, id: Optional[str]) -> int:
        """Parse task id from string or return 0."""
        if id is None:
            return 0
        try:
            return int(id)
        except (ValueError, TypeError):
            return 0
    
    def _validate_task_id(self, task_id: int) -> None:
        """Validate task id is in range."""
        if task_id >= len(self.domain.task_files):
            raise ValueError(
                f"Task ID {task_id} out of range (0-{len(self.domain.task_files)-1})"
            )
    
    def _load_current_task(self, task_id: int) -> None:
        """Load task files into current state."""
        nl_file, pddl_file = self.domain.task_files[task_id]
        self.current_task_nl = (self.domain_path / nl_file).read_text().strip()
        self.current_task_pddl = (self.domain_path / pddl_file).read_text().strip()
    
    def _build_observation(self) -> dict:
        """Build observation dict for agent."""
        context = None
        if self.domain.context:
            context = (
                self.domain.context.nl,
                self.domain.context.pddl,
                self.domain.context.solution,
            )
        
        return {
            "observations": [self.current_task_nl],
            "env_name": "pddl",
            "domain_name": self.domain_name,
            "domain_nl": self.domain.nl,
            "domain_pddl": self.domain.pddl,
            "context": context,
            "task_id": self.current_task_id,
            "task_type": self.domain_name.upper(),
        }
    
    async def run(self, action: List[str]) -> List[str]:
        """Execute planning actions and return [result, stdout, stderr].
        
        Args:
            action: List of PDDL problem file content
            
        Returns:
            List with three strings: [result, stdout, stderr]
        """
        if isinstance(action, str):
            action = [action]
        
        if not action:
            return []
        
        # Execute first action (PDDL env typically only uses one action)
        result_text = await self._run(action[0])
        
        # Return as [result, stdout, stderr]
        if self.last_result:
            return [result_text, self.last_result.stdout, self.last_result.stderr]
        return [result_text, "", ""]
    
    async def _run(self, action: str) -> str:
        """Execute one planning step.
        
        Args:
            action: Generated PDDL problem file content
            
        Returns:
            Result string (plan or error message)
        """
        self._step_count += 1
        
        # Prepare working directory
        work_dir = self._create_work_dir()
        problem_file = work_dir / "problem.pddl"
        problem_file.write_text(action)
        
        domain_file = self.domain_path / "domain.pddl"
        
        # Call solver
        self._log_solving()
        self.last_result = await self.solver.solve(domain_file, problem_file, work_dir)
        
        # Update state based on result
        self._update_state_from_result(self.last_result)
        
        # Log and return
        self._log_result(self.last_result)
        return self._format_result_message(self.last_result)
    
    def _create_work_dir(self) -> Path:
        """Create temporary working directory for solver."""
        if self.logger:
            base = Path(self.logger.get_base_dir())
        else:
            base = Path("logs") / "temp"
        
        work_dir = base / f"pddl_temp_{self.domain_name}_t{self.current_task_id}"
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir
    
    def _log_solving(self) -> None:
        """Log solver invocation."""
        if self.logger:
            self.logger.info(
                f"Solving: domain={self.domain_name}, task={self.current_task_id}"
            )
    
    def _update_state_from_result(self, result: PlanResult) -> None:
        """Update environment state based on solver result."""
        self._success = result.success
        self._done = True
    
    def _log_result(self, result: PlanResult) -> None:
        """Log solver result."""
        if not self.logger:
            return
        
        if result.success:
            self.logger.info(
                f"Plan found! Cost: {result.cost}, Time: {result.time_seconds:.2f}s"
            )
        else:
            hints = " | ".join(result.error_hints) if result.error_hints else "Check logs"
            self.logger.warning(
                f"No solution. Time: {result.time_seconds:.2f}s | {hints}"
            )
    
    @staticmethod
    def _format_result_message(result: PlanResult) -> str:
        """Format result as human-readable message."""
        if result.success:
            return f"SUCCESS: Plan found with cost {result.cost}\n{result.plan}"
        
        hints = " | ".join(result.error_hints) if result.error_hints else "Check solver logs"
        return f"ERROR: No solution found. Time: {result.time_seconds:.2f}s | {hints}"
    
    def report(self) -> dict:
        """Generate report dict with planning metrics."""
        cost = 0.0
        time_sec = 0.0
        
        if self.last_result:
            cost = self.last_result.cost if self._success else 0.0
            time_sec = self.last_result.time_seconds
        
        return {
            "success": self._success,
            "steps": self._step_count,
            "cost": cost,
            "time": time_sec,
            "task_type": self.domain_name.upper(),
            "task_id": self.current_task_id,
        }
