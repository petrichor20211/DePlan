"""PDDL Environment for LLM+P planning.

This environment loads PDDL domains, manages planning tasks, calls fast-downward
solver, and evaluates plan quality.
"""

import asyncio
import glob
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from base.environment import Env


def postprocess(text: str) -> str:
    """Remove leading/trailing whitespace from text."""
    return text.strip()


def get_plan_cost(output: str) -> float:
    """Extract plan cost from fast-downward output.
    
    Args:
        output: Last line of plan file containing cost info
        
    Returns:
        Plan cost (default 1e5 if not found)
    """
    splitted = output.split()
    counter = 0
    found = False
    cost = 1e5
    for i, token in enumerate(splitted):
        if token == "cost":
            counter = i
            found = True
            break
    if found:
        cost = float(splitted[counter + 2])
    return cost


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
        self._step_count = 0
        self._done = False
        self._success = False
        
        # Paths
        self.base_path = Path(__file__).parent
        self.domain_path = self.base_path / "domains" / self.domain_name
        self.solver_path = self.base_path / ".." / ".." / "support" / "downward-release-22.06.1"
        
        # Load domain files
        self._load_domain_files()
        
        # Load tasks
        self.tasks: List[Tuple[str, str]] = []
        self._load_tasks()
        
        # Current task state
        self.current_task_id: Optional[int] = None
        self.current_task_nl: Optional[str] = None
        self.current_task_pddl: Optional[str] = None
        
        # Planning results
        self.generated_pddl: Optional[str] = None
        self.plan: Optional[str] = None
        self.plan_cost: float = 1e10
        self.planning_time: float = 0.0
        
        # Solver output
        self.last_stdout: str = ""
        self.last_stderr: str = ""
        
    def _load_domain_files(self):
        """Load domain PDDL and natural language description."""
        # Load domain.pddl
        domain_pddl_file = self.domain_path / "domain.pddl"
        if not domain_pddl_file.exists():
            error_msg = f"Domain PDDL not found: {domain_pddl_file}"
            if self.logger:
                self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        with open(domain_pddl_file, 'r') as f:
            self.domain_pddl = postprocess(f.read())
            
        # Load domain.nl (optional)
        domain_nl_file = self.domain_path / "domain.nl"
        if domain_nl_file.exists():
            with open(domain_nl_file, 'r') as f:
                self.domain_nl = postprocess(f.read())
        else:
            self.domain_nl = ""
            
        # Load context example
        context_nl_file = self.domain_path / "p_example.nl"
        context_pddl_file = self.domain_path / "p_example.pddl"
        context_sol_file = self.domain_path / "p_example.sol"
        
        self.context = None
        if all(f.exists() for f in [context_nl_file, context_pddl_file, context_sol_file]):
            with open(context_nl_file, 'r') as f:
                context_nl = postprocess(f.read())
            with open(context_pddl_file, 'r') as f:
                context_pddl = postprocess(f.read())
            with open(context_sol_file, 'r') as f:
                context_sol = postprocess(f.read())
            self.context = (context_nl, context_pddl, context_sol)
            
    def _load_tasks(self):
        """Load all planning tasks in the domain."""
        nls = []
        for fn in glob.glob(str(self.domain_path / "*.nl")):
            fn_name = os.path.basename(fn)
            # Exclude domain.nl and p_example.nl
            if "domain" not in fn_name and "p_example" not in fn_name:
                pddl_file = fn.replace(".nl", ".pddl")
                if os.path.exists(pddl_file):
                    nls.append(fn_name)
        
        # Sort tasks by name
        sorted_nls = sorted(nls)
        self.tasks = [(nl, nl.replace(".nl", ".pddl")) for nl in sorted_nls]
        
        if self.logger:
            self.logger.info(f"Loaded {len(self.tasks)} tasks for domain '{self.domain_name}'")
            
    def reset(self, running_config: dict, id: Optional[str] = None) -> dict:
        """Reset environment with a new task.
        
        Args:
            running_config: Configuration dict with 'task_id' (int) or 'start_id' offset
            id: Optional instance id for logging
            
        Returns:
            Dict with observations, domain info, and context
        """
        self._step_count = 0
        self._done = False
        self._success = False
        self.plan = None
        self.plan_cost = 1e10
        self.planning_time = 0.0
        self.generated_pddl = None
        
        # Determine task id
        task_id = running_config.get("task_id", 0)
        if id is not None:
            try:
                task_id = int(id)
            except (ValueError, TypeError):
                pass
                
        if task_id >= len(self.tasks):
            raise ValueError(f"Task ID {task_id} out of range (0-{len(self.tasks)-1})")
            
        self.current_task_id = task_id
        nl_file, pddl_file = self.tasks[task_id]
        
        # Load task files
        with open(self.domain_path / nl_file, 'r') as f:
            self.current_task_nl = postprocess(f.read())
        with open(self.domain_path / pddl_file, 'r') as f:
            self.current_task_pddl = postprocess(f.read())
            
        self.id = f"pddl-{self.domain_name}-t{task_id}"
        
        if self.logger:
            self.logger.info(f"Reset PDDL env: domain={self.domain_name}, task={task_id} ({nl_file})")
            
        return {
            "observations": [self.current_task_nl],
            "env_name": "pddl",
            "domain_name": self.domain_name,
            "domain_nl": self.domain_nl,
            "domain_pddl": self.domain_pddl,
            "context": self.context,
            "task_id": task_id,
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
        result = await self._run(action[0])
        
        # Return as [result, stdout, stderr]
        return [result, self.last_stdout, self.last_stderr]
        
    async def _run(self, action: str) -> str:
        """Execute one planning step.
        
        Args:
            action: Generated PDDL problem file content
            
        Returns:
            Result string (plan or error message)
        """
        self._step_count += 1
        self.generated_pddl = action
        
        # Create temp directory for this run
        if self.logger:
            log_base = Path(self.logger.get_base_dir())
        else:
            log_base = Path("logs") / "temp"
            
        temp_dir = log_base / f"pddl_temp_{self.domain_name}_t{self.current_task_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Write problem PDDL file
        problem_file = temp_dir / "problem.pddl"
        with open(problem_file, 'w') as f:
            f.write(action)
            
        # Copy domain PDDL
        domain_file = self.domain_path / "domain.pddl"
        
        # Call fast-downward
        plan_file = temp_dir / "plan"
        sas_file = temp_dir / "output.sas"
        
        start_time = time.time()
        
        # Initialize stdout/stderr
        stdout_text = ""
        stderr_text = ""
        
        try:
            # Build fast-downward command
            fd_script = self.solver_path / "fast-downward.py"
            if not fd_script.exists():
                return f"ERROR: fast-downward not found at {fd_script}"
                
            cmd = [
                "python",
                str(fd_script),
                "--alias", "seq-opt-fdss-1",
                "--search-time-limit", "200",
                "--plan-file", str(plan_file),
                "--sas-file", str(sas_file),
                str(domain_file),
                str(problem_file),
            ]
            
            if self.logger:
                self.logger.info(f"Running fast-downward: {' '.join(cmd)}")
                
            # Run async subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await process.communicate()
            self.planning_time = time.time() - start_time
            
            # Save solver logs
            stdout_file = temp_dir / "solver_stdout.log"
            stderr_file = temp_dir / "solver_stderr.log"
            
            stdout_text = stdout.decode('utf-8', errors='replace')
            stderr_text = stderr.decode('utf-8', errors='replace')
            
            # Save to instance variables
            self.last_stdout = stdout_text
            self.last_stderr = stderr_text
            
            with open(stdout_file, 'w') as f:
                f.write(stdout_text)
            with open(stderr_file, 'w') as f:
                f.write(stderr_text)
            
            if self.logger:
                self.logger.info(f"Solver logs saved: {stdout_file}, {stderr_file}")
                # Log key info from solver output
                if stderr_text.strip():
                    self.logger.warning(f"Solver stderr (first 500 chars):\n{stderr_text[:500]}")
            
            # Parse plan files
            best_cost = 1e10
            best_plan = None
            
            for plan_path in glob.glob(str(plan_file) + "*"):
                with open(plan_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        cost = get_plan_cost(lines[-1])
                        if cost < best_cost:
                            best_cost = cost
                            best_plan = "\n".join([line.strip() for line in lines[:-1]])
                            
            if best_plan:
                self.plan = best_plan
                self.plan_cost = best_cost
                self._success = True
                self._done = True
                
                if self.logger:
                    self.logger.info(f"Plan found! Cost: {best_cost}, Time: {self.planning_time:.2f}s")
                    
                return f"SUCCESS: Plan found with cost {best_cost}\n{best_plan}"
            else:
                self._done = True
                # Extract key error info from solver output
                error_hints = []
                if "unsolvable" in stdout_text.lower():
                    error_hints.append("Problem appears to be UNSOLVABLE")
                if "parse error" in stdout_text.lower() or "parse error" in stderr_text.lower():
                    error_hints.append("PARSE ERROR detected")
                if "time limit" in stdout_text.lower():
                    error_hints.append("TIME LIMIT exceeded")
                    
                hint_text = " | ".join(error_hints) if error_hints else "Check solver logs for details"
                error_msg = f"No solution found. Time: {self.planning_time:.2f}s | {hint_text}"
                
                if self.logger:
                    self.logger.warning(error_msg)
                    self.logger.info(f"Full solver output saved in {temp_dir}")
                    
                return f"ERROR: {error_msg}"
                
        except Exception as e:
            self.planning_time = time.time() - start_time
            self._done = True
            error_msg = f"Planning failed: {e}"
            if self.logger:
                self.logger.error(error_msg)
            return f"ERROR: {error_msg}"
            
    def report(self) -> dict:
        """Generate report dict with planning metrics."""
        return {
            "success": self._success,
            "steps": self._step_count,
            "cost": self.plan_cost if self._success else 0.0,
            "time": self.planning_time,
            "task_type": self.domain_name.upper(),
            "task_id": self.current_task_id,
        }

