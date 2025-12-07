"""Fast-downward solver wrapper for PDDL planning."""

import asyncio
import glob
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .pddl_utils import extract_plan_cost, extract_solver_errors


@dataclass
class PlanResult:
    """Result from planning solver."""
    success: bool
    plan: Optional[str]
    cost: float
    time_seconds: float
    stdout: str
    stderr: str
    error_hints: list[str]


class FastDownwardSolver:
    """Wrapper for fast-downward planner."""
    
    def __init__(self, solver_path: Path, search_time_limit: int = 200):
        """Initialize solver.
        
        Args:
            solver_path: Path to downward-release directory
            search_time_limit: Max search time in seconds
        """
        self.solver_path = solver_path
        self.search_time_limit = search_time_limit
        self.script = solver_path / "fast-downward.py"
        
        if not self.script.exists():
            raise FileNotFoundError(f"fast-downward.py not found: {self.script}")
    
    async def solve(
        self,
        domain_file: Path,
        problem_file: Path,
        output_dir: Path,
    ) -> PlanResult:
        """Run planner to solve a problem.
        
        Args:
            domain_file: Path to domain PDDL file
            problem_file: Path to problem PDDL file
            output_dir: Directory for output files
            
        Returns:
            PlanResult with solution or error info
        """
        plan_file = output_dir / "plan"
        sas_file = output_dir / "output.sas"
        
        cmd = self._build_command(domain_file, problem_file, plan_file, sas_file)
        
        start = time.time()
        stdout, stderr = await self._run_subprocess(cmd)
        elapsed = time.time() - start
        
        # Save logs
        self._save_logs(output_dir, stdout, stderr)
        
        # Parse results
        plan, cost = self._parse_plans(plan_file)
        
        if plan:
            return PlanResult(
                success=True,
                plan=plan,
                cost=cost,
                time_seconds=elapsed,
                stdout=stdout,
                stderr=stderr,
                error_hints=[],
            )
        else:
            hints = extract_solver_errors(stdout, stderr)
            return PlanResult(
                success=False,
                plan=None,
                cost=1e10,
                time_seconds=elapsed,
                stdout=stdout,
                stderr=stderr,
                error_hints=hints,
            )
    
    def _build_command(
        self,
        domain_file: Path,
        problem_file: Path,
        plan_file: Path,
        sas_file: Path,
    ) -> list[str]:
        """Build fast-downward command line."""
        return [
            "python",
            str(self.script),
            "--alias", "seq-opt-fdss-1",
            "--search-time-limit", str(self.search_time_limit),
            "--plan-file", str(plan_file),
            "--sas-file", str(sas_file),
            str(domain_file),
            str(problem_file),
        ]
    
    @staticmethod
    async def _run_subprocess(cmd: list[str]) -> tuple[str, str]:
        """Run subprocess and capture output."""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        return (
            stdout_bytes.decode('utf-8', errors='replace'),
            stderr_bytes.decode('utf-8', errors='replace'),
        )
    
    @staticmethod
    def _save_logs(output_dir: Path, stdout: str, stderr: str) -> None:
        """Save solver logs to files."""
        (output_dir / "solver_stdout.log").write_text(stdout)
        (output_dir / "solver_stderr.log").write_text(stderr)
    
    @staticmethod
    def _parse_plans(plan_file: Path) -> tuple[Optional[str], float]:
        """Parse plan files and return best plan.
        
        Args:
            plan_file: Base path for plan files (may have .1, .2, etc. suffixes)
            
        Returns:
            Tuple of (plan_text, cost) or (None, 1e10) if no plan found
        """
        best_cost = 1e10
        best_plan = None
        
        for path_str in glob.glob(str(plan_file) + "*"):
            path = Path(path_str)
            lines = path.read_text().splitlines()
            
            if not lines:
                continue
                
            cost = extract_plan_cost(lines[-1])
            if cost < best_cost:
                best_cost = cost
                best_plan = "\n".join(line.strip() for line in lines[:-1])
        
        return best_plan, best_cost

