"""Test script to verify solver returns detailed information."""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.pddl.env import PDDLEnv


MOCK_PDDL = """(define (problem blocks-problem)
  (:domain blocksworld)
  (:objects b1 b2 b3)
  (:init (on b2 b3) (on b3 b1) (ontable b1) (clear b2) (armempty))
  (:goal (and (on b2 b3) (on b3 b1)))
)"""


async def test_solver_info():
    """Test if solver returns detailed information."""
    # Initialize and reset environment
    env = PDDLEnv(domain_name="blocksworld")
    env.reset({"domain_name": "blocksworld", "task_id": 0}, id="0")
    
    print(f"\n[Test] Solver Info Output\n{MOCK_PDDL}\n")
    
    # Call solver
    observations = await env.run([MOCK_PDDL])
    result = observations if observations else "No output"
    
    print(f"[Result] {result}")

if __name__ == "__main__":
    asyncio.run(test_solver_info())

