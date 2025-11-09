"""Simple test script for LLM PDDL planner."""

import asyncio
from agents.llm_pddl.agent import LLMPDDLAgent
from envs.pddl.env import PDDLEnv


async def test_single_task():
    """Test a single PDDL planning task."""
    # Configuration
    config = {
        "profile": "deepseek",
        "domain_name": "barman",
        "task_id": 0,
        "use_context": False,
    }
    
    # Initialize environment
    print("Initializing PDDL environment...")
    env = PDDLEnv(domain_name="barman")
    
    # Initialize agent
    print("Initializing LLM PDDL agent...")
    agent = LLMPDDLAgent(use_context=False)
    
    # Reset with task
    print("Resetting environment with task 0...")
    init_info = env.reset(config, id="0")
    agent.reset(config, init_info)
    
    # Get initial observation
    observations = init_info["observations"]
    print(f"\nTask description:\n{observations[0][:200]}...\n")
    
    # Agent generates PDDL
    print("Agent generating PDDL problem file...")
    actions = await agent.act(observations)
    
    print(f"Generated PDDL ({len(actions[0])} chars)")
    print(f"First 300 chars:\n{actions[0][:300]}...\n")
    
    # Environment solves PDDL
    print("Environment calling fast-downward solver...")
    observations = await env.run(actions)
    
    # Check result
    success = env.is_success()
    report = env.report()
    
    print(f"\n{'='*60}")
    print(f"Result: {'SUCCESS' if success else 'FAILURE'}")
    print(f"Plan cost: {report['cost']}")
    print(f"Planning time: {report['time']:.2f}s")
    print(f"LLM cost: ${agent.total_cost:.4f}")
    print(f"{'='*60}")
    
    if success:
        print("\nPlan found:")
        print(env.plan[:500])
    else:
        print("\nNo plan found or error occurred")
        

if __name__ == "__main__":
    asyncio.run(test_single_task())

