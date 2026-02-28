import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from base.environment import Env
from utils.llm import AsyncLLM
from utils.pddl_solver import FastDownwardSolver
from utils.embeddings import EmbeddingClient, cosine_sim


UNRELATED_PDDL_TEXT = (
    "The stock market fluctuated today as investors reacted to new economic data. "
    "Meanwhile, a local sports team won their match after a dramatic overtime."
)

UNRELATED_PLAN_TEXT = (
    "(move robby bedroom bathroom)\n"
    "(clean robby bathroom)\n"
    "(move robby bathroom livingroom)\n"
)


RECONSTRUCT_PROMPT_TEMPLATE = """You are an AI Scene Interpreter.
Your task is to synthesize the following PDDL code into a single, cohesive natural language paragraph.

### INPUT PDDL:
{gen_pddl}

### INSTRUCTIONS:
1. Combine Context and Demand:
   - Start by describing the current state of the environment using the information in (:init).
   - Incorporate the descriptions found in the inline comments to ensure accuracy.
   - Transition naturally into the final goal defined in (:goal).
2. Style:
   - Use plain, professional English.
   - Avoid using PDDL terms like "predicates", "objects", or "init section".
   - The result should sound like a human briefing a robot on what to do.
3. Format: Output ONLY one paragraph of text. No headers, no lists, no JSON.

### YOUR RECONSTRUCTION:
"""


class DeBenchEnv(Env):
    id = "debench"

    def __init__(self, logger=None):
        self.logger = logger
        self._step_count = 0
        self._done = False
        self._success = False

        self._tasks: List[Dict[str, Any]] = []
        self._task: Optional[Dict[str, Any]] = None
        self._demand_context: str = ""
        self._domain_pddl: str = ""
        self._domain_nl: str = ""

        self._gen_pddl: str = ""
        self._gen_plan: str = ""
        self._gen_demand_context: str = ""

        self._metrics: Dict[str, Any] = {}

        self._solver: Optional[FastDownwardSolver] = None
        self._embedder: Optional[EmbeddingClient] = None
        self._reconstruct_llm: Optional[AsyncLLM] = None

        self._data_path: Path = Path("envs/debench/data.jsonl")
        self._domains_path: Path = Path("envs/debench/domains")
        self._exp_name: Optional[str] = None
        self._negatives_path: Path = Path("envs/debench/domains/negatives.json")
        self._negatives: Dict[str, Dict[str, str]] = {}

    def _load_tasks(self) -> None:
        if self._tasks:
            return
        lines = self._data_path.read_text(encoding="utf-8").splitlines()
        self._tasks = [json.loads(line) for line in lines if line.strip()]

    def _load_unrelated(self, bucket: str) -> tuple[str, str]:
        if bucket in self._negatives:
            neg = self._negatives[bucket]
            return neg.get("unrelated_pddl", UNRELATED_PDDL_TEXT), neg.get(
                "unrelated_plan", UNRELATED_PLAN_TEXT
            )
        if "default" in self._negatives:
            neg = self._negatives["default"]
            return neg.get("unrelated_pddl", UNRELATED_PDDL_TEXT), neg.get(
                "unrelated_plan", UNRELATED_PLAN_TEXT
            )
        return UNRELATED_PDDL_TEXT, UNRELATED_PLAN_TEXT

    def _select_task(self, id: Optional[str]) -> Dict[str, Any]:
        if id is None:
            return self._tasks[0]
        if isinstance(id, str) and id.isdigit():
            idx = int(id)
            return self._tasks[idx]
        for t in self._tasks:
            if t.get("id") == id:
                return t
        return self._tasks[0]

    def reset(self, running_config: dict, id: Optional[str] = None) -> dict:
        self._step_count = 0
        self._done = False
        self._success = False
        self._metrics = {}

        if "data_path" in running_config:
            self._data_path = Path(running_config["data_path"])
        if "domains_path" in running_config:
            self._domains_path = Path(running_config["domains_path"])
        self._exp_name = running_config.get("exp_name")
        if "negatives_path" in running_config:
            self._negatives_path = Path(running_config["negatives_path"])

        self._load_tasks()
        if self._negatives_path.exists():
            self._negatives = json.loads(self._negatives_path.read_text(encoding="utf-8")).get(
                "by_bucket", {}
            )

        self._task = self._select_task(id)

        bucket = self._task["bucket"]
        domain_path = self._domains_path / bucket / "domain.pddl"
        self._domain_pddl = domain_path.read_text(encoding="utf-8")
        self._domain_nl = ""

        demand = self._task["demand"]
        context = self._task["context"]
        self._demand_context = f"Demand: {demand}\nContext: {context}"

        solver_path = Path(running_config.get("solver_path", "support/downward-release-22.06.1"))
        self._solver = FastDownwardSolver(solver_path)

        embedding_profile = running_config.get("embedding_profile", "embedding")
        self._embedder = EmbeddingClient(profile=embedding_profile)

        reconstruct_profile = running_config.get("reconstruct_profile", "deepseek")
        self._reconstruct_llm = AsyncLLM(reconstruct_profile)

        return {
            "observations": [self._demand_context],
            "domain_pddl": self._domain_pddl,
            "domain_nl": self._domain_nl,
            "context": None,
        }

    async def _run(self, action: str) -> Any:
        self._step_count += 1
        self._gen_pddl = action or ""

        bucket = self._task["bucket"]
        task_id = self._task["id"]

        work_dir = Path("outputs") / "tmp" / bucket / task_id
        work_dir.mkdir(parents=True, exist_ok=True)
        domain_file = work_dir / "domain.pddl"
        problem_file = work_dir / "problem.pddl"
        domain_file.write_text(self._domain_pddl, encoding="utf-8")
        problem_file.write_text(self._gen_pddl, encoding="utf-8")

        plan_result = await self._solver.solve(domain_file, problem_file, work_dir)
        self._gen_plan = plan_result.plan or ""
        self._success = bool(plan_result.success)
        solve_success = 1 if plan_result.success else 0

        prompt = RECONSTRUCT_PROMPT_TEMPLATE.format(gen_pddl=self._gen_pddl)
        self._gen_demand_context, _ = await self._reconstruct_llm(prompt)

        emb_demand = await self._embedder.embed(self._demand_context)
        emb_gen_pddl = await self._embedder.embed(self._gen_pddl)
        unrelated_pddl, unrelated_plan = self._load_unrelated(bucket)

        emb_unrel_pddl = await self._embedder.embed(unrelated_pddl)
        emb_gen_plan = await self._embedder.embed(self._gen_plan)
        emb_unrel_plan = await self._embedder.embed(unrelated_plan)
        emb_gen_context = await self._embedder.embed(self._gen_demand_context)

        sim_demand_gen_pddl = cosine_sim(emb_demand, emb_gen_pddl)
        sim_demand_unrel_pddl = cosine_sim(emb_demand, emb_unrel_pddl)
        sim_demand_gen_plan = cosine_sim(emb_demand, emb_gen_plan)
        sim_demand_unrel_plan = cosine_sim(emb_demand, emb_unrel_plan)
        sim_demand_gen_context = cosine_sim(emb_demand, emb_gen_context)
        sim_demand_unrel_plan = cosine_sim(emb_demand, emb_unrel_plan)

        pddl_context_sim_margin = sim_demand_gen_pddl - sim_demand_unrel_pddl
        plan_context_sim_margin = sim_demand_gen_plan - sim_demand_unrel_plan
        self._metrics = {
            "solve_success": solve_success,
            "pddl_context_sim_margin": pddl_context_sim_margin,
            "plan_context_sim_margin": plan_context_sim_margin,
            "demand_context vs gen_pddl": sim_demand_gen_pddl,
            "demand_context vs gen_plan": sim_demand_gen_plan,
            "demand_context vs gen_demand_context": sim_demand_gen_context,
            "demand_context vs unrelated_pddl": sim_demand_unrel_pddl,
            "demand_context vs unrelated_plan": sim_demand_unrel_plan,
        }
        self._done = True
        return "done"

    def report(self) -> dict:
        return {
            "id": self._task["id"],
            "bucket": self._task["bucket"],
            "demand": self._task["demand"],
            "context": self._task["context"],
            "gen_plan": self._gen_plan,
            "gen_demand_context": self._gen_demand_context,
            **self._metrics,
        }
