import argparse
import asyncio
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Type

from base.agent import Agent
from base.environment import Env


def load_class(path: str) -> Type:
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


def aggregate_metrics(instances: List[Dict[str, Any]]) -> Dict[str, Any]:
    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    metrics = [
        "solve_success",
        "pddl_context_sim_margin",
        "plan_context_sim_margin",
        "demand_context vs gen_pddl",
        "demand_context vs gen_plan",
        "demand_context vs gen_demand_context",
        "demand_context vs unrelated_pddl",
        "demand_context vs unrelated_plan",
    ]

    overall = {"count": len(instances)}
    for m in metrics:
        overall[m] = avg([float(x.get(m, 0.0)) for x in instances])

    by_bucket: Dict[str, List[Dict[str, Any]]] = {}
    for inst in instances:
        by_bucket.setdefault(inst["bucket"], []).append(inst)

    bucket_stats: Dict[str, Any] = {}
    for bucket, items in by_bucket.items():
        stats = {"count": len(items)}
        for m in metrics:
            stats[m] = avg([float(x.get(m, 0.0)) for x in items])
        bucket_stats[bucket] = stats

    return {"overall": overall, "by_bucket": bucket_stats}




async def run_one(agent: Agent, env: Env, cfg: Dict[str, Any], idx: int) -> Dict[str, Any]:
    init = env.reset(cfg, str(idx))
    obs = init["observations"]
    agent.reset(cfg, init)
    actions = await agent.act(obs)
    await env.run(actions)
    result = env.report()
    agent_report = agent.report()

    ordered = {
        "id": result.get("id"),
        "bucket": result.get("bucket"),
        "demand": result.get("demand"),
        "context": result.get("context"),
        "solve_success": result.get("solve_success", 0),
        "pddl_context_sim_margin": result.get("pddl_context_sim_margin", 0.0),
        "plan_context_sim_margin": result.get("plan_context_sim_margin", 0.0),
        "demand_context vs gen_pddl": result.get("demand_context vs gen_pddl", 0.0),
        "demand_context vs gen_plan": result.get("demand_context vs gen_plan", 0.0),
        "demand_context vs gen_demand_context": result.get("demand_context vs gen_demand_context", 0.0),
        "demand_context vs unrelated_pddl": result.get("demand_context vs unrelated_pddl", 0.0),
        "demand_context vs unrelated_plan": result.get("demand_context vs unrelated_plan", 0.0),
        "gen_plan": result.get("gen_plan", ""),
        "gen_demand_context": result.get("gen_demand_context", ""),
    }

    ordered.update(agent_report)
    ordered["success"] = bool(ordered.get("solve_success", 0))
    return ordered


async def run_all(cfg: Dict[str, Any]) -> Dict[str, Any]:
    agent_cls = load_class(cfg["agent"])
    env_cls = load_class(cfg["env"])

    agent: Agent = agent_cls()
    env: Env = env_cls()

    tasks = Path(cfg["data_path"]).read_text(encoding="utf-8").splitlines()
    total = len(tasks)
    limit = cfg.get("limit")
    if limit:
        total = min(total, int(limit))

    instances: List[Dict[str, Any]] = []
    for i in range(total):
        inst = await run_one(agent, env, cfg, i)
        instances.append(inst)

        bucket = inst["bucket"]
        task_id = inst["id"]
        pddl_out = Path("outputs") / cfg["exp_name"] / bucket / f"{task_id}.pddl"
        pddl_out.parent.mkdir(parents=True, exist_ok=True)
        pddl_out.write_text(env._gen_pddl, encoding="utf-8")
        inst["gen_pddl_path"] = str(pddl_out)

    agg = aggregate_metrics(instances)
    return {
        "exp_name": cfg["exp_name"],
        **agg,
        "instances": instances,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="agents.deplan.agent.DePlanAgent")
    parser.add_argument("--env", default="envs.debench.env.DeBenchEnv")
    parser.add_argument("--data", dest="data_path", default="envs/debench/data.jsonl")
    parser.add_argument("--domains", dest="domains_path", default="envs/debench/domains")
    parser.add_argument("--solver-path", default="support/downward-release-22.06.1")
    parser.add_argument("--exp-name", default="debench")
    parser.add_argument("--profile", default="deepseek")
    parser.add_argument("--embedding-profile", default="embedding")
    parser.add_argument("--reconstruct-profile", default="deepseek")
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    cfg = vars(args)

    result = asyncio.run(run_all(cfg))
    out_path = Path("outputs") / cfg["exp_name"] / "result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
