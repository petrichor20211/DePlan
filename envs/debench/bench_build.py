DEMAND_GEN_PROMPT = """
You are a task data generator. Please generate data based on the given demand
buckets.
Requirements:
1) Output strict JSONL, one object per line.
2) Each object must include fields: demand, context, bucket.
3) bucket must be exactly one of the given demand buckets.
4) demand and context should be natural, concise English.
5) Generate N lines.

Demand buckets:
- Household chores and environment
- Medication and health management
- Safety and emergency response
- Daily Needs Fulfillment
- Personal care and hygiene

Example (format only, do not reuse content):
{"id":"01_water_fill_and_deliver", "demand":"I'm a bit thirsty—could you bring me a cup of water?","context":"The elder
is sitting in the living room. The robot is currently in the bedroom, while the cup
can be found in the kitchen.","bucket":"Household chores and environment"}

Now output JSONL only:
"""


DOMAIN_GEN_PROMPT = """
You are a PDDL domain engineer. You will be given a JSONL file where each line
has:
{id, demand, context, bucket}. The buckets are:
- Household chores and environment
- Medication and health management
- Safety and emergency response
- Daily Needs Fulfillment
- Personal care and hygiene

Task:
Generate PDDL STRIPS domain code for each bucket, such that the domain can
represent and solve ALL demands that belong to that bucket in the JSONL. Your
domain MUST cover the actions implied by the demands and contexts (objects,
locations, tools, states).

Requirements:
1) Output exactly five PDDL domain definitions, one per bucket.
2) Each domain should be self-contained and named clearly (e.g., (domain
elderly-homecare-household)).
3) Use STRIPS style: :requirements :strips :typing (no numeric fluents).
4) Include types, predicates, and actions needed to satisfy all demands in
that bucket.
5) Actions must have realistic preconditions and effects that make the demand
achievable.
6) Keep predicates and actions minimal but sufficient to cover all demands in
the bucket.
7) Do not include problem files.

Output format:
For each bucket, output:
- A one-line comment: ;; Bucket: <bucket name>
- The PDDL (domain ...) block
No extra text.

Now read the JSONL below and produce the five domains:
"""


async def build_demand_jsonl(llm, n: int, out_path: str) -> None:
    prompt = DEMAND_GEN_PROMPT.replace("Generate N lines.", f"Generate {n} lines.")
    response, _ = await llm(prompt)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(response.strip() + "\n")


async def build_domains(llm, jsonl_path: str, out_dir: str) -> None:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        jsonl = f.read()
    prompt = DOMAIN_GEN_PROMPT + "\n" + jsonl
    response, _ = await llm(prompt)
    with open(out_dir, "w", encoding="utf-8") as f:
        f.write(response)

