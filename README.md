# DePlan: LLM-based PDDL Planning

## Project Structure

```
DePlan/
├── agents/
│   └── deplan/                # DePlan agent
│       ├── agent.py           # DePlanAgent implementation
│       └── prompts/           # Prompt templates
├── envs/
│   └── pddl/                  # PDDL environment
│       ├── env.py             # PDDLEnv implementation
│       └── domains/           # PDDL domain files
│           ├── barman/
│           ├── blocksworld/
│           └── ...
├── base/                      # Base abstract classes
├── utils/                     # Utility modules
├── configs/                   # LLM configurations
├── support/
│   └── downward-release-22.06.1/  # Fast-Downward solver
└── run.py                     # Unified entry point
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Fast-Downward Solver

DePlan requires the Fast-Downward planner to solve PDDL problems.

**Download and Setup**:

```bash
# Download Fast-Downward (version 22.06.1 recommended)
mkdir -p support && cd support
wget https://github.com/aibasel/downward/releases/download/release-22.06.1/downward-release-22.06.1.tar.gz
tar -xzf downward-release-22.06.1.tar.gz
cd downward-release-22.06.1
./build.py
```

**Expected Directory Structure**:
```
DePlan/
└── support/
    └── downward-release-22.06.1/
        ├── fast-downward.py
        ├── builds/
        └── ...
```
### 3. Configure LLM

Edit `configs/profiles.yaml` with your API credentials:

```yaml
models:
  deepseek:
    api_key: "your-api-key"
    base_url: "https://api.deepseek.com/v1"
    model: "deepseek-chat"
    temperature: 0.0
```

### 4. Run Single Task Test

```bash
python scripts/test_single.py
```

### 5. Run Batch Tasks

**Without context (direct PDDL generation)**:
```bash
python run.py -a deplan -e pddl -n 5 \
  --profile deepseek \
  --domain_name barman \
  --use_context false
```

**With context (in-context learning)**:
```bash
python run.py -a deplan -e pddl -n 5 \
  --profile deepseek \
  --domain_name barman \
  --use_context true
```

**Backward compatibility (use `llm_pddl` alias)**:
```bash
python run.py -a llm_pddl -e pddl -n 5 \
  --profile deepseek \
  --domain_name barman \
  --use_context false
```

**Using config file**:
```bash
python run.py -C test_pddl.yaml
```

## Core Components

### PDDLEnv (Environment)

**Responsibilities**:
- Load PDDL domain files (domain.pddl, domain.nl)
- Manage planning tasks (p01.nl/pddl, p02.nl/pddl, etc.)
- Provide context examples (p_example.*)
- Invoke fast-downward solver
- Evaluate plan quality

**Parameters**:
- `domain_name`: PDDL domain name (e.g., "barman", "blocksworld")
- `task_id`: Task ID (0-based index)

### DePlanAgent (Agent)

**Responsibilities**:
- Convert natural language task descriptions to PDDL problem files
- Support in-context learning (optional)
- Manage LLM invocations and cost tracking

**Parameters**:
- `use_context`: Enable in-context learning (with/without examples)
- `profile`: LLM configuration name

**Aliases**:
- `deplan` (recommended)
- `llm_pddl` (backward compatibility)

## Supported PDDL Domains

- barman - Bartender robot
- blocksworld - Blocks world
- floortile - Floor tile
- grippers - Gripper robot
- storage - Storage logistics
- termes - Construction robot
- tyreworld - Tire repair
- manipulation - Object manipulation

## Method Comparison

| Mode | Context | Description |
|------|---------|-------------|
| use_context=False | No | Direct PDDL generation from task description |
| use_context=True | Yes | PDDL generation with in-context examples |

## Output

Execution generates the following in `logs/<run_id>/`:
- `running_logs/run.log` - Full execution logs
- `pddl_temp_*/` - PDDL temporary files and solver results
- `results.json` - Structured evaluation results

results.json contains:
- Success rate
- Average plan cost
- Average solving time
- LLM invocation cost
- Per-task detailed results

## Architecture Design



## References

- Original project: [llm-pddl](https://github.com/Cranial-XIX/llm-pddl)
- Architecture reference: [ReCode](https://github.com/zhaoyang-yu/ReCode)
- Solver: [Fast-Downward](https://www.fast-downward.org/)
