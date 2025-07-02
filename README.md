# Building Multi-Agent Workflows with LangGraph Workshop

Learn to build sophisticated multi-agent systems by mastering six architectural patterns plus production-ready implementation techniques.

## Workshop Philosophy

This workshop teaches LangGraph through architectural patterns, not code syntax. You'll understand **when** and **why** to choose different approaches, progressing from simple workflows to sophisticated multi-agent systems.

> [!CAUTION]
> LangGraph is a relatively recent library that is continuously updated with new syntax, not all LLMs have caught on to this so always cross-check with documentation when unsure. Ask AI to check latest documentation before generating any code.

## Prerequisites

- Python 3.9+ or Anaconda/Miniconda
- OpenAI API key
- Basic Python and AI understanding
- Review [Multi-Agent Patterns: Practical Guide](PATTERNS.md) before starting

## Quick Start

1. **Clone the repository**:

   - **with https**:

   ```bash
   git clone https://github.com/TandemCreativeDev/fac-ws_ai_multi-agent.git
   cd fac-ws_ai_multi-agent
   ```

   - **with ssh**:

   ```bash
   git clone git@github.com:TandemCreativeDev/fac-ws_ai_multi-agent.git
   cd fac-ws_ai_multi-agent
   ```

2. **Setup environment**

   - **with conda** (recommended for Mac and Linux):

   ```bash
   conda env create -f environment.yml
   conda activate multi-agent
   ```

   - **with pip only** (easier for Windows):

   ```bash
   pip install langchain-openai langgraph python-dotenv matplotlib
   ```

3. **Create `.env` file**:
   ```
   OPENAI_API_KEY=your_key_here
   ```

## How This Workshop Works

1. **Two versions per pattern**: Start with `patterns_simple/` to understand concepts, then do exercises in `patterns/`
2. **Six architectural patterns + production implementation**: Each builds on previous concepts
3. **Three exercises per file**: Modify the code to complete each exercise
4. **Generated output**: Check `generated/` folder after running the patterns
5. **Incremental Difficulty**: Each pattern builds on previous concepts
6. **Focus on concepts, not syntax**: Focus on **when** to use each pattern, not just **how**

## Approach Each Pattern

For each pattern:

1. **Read** the simple version in `patterns_simple/`
2. **Run** the full version in `patterns/`
3. **Examine** the generated output in `generated/`

   > #### Output Structure
   >
   > Full pattern implementations generate timestamped folders in `generated/`:
   >
   > ```
   > generated/
   > └── 01_sequential_workflow_20250602_143022/
   >     ├── original_code.py
   >     ├── refactored_code.py
   >     └── AUDIT_TRAIL.md
   > ```

4. **Complete** the 3 exercises by modifying the code in the file you just ran (the one in `patterns/`)
5. **Discuss** when you'd use this pattern vs others

## The Six Architectural Patterns

### [Pattern 1: Sequential Workflow](PATTERNS.md#pattern-1-sequential-workflow) 🔁👨‍💻🕵️🔧✅

- **File**: `patterns/01_sequential_workflow.py`
- **Concept**: Linear pipeline (coder → reviewer → refactorer)
- **Use case**: Predictable, step-by-step processes
- **File**: `patterns/01_sequential_workflow.py`
- **Concept**: Linear pipeline (coder → reviewer → refactorer)
- **Use case**: Predictable, step-by-step processes

**Run and explore**:

```bash
python patterns/01_sequential_workflow.py
# Check generated/ folder for output
```

#### **Your 3 Exercises** 🎯 _LangGraph Focus_ (modify the code):

1. **Add a tester agent**: Create `tester_agent` function that generates unit tests. Add node after refactorer.

> [!IMPORTANT]  
> The tester agent's state key must be `tests` for the utils function to be able to pick it up and add to the output folder.

2. **Change focus to security**: Modify all prompts to emphasise security vulnerabilities instead of general quality.
3. **Add conditional routing**: Route to different refactoring approaches based on code type (web, API, data processing).

---

### [Pattern 2: Conditional Routing](PATTERNS.md#pattern-2-conditional-routing) 🧠🛣️🎯📋

- **File**: `patterns/02_conditional_routing.py`
- **Concept**: Router analyzes content and routes to appropriate specialist
- **Use case**: Domain-specific expert selection based on code characteristics

**Run and explore**:

```bash
python patterns/02_conditional_routing.py
# Watch router decision in output
```

#### **Your 3 Exercises** 🎯 _LangGraph Focus_ (modify the code):

1. **Add database expert**: Create `database_expert_agent` and update router logic to route SQL/schema code.
2. **Smart routing**: Make router consider task description as well as code content for routing decisions.
3. **Multi-expert routing**: Allow router to send code to multiple experts when it contains mixed concerns.

---

### [Pattern 3: Parallel Processing](PATTERNS.md#pattern-3-parallel-processing) 🧠⚙️⚙️⚙️📦

- **File**: `patterns/03_parallel_processing.py`
- **Concept**: Concurrent analysis by multiple specialists
- **Use case**: Independent tasks that can run simultaneously
- **File**: `patterns/03_parallel_processing.py`
- **Concept**: Concurrent analysis by multiple specialists
- **Use case**: Independent tasks that can run simultaneously

**Run and explore**:

```bash
python patterns/03_parallel_processing.py
# Check SYNTHESIS_REPORT.md
```

#### **Your 3 Exercises** 🎯 _LangGraph Focus_ (modify the code):

1. **Add documentation agent**: Create `documentation_agent` that generates docstrings. Run in parallel.
2. **Add fallback routing**: If an expert agent fails, route to a general expert as fallback.
3. **Weighted synthesis**: Give security 2x weight in final recommendations.

---

### [Pattern 4: Supervisor Agents](PATTERNS.md#pattern-4-supervisor-agents) 🧑‍🏫🧑‍🔧🧑‍🔬🗂️

- **File**: `patterns/04_supervisor_agents.py`
- **Concept**: Intelligent coordination of specialist agents
- **Use case**: Complex tasks requiring dynamic expertise
- **File**: `patterns/04_supervisor_agents.py`
- **Concept**: Intelligent coordination of specialist agents
- **Use case**: Complex tasks requiring dynamic expertise

**Run and explore**:

```bash
python patterns/04_supervisor_agents.py
# Check EXPERT_ANALYSIS.md
```

#### **Your 3 Exercises** 🎯 _LangGraph Focus_ (modify the code):

1. **Add database expert**: Create `database_expert_agent` for SQL/schema review. Update supervisor logic.

> [!IMPORTANT]  
> The database agent's state key must be `database_report` for the utils function to be able to pick it up and add to the output folder.

2. **Smart content routing**: Make supervisor check code content (e.g., "if 'sql' in code: route to database expert").

> [!IMPORTANT]  
> The state key must be `task_type` for the utils function to be able to pick it up and add to the output folder.

3. **Expert collaboration**: Let security expert see quality report before finalising.

---

### [Pattern 5: Evaluator-Optimiser](PATTERNS.md#pattern-5-evaluator-optimiser) 🔍📈♻️🛠️

- **File**: `patterns/05_evaluator_optimiser.py`
- **Concept**: Continuous improvement through feedback loops
- **Use case**: Iteratively refinable outputs
- **File**: `patterns/05_evaluator_optimiser.py`
- **Concept**: Continuous improvement through feedback loops
- **Use case**: Iteratively refinable outputs

**Run and explore**:

```bash
python patterns/05_evaluator_optimiser.py
# Watch score progression
```

#### **Your 3 Exercises** 🎯 _LangGraph Focus_ (modify the code):

1. **Adjust thresholds**: Change `quality_threshold = 7` to 9. How many iterations now? Play around with `max_iterations` too.
2. **Add fast track**: If initial score ≥ 8, skip refactoring entirely.
3. **Multi-criteria routing**: Score separately for security, performance, readability. Route based on lowest score.

---

### [Pattern 6: Orchestrator-Worker](PATTERNS.md#pattern-6-orchestrator-worker) 🎼👷👷‍♀️📋🔗

- **File**: `patterns/06_orchestrator_worker.py`
- **Concept**: Dynamic task breakdown with isolated worker execution
- **Use case**: Complex tasks requiring unpredictable decomposition

**Run and explore**:

```bash
python patterns/06_orchestrator_worker.py
# Note dynamic worker creation
```

#### **Your 3 Exercises** 🎯 _LangGraph Focus_ (modify the code):

1. **Smart task detection**: Make orchestrator identify task type (frontend, backend, database) and assign appropriate worker types.
2. **Worker specialisation**: Create different worker agents for different task types with specialised prompts.
3. **Dependency handling**: Allow orchestrator to create subtasks with dependencies (e.g., "database schema must complete before API").

---

## Key Learning Progression

1. **Understand**: Run simple version, trace execution flow
2. **Experiment**: Complete the 4 exercises for each pattern
3. **Analyse**: Compare patterns - when would you choose each?
4. **Build**: Combine patterns for your use case
5. **Deploy**: Apply production-ready techniques from Pattern 6

## Workshop Tips

### Pattern Selection Guide

| Your Need                           | Use This Pattern      |
| ----------------------------------- | --------------------- |
| Step-by-step process                | Sequential            |
| Quality-based branching             | Conditional           |
| Speed through parallelism           | Parallel              |
| Complex coordination                | Supervisor            |
| Iterative improvement               | Evaluator             |
| Dynamic task decomposition          | Orchestrator          |
| Production deployment (any pattern) | + Production Concerns |

### Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents/)
- [LangGraph Workflows](https://langchain-ai.github.io/langgraph/tutorials/workflows/)
- [LangGraph Agent Supervisor](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)

---

## Next Steps

After completing all exercises:

1. Identify your use case's requirements
2. Select appropriate architectural pattern(s) (1-6)
3. Combine patterns if needed, add tools
4. Experiment using different models for different agents (perhaps a reasoning model for reviews etc)
5. Apply production-ready techniques
6. Deploy using [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/langgraph_platform/)

---

**Remember**: Master the patterns, then combine them creatively. Apply production concerns to make them deployment-ready. The best solution uses the simplest pattern that meets your requirements.

## Author

[**TandemCreativeDev**](https://github.com/TandemCreativeDev) - hello@runintandem.com
