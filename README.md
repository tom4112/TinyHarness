# TinyHarness

**TinyHarness** operates as a harness of [LangChain](https://www.langchain.com/)-compatible middlewares.

This approach is built on a "don't reinvent the wheel" philosophy: the LangChain ecosystem is state-of-the-art, so TinyHarness simply adds a layer of specialized middlewares to boost agent performance while maintaining 100% compatibility with existing LangChain and LangGraph features.

---
## Cognitive Toolkit

**TinyPlannerMiddleware** is a strategic reinterpretation of LangChain's [TodoListMiddleware](https://reference.langchain.com/python/langchain/agents/middleware/todo/TodoListMiddleware). While the standard implementation provides a simple checklist, TinyHarness extends this concept to handle complex, long-horizon tasks through:

- **Unique Task IDs**: Enabling precise cross-referencing and targeted updates.
- **Explicit FAILED Status**: Allows the agent to recognize when a step has crashed or yielded poor results, triggering reasoning about why it failed rather than blindly moving on.
- **Execution Outcomes**: Persisting the "what happened" for each step to ground future actions.
- **Change History**: Tracking how the plan has evolved to provide a full audit trail of the agent's logic.<br><br>

**TinyFileSystemMiddleware** is an opinionated modification of Deep Agents' [FilesystemMiddleware](https://reference.langchain.com/python/deepagents/middleware/filesystem/FilesystemMiddleware):

- **Isolation Sandbox**: The workspace can only be nested within the CWD, and file operations are strictly protected against directory traversal (```..```) security breaks.

- **Enriched Metadata**: The list of files returned by ```ls``` is accompanied by additional metadata such as file size and last modification timestamp. It is formatted as a Markdown table, making it perfectly suited for both agents and humans.

- **Proactive Context Injection**: The output of the ```ls``` tool is automatically injected into the system prompt at each conversation turn, ensuring the agent always tracks its latest context without needing to waste turns executing the tool on its own.

---

## Roadmap

TinyHarness aims to become a comprehensive agent harness by stacking specialized middlewares. The goal is to provide a complete "cognitive toolkit" that can be plugged into any LangChain agent.

As the project grows, I will continue adding custom middlewares and recommending existing community-standard middlewares to fill out the harness.

---

## Quick Start

### 1. Installation

Install the package in editable mode within your virtual environment:
```bash
git clone https://github.com/tom4112/tinyharness.git
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Basic Integration

```python
from langchain_mistralai import ChatMistralAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from tinyharness.tiny_planner import TinyPlannerMiddleware
from tinyharness.tiny_filesystem import TinyFilesystemMiddleware

# 1. Setup the model, middlewares, and checkpointer
# Assumes MISTRAL_API_KEY is in your environment
llm = ChatMistralAI(model="mistral-medium-latest")
planner_middleware = TinyPlannerMiddleware()
vfs = TinyFilesystemMiddleware(root_dir="./workspace")
checkpointer = MemorySaver()

# The config allows the agent to persist the plan across different turns
config = {"configurable": {"thread_id": "stock-research-001"}}

# 2. Create the agent with the harness middlewares
agent = create_agent(
    model=llm,
    tools=[],
    middleware=[planner_middleware, vfs],
    checkpointer=checkpointer
)

# 3. Execute a complex task
request = HumanMessage(content="""
    How would you research a new stock?
                       
    1. Create a 3-step plan in your planner
    2. Create a draft file in your workspace for each step
                       
    Cross-reference filenames in the planner.
    """)

input_data = {
    "messages": [request]
}
state = agent.invoke(input_data, config)
print(state["messages"][-1].content)
```

Outcome
```text
The draft files for all three steps of researching a new stock have been successfully created and saved in your workspace:

1. **Fundamental Analysis**: `/fundamental_analysis_draft.md`
   - Focuses on financial statements, revenue growth, profit margins, and competitive positioning.

2. **Technical Analysis**: `/technical_analysis_draft.md`
   - Covers price trends, trading volume, key indicators (e.g., moving averages, RSI), and chart patterns.

3. **Market Sentiment and News Analysis**: `/market_sentiment_draft.md`
   - Includes recent news, analyst ratings, macroeconomic factors, and investor sentiment.

All tasks in the planner are now marked as **completed**. You can review, edit, or expand these drafts as needed for your research.
```


### 3. State Inspection & Follow-up

One of the core benefits of TinyHarness is that the state is "live." You can inspect the planner at any time, even outside of the LLM conversation.

```python
# View the current high-level plan
print(planner_middleware.planner.show_tasks(include_history=False))
```

Outcome:
```text
--- CURRENT PLAN ---
#1 [COMPLETED] Create a draft file for Step 1: Fundamental Analysis of the stock, focusing on financial statements, revenue growth, profit margins, and competitive positioning. Save as `fundamental_analysis_draft.md`.
  > Latest Outcome: Successfully created the draft file for Step 1: Fundamental Analysis of the stock. Saved as `fundamental_analysis_draft.md`.
#2 [COMPLETED] Create a draft file for Step 2: Technical Analysis of the stock, focusing on price trends, trading volume, and key indicators like moving averages and RSI. Save as `technical_analysis_draft.md`.
  > Latest Outcome: Successfully created the draft file for Step 2: Technical Analysis of the stock. Saved as `technical_analysis_draft.md`.
#3 [COMPLETED] Create a draft file for Step 3: Market Sentiment and News Analysis, focusing on recent news, analyst ratings, and macroeconomic factors. Save as `market_sentiment_draft.md`.
  > Latest Outcome: Successfully created the draft file for Step 3: Market Sentiment and News Analysis. Saved as `market_sentiment_draft.md`.
```

This is equally true for the filesystem:
```python
print(vfs.ls())
```

Outcome:
```text
| path                           |   size_bytes | last_modified       |
|:-------------------------------|-------------:|:--------------------|
| /technical_analysis_draft.md   |         1876 | 2026-05-16 13:42:57 |
| /fundamental_analysis_draft.md |         2263 | 2026-05-16 13:42:57 |
| /market_sentiment_draft.md     |         2300 | 2026-05-16 13:42:57 |
```

**Dynamic Plan Adjustment**

Because we are using a Checkpointer, the agent remembers the specific plan across turns. In this example, we ask to add a new task to the plan...

```python
input_data = {
    "messages": HumanMessage(content="Add a new task to your planner to remember to ask our financial analysts about which stock they are interested in")
}
state = agent.invoke(input_data, config)
print(state["messages"][-1].content)
```

Outcome:
```text
The task **"Ask the financial analysts which specific stock they are interested in researching further."** has been added to the planner as **Task #4**. You can proceed whenever you're ready!
```

...before changing our mind:

```python
input_data = {
    "messages": HumanMessage(content="My bad, I remember now that they wanted to review NVDA!")
}
state = agent.invoke(input_data, config)
print(state["messages"][-1].content)
```

Outcome:
```text
The task has been updated to reflect that the financial analysts are interested in reviewing **NVDA (NVIDIA Corporation)**. Would you like to proceed with researching NVDA using the drafts created earlier?
```

Let's see in the next section how it reflects in our planner.

**The Audit Trail (Change History)**

For debugging and transparency, you can view the full history of every task. This shows timestamps, state transitions, and outcomes:
```python
print(planner_middleware.planner.show_tasks(include_history=True))
```

Outcome:
```text
--- CURRENT PLAN ---
#1 [COMPLETED] Create a draft file for Step 1: Fundamental Analysis of the stock, focusing on financial statements, revenue growth, profit margins, and competitive positioning. Save as `fundamental_analysis_draft.md`.
  └─ [2026-05-16 13:42:34] none -> todo (Outcome: Task created)
  └─ [2026-05-16 13:42:58] todo -> completed (Outcome: Successfully created the draft file for Step 1: Fundamental Analysis of the stock. Saved as `fundamental_analysis_draft.md`.)
#2 [COMPLETED] Create a draft file for Step 2: Technical Analysis of the stock, focusing on price trends, trading volume, and key indicators like moving averages and RSI. Save as `technical_analysis_draft.md`.
  └─ [2026-05-16 13:42:34] none -> todo (Outcome: Task created)
  └─ [2026-05-16 13:42:58] todo -> completed (Outcome: Successfully created the draft file for Step 2: Technical Analysis of the stock. Saved as `technical_analysis_draft.md`.)
#3 [COMPLETED] Create a draft file for Step 3: Market Sentiment and News Analysis, focusing on recent news, analyst ratings, and macroeconomic factors. Save as `market_sentiment_draft.md`.
  └─ [2026-05-16 13:42:34] none -> todo (Outcome: Task created)
  └─ [2026-05-16 13:42:58] todo -> completed (Outcome: Successfully created the draft file for Step 3: Market Sentiment and News Analysis. Saved as `market_sentiment_draft.md`.)
#4 [COMPLETED] Ask the financial analysts which specific stock they are interested in researching further.
  └─ [2026-05-16 13:48:18] none -> todo (Outcome: Task created)
  └─ [2026-05-16 13:51:10] todo -> completed (Outcome: The financial analysts have confirmed they want to review NVDA (NVIDIA Corporation).)
```

Notice how Task #4 transitions from todo -> completed with the outcome note "The financial analysts have confirmed they want to review NVDA (NVIDIA Corporation)." This structured history allows the agent to reason about why certain paths were abandoned in future steps.

**Edit files**

The filesystem also allows the agent to edit existing files within its sandbox:

```python
input_data = {
    "messages": HumanMessage(content="Update your 3 files to include references to NVDA now")
}
state = agent.invoke(input_data, config)
print(state["messages"][-1].content)
```

Outcome:
```text
The three draft files have been updated to specifically reference **NVDA (NVIDIA Corporation)**:

1. **Fundamental Analysis**: `/fundamental_analysis_draft.md`
   - Now focuses on NVIDIA's financial health, performance, and competitive positioning in the semiconductor and AI industries.

2. **Technical Analysis**: `/technical_analysis_draft.md`
   - Now emphasizes NVIDIA's price trends, trading volume, and key technical indicators.

3. **Market Sentiment and News Analysis**: `/market_sentiment_draft.md`
   - Now targets NVIDIA-specific news, analyst ratings, and macroeconomic factors in the semiconductor/AI industries.

You can now proceed with further customization or research for NVDA using these updated drafts.
```

A quick inspection to the filesystem state confirms the recent modifications:
```python
print(vfs.ls())
```

Outcome:
```text
| path                           |   size_bytes | last_modified       |
|:-------------------------------|-------------:|:--------------------|
| /technical_analysis_draft.md   |         2065 | 2026-05-16 13:53:44 |
| /fundamental_analysis_draft.md |         2459 | 2026-05-16 13:53:44 |
| /market_sentiment_draft.md     |         2525 | 2026-05-16 13:53:44 |
```
