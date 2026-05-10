# TinyHarness

**TinyHarness** operates as a harness of [LangChain](https://www.langchain.com/)-compatible middlewares.

This approach is built on a "don't reinvent the wheel" philosophy: the LangChain ecosystem is state-of-the-art, so TinyHarness simply adds a layer of specialized middlewares to boost agent performance while maintaining 100% compatibility with existing LangChain and LangGraph features.

**TinyPlannerMiddleware** is a strategic reinterpretation of LangChain's [TodoListMiddleware](https://reference.langchain.com/python/langchain/agents/middleware/todo/TodoListMiddleware). While the standard implementation provides a simple checklist, TinyHarness extends this concept to handle complex, long-horizon tasks through:

- **Unique Task IDs**: Enabling precise cross-referencing and targeted updates.
- **Explicit FAILED Status**: Allows the agent to recognize when a step has crashed or yielded poor results, triggering reasoning about why it failed rather than blindly moving on.
- **Execution Outcomes**: Persisting the "what happened" for each step to ground future actions.
- **Change History**: Tracking how the plan has evolved to provide a full audit trail of the agent's logic.

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

# 1. Setup the model, middleware, and checkpointer
# Assumes MISTRAL_API_KEY is in your environment
llm = ChatMistralAI(model="mistral-medium-latest")
planner_middleware = TinyPlannerMiddleware()
checkpointer = MemorySaver()

# The config allows the agent to persist the plan across different turns
config = {"configurable": {"thread_id": "stock-research-001"}}

# 2. Create the agent with the harness middleware
agent = create_agent(
    model=llm,
    tools=[],
    middleware=[planner_middleware],
    checkpointer=checkpointer
)

# 3. Execute a complex task
input_data = {
    "messages": [HumanMessage("How would you research a new stock? Create a plan first.")]
}
state = agent.invoke(input_data, config)
print(state["messages"][-1].content)
```

Outcome
```text
The plan to research a new stock has been structured as follows:

--- **CURRENT PLAN** ---
1. **[TODO]** Gather basic information about the company (e.g., name, ticker symbol, industry, and market capitalization).
2. **[TODO]** Analyze the company's financial health by reviewing its financial statements (income statement, balance sheet, cash flow statement).
3. **[TODO]** Evaluate the company's competitive position and industry trends.
4. **[TODO]** Assess the company's management team and corporate governance practices.
5. **[TODO]** Review analyst reports and price targets for the stock.
6. **[TODO]** Analyze the stock's historical performance and valuation metrics (e.g., P/E ratio, PEG ratio, dividend yield).
7. **[TODO]** Identify potential risks and catalysts that could impact the stock's performance.
8. **[TODO]** Summarize findings and provide a recommendation or outlook on the stock.

I will proceed with executing the plan step-by-step. Let me know if you'd like to prioritize any specific task or provide additional guidance.
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
#1 [TODO] Gather basic information about the company (e.g., name, ticker symbol, industry, and market capitalization).
#2 [TODO] Analyze the company's financial health by reviewing its financial statements (income statement, balance sheet, cash flow statement).
#3 [TODO] Evaluate the company's competitive position and industry trends.
#4 [TODO] Assess the company's management team and corporate governance practices.
#5 [TODO] Review analyst reports and price targets for the stock.
#6 [TODO] Analyze the stock's historical performance and valuation metrics (e.g., P/E ratio, PEG ratio, dividend yield).
#7 [TODO] Identify potential risks and catalysts that could impact the stock's performance.
#8 [TODO] Summarize findings and provide a recommendation or outlook on the stock.
```

**Dynamic Plan Adjustment**

Because we are using a Checkpointer, the agent remembers the specific plan across turns. In this example, the user asks to cancel a specific task, and the agent uses the harness to update the status:

```python
input_data = {
    "messages": [HumanMessage("Actually, cancel task 7 please")]
}
state = agent.invoke(input_data, config)
print(state["messages"][-1].content)
```

Outcome:
```text
Task **#7** has been canceled as requested. Here’s the updated plan:

--- **CURRENT PLAN** ---
1. **[TODO]** Gather basic information about the company (e.g., name, ticker symbol, industry, and market capitalization).
2. **[TODO]** Analyze the company's financial health by reviewing its financial statements (income statement, balance sheet, cash flow statement).
3. **[TODO]** Evaluate the company's competitive position and industry trends.
4. **[TODO]** Assess the company's management team and corporate governance practices.
5. **[TODO]** Review analyst reports and price targets for the stock.
6. **[TODO]** Analyze the stock's historical performance and valuation metrics (e.g., P/E ratio, PEG ratio, dividend yield).
7. **[FAILED]** Identify potential risks and catalysts that could impact the stock's performance.
   > *Outcome: Task canceled as per user request.*
8. **[TODO]** Summarize findings and provide a recommendation or outlook on the stock.
```

**The Audit Trail (Change History)**

For debugging and transparency, you can view the full history of every task. This shows timestamps, state transitions, and outcomes:
```python
print(planner_middleware.planner.show_tasks(include_history=True))
```

Outcome:
```text
--- CURRENT PLAN ---
#1 [TODO] Gather basic information about the company (e.g., name, ticker symbol, industry, and market capitalization).
  └─ [2026-05-10 21:37:16] none -> todo (Outcome: Task created)
#2 [TODO] Analyze the company's financial health by reviewing its financial statements (income statement, balance sheet, cash flow statement).
  └─ [2026-05-10 21:37:16] none -> todo (Outcome: Task created)
#3 [TODO] Evaluate the company's competitive position and industry trends.
  └─ [2026-05-10 21:37:16] none -> todo (Outcome: Task created)
#4 [TODO] Assess the company's management team and corporate governance practices.
  └─ [2026-05-10 21:37:16] none -> todo (Outcome: Task created)
#5 [TODO] Review analyst reports and price targets for the stock.
  └─ [2026-05-10 21:37:16] none -> todo (Outcome: Task created)
#6 [TODO] Analyze the stock's historical performance and valuation metrics (e.g., P/E ratio, PEG ratio, dividend yield).
  └─ [2026-05-10 21:37:16] none -> todo (Outcome: Task created)
#7 [FAILED] Identify potential risks and catalysts that could impact the stock's performance.
  └─ [2026-05-10 21:37:16] none -> todo (Outcome: Task created)
  └─ [2026-05-10 21:40:44] todo -> failed (Outcome: Task canceled as per user request.)
#8 [TODO] Summarize findings and provide a recommendation or outlook on the stock.
  └─ [2026-05-10 21:37:16] none -> todo (Outcome: Task created)
```

Notice how Task #7 transitions from todo -> failed with the outcome note "Task canceled as per user request." This structured history allows the agent to reason about why certain paths were abandoned in future steps.
