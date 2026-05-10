from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, get_args
from typing import Literal, get_args, Optional, Callable, Awaitable
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage
from langchain_core.tools import StructuredTool
from dataclasses import dataclass, field

TinyTaskStatus = Literal["todo", "in_progress", "completed", "failed"]

@dataclass
class _TinyEvent:
    """
    Records a state transition for a task.
    
    Attributes:
        old_status: The status before this event occurred.
        new_status: The status after this event.
        timestamp: The time of the update.
        outcome: A textual description of what was achieved or why the state changed.
    """
    old_status: TinyTaskStatus | None
    new_status: TinyTaskStatus
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    outcome: str | None = None

@dataclass
class _TinyTask:
    """
    A single unit of work within the agent's plan.
    
    Attributes:
        id: Unique identifier used by the LLM to reference this task.
        content: The actual instruction or description of the task.
        status: Current state of the task.
        outcome: The final result or findings (if completed).
        history: A log of all transitions this task has undergone.
    """
    id: int
    content: str
    status: TinyTaskStatus = "todo"
    outcome: str | None = None
    history: list[_TinyEvent] = field(default_factory=list)

class _TinyPlanner:
    """
    Manages a list of tasks and provides the tool definitions for LLM interaction.
    
    This class acts as the 'brain' for the agent's persistence, allowing it to
    track long-horizon goals without getting lost in the conversation history.
    """
    def __init__(self):
        self.last_id = 0
        self.task_list = []

    def add_tasks(self, tasks: list[str]) -> str:
        """
        Adds multiple new tasks to the plan and initializes their histories in one go.

        Args:
            tasks: A list of descriptions for what needs to be done.

        Returns:
            A combined confirmation string for the LLM.
        """
        new_ids = []
        
        for content in tasks:
            self.last_id += 1
            
            creation_event = _TinyEvent(
                old_status=None,
                new_status="todo",
                outcome="Task created"
            )
            
            new_task = _TinyTask(
                id=self.last_id,
                content=content,
                history=[creation_event]
            )
            self.task_list.append(new_task)
            new_ids.append(f"#{self.last_id}")

        return f"Successfully added {len(tasks)} tasks: {', '.join(new_ids)}"

    def update_task(self, id: int, new_status: TinyTaskStatus, outcome: str | None = None)->str:
        """
        Updates task status and appends a transition event to its history.

        Args:
            id: The unique task ID to update.
            new_status: The new state (todo, in_progress, completed, failed).
            outcome: Optional details about the result or reason for the change.

        Returns:
            A success message or an error string if the ID or status is invalid.
        """
        task = next((t for t in self.task_list if t.id == id), None)
        
        if not task:
            return f"Error: Task #{id} not found."

        allowed = list(get_args(TinyTaskStatus))
        if new_status not in allowed:
            return f"Error: Invalid status '{new_status}'. Use one of {allowed}."
            
        event = _TinyEvent(
            old_status=task.status,
            new_status=new_status,
            outcome=outcome
        )
        
        task.history.append(event)
        task.status = new_status
        task.outcome = outcome
            
        return f"Task #{id} updated to {new_status}."

    def show_tasks(self, include_history: bool = False):
        """
        Renders the current plan as a human-readable (and LLM-readable) string.

        Args:
            include_history: If True, renders the full audit trail for every task.

        Returns:
            A formatted string block representing the current state of work.
        """
        if not self.task_list:
            return "No tasks in the current plan."
        
        lines = ["--- CURRENT PLAN ---"]
        for task in self.task_list:
            lines.append(f"#{task.id} [{task.status.upper()}] {task.content}")
            
            if include_history:
                for event in task.history:
                    old = event.old_status if event.old_status else "none"
                    new = event.new_status
                    
                    outcome_str = f" (Outcome: {event.outcome})" if event.outcome else ""
                    lines.append(f"  └─ [{event.timestamp}] {old} -> {new}{outcome_str}")
            elif task.outcome:
                lines.append(f"  > Latest Outcome: {task.outcome}")
        
        return "\n".join(lines)

class TinyPlannerMiddleware(AgentMiddleware):
    """A LangChain compatible middleware to add a structured planner to agents"""
    def __init__(self, planner: Optional[_TinyPlanner] = None):
        super().__init__()
        self.planner = planner or _TinyPlanner()
        
        self.tools = [
            StructuredTool.from_function(
                name="add_task_to_plan",
                func=self.planner.add_tasks,
                description="Add new structured tasks to your plan."
            ),
            StructuredTool.from_function(
                name="update_task_status",
                func=self.planner.update_task,
                description="Update the status and outcome of an existing task by ID."
            )
        ]

    def _inject_plan(self, request: ModelRequest) -> ModelRequest:
        current_plan = self.planner.show_tasks(include_history=False)
        instruction = (
            f"\n\nCRITICAL: You are using a structured planner.\n{current_plan}\n"
            "Plan ahead! Use 'add_tasks' to batch multiple steps at once. "
            "Always update the status of a task immediately after completing it."
        )
        
        if request.system_message:
            existing_content = request.system_message.content
            new_content = f"{existing_content}\n{instruction}"
            return request.override(system_message=SystemMessage(content=new_content))
        
        return request.override(system_message=SystemMessage(content=instruction))

    def wrap_model_call(
        self, 
        request: ModelRequest, 
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        return handler(self._inject_plan(request))

    async def awrap_model_call(
        self, 
        request: ModelRequest, 
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        return await handler(self._inject_plan(request))

