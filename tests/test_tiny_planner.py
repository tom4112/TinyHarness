import pytest
from langchain_core.messages import SystemMessage
from langchain.agents.middleware import ModelRequest
from tinyharness.tiny_planner import TinyPlannerMiddleware, _TinyPlanner

@pytest.fixture
def clean_planner():
    """Provides a fresh planner instance for isolated unit testing."""
    return _TinyPlanner()

def test_planner_batch_add_tasks(clean_planner):
    """Verifies that multiple tasks can be added at once with sequential IDs and history logs."""
    confirmation = clean_planner.add_tasks([
        "Download financial statement datasets",
        "Extract core fundamental valuation metrics"
    ])
    
    assert "Successfully added 2 tasks: #1, #2" in confirmation
    assert len(clean_planner.task_list) == 2
    
    # Check structural layout of task #1
    task1 = clean_planner.task_list[0]
    assert task1.id == 1
    assert task1.content == "Download financial statement datasets"
    assert task1.status == "todo"
    assert len(task1.history) == 1
    assert task1.history[0].new_status == "todo"
    assert task1.history[0].outcome == "Task created"

def test_planner_status_state_transitions(clean_planner):
    """Validates full state transition tracking and outcome notes."""
    clean_planner.add_tasks(["Verify API variables"])
    
    # 1. Transition task #1 from todo -> in_progress
    res1 = clean_planner.update_task(id=1, new_status="in_progress")
    assert "Task #1 updated to in_progress." in res1
    assert clean_planner.task_list[0].status == "in_progress"
    
    # 2. Transition task #1 from in_progress -> completed with an outcome
    res2 = clean_planner.update_task(id=1, new_status="completed", outcome="Tokens confirmed")
    assert "Task #1 updated to completed." in res2
    
    task = clean_planner.task_list[0]
    assert task.status == "completed"
    assert task.outcome == "Tokens confirmed"
    assert len(task.history) == 3  # created -> in_progress -> completed

def test_planner_error_handling_boundaries(clean_planner):
    """Ensures errors are safely intercepted on invalid statuses or missing IDs."""
    clean_planner.add_tasks(["Run testing suite"])
    
    # Attempting to touch an unrecognized ID
    error_id = clean_planner.update_task(id=99, new_status="completed")
    assert "Error: Task #99 not found." in error_id
    
    # Attempting to assign an unsupported state literal flag
    error_status = clean_planner.update_task(id=1, new_status="arbitrary_state")
    assert "Error: Invalid status 'arbitrary_state'" in error_status

def test_planner_render_output_formatting(clean_planner):
    """Validates the structure of plan text generations for both overview and deep history audits."""
    # Test empty layout
    assert clean_planner.show_tasks() == "No tasks in the current plan."
    
    clean_planner.add_tasks(["Audit storage benchmarks"])
    clean_planner.update_task(id=1, new_status="failed", outcome="Disk space full")
    
    # Verify summary layout view
    summary = clean_planner.show_tasks(include_history=False)
    assert "--- CURRENT PLAN ---" in summary
    assert "#1 [FAILED] Audit storage benchmarks" in summary
    assert "> Latest Outcome: Disk space full" in summary
    
    # Verify deep tracking audit trail layout view
    full_audit = clean_planner.show_tasks(include_history=True)
    assert "└─ [" in full_audit
    assert "none -> todo (Outcome: Task created)" in full_audit
    assert "todo -> failed (Outcome: Disk space full)" in full_audit

def test_planner_proactive_context_injection(clean_planner):
    """Verifies the system instructions interceptor successfully bundles the active plan."""
    clean_planner.add_tasks(["Perform integration sanity test"])
    middleware = TinyPlannerMiddleware(planner=clean_planner)
    
    mock_request = ModelRequest(
        messages=[],
        system_message=SystemMessage(content="Initialize runtime context."),
        model="mock-model"
    )
    
    captured_request = None
    def mock_handler(req: ModelRequest):
        nonlocal captured_request
        captured_request = req
        return None
        
    middleware.wrap_model_call(mock_request, mock_handler)
    
    assert captured_request is not None
    final_prompt = captured_request.system_message.content
    
    assert "Initialize runtime context." in final_prompt
    assert "CRITICAL: You are using a structured planner." in final_prompt
    assert "#1 [TODO] Perform integration sanity test" in final_prompt