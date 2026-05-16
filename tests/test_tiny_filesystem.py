import pytest
from pathlib import Path
from langchain_core.messages import SystemMessage
from langchain.agents.middleware import ModelRequest
from tinyharness.tiny_filesystem import TinyFilesystemMiddleware

@pytest.fixture
def temp_workspace(tmp_path) -> Path:
    """Provides a clean, isolated temporary directory for file sandboxing."""
    return tmp_path / "workspace"

def test_sandbox_strict_cwd_enforcement():
    """Verifies that the middleware completely blocks paths escaping the CWD boundary."""
    # Attempting to target a system root or directory above current working tree
    with pytest.raises(PermissionError, match="Security Violation"):
        TinyFilesystemMiddleware(root_dir="/absolute/outside/path")

    with pytest.raises(PermissionError, match="Security Violation"):
        TinyFilesystemMiddleware(root_dir="../../escaped_path")

def test_tools_property_interception(temp_workspace):
    """Ensures our custom Markdown 'ls' tool replaces the default DeepAgents tool."""
    vfs = TinyFilesystemMiddleware(root_dir=temp_workspace)
    
    # Extract exposed tools via the dynamic property override
    active_tools = vfs.tools
    tool_names = [t.name for t in active_tools]
    
    # Core write capability must remain intact, but 'ls' should target our custom function
    assert "write_file" in tool_names or "edit_file" in tool_names
    assert "ls" in tool_names
    
    # Locate the specific 'ls' tool wrapper object
    ls_tool = next(t for t in active_tools if t.name == "ls")
    assert ls_tool.description == "List files in a directory recursively with sizes and timestamps."

def test_ls_markdown_rendering(temp_workspace):
    """Verifies that directory listings produce accurate, structural markdown tables."""
    vfs = TinyFilesystemMiddleware(root_dir=temp_workspace)
    
    # Test empty directory state
    assert vfs.ls() == "No files found."
    
    # Seed mock files into the sandboxed location
    file1 = temp_workspace / "sample.txt"
    file1.parent.mkdir(parents=True, exist_ok=True)
    file1.write_text("Hello World")  # 11 bytes
    
    nested_dir = temp_workspace / "analysis"
    nested_dir.mkdir()
    file2 = nested_dir / "report.md"
    file2.write_text("Data")  # 4 bytes
    
    markdown_output = vfs.ls()
    
    # FIX: Assert that structural content strings exist regardless of dynamic column spacing padding
    assert "path" in markdown_output
    assert "size_bytes" in markdown_output
    assert "last_modified" in markdown_output
    assert "/sample.txt" in markdown_output
    assert "11" in markdown_output
    assert "/analysis/report.md" in markdown_output
    assert "4" in markdown_output

def test_ls_traversal_attack_protection(temp_workspace):
    """Verifies that runtime path evaluation blocks directory traversal breakouts."""
    vfs = TinyFilesystemMiddleware(root_dir=temp_workspace)
    
    # Attempt to query tracking parameters outside the root target via runtime string parameters
    with pytest.raises(PermissionError, match="Access denied"):
        vfs.ls(path="../../secret_system_files")

def test_proactive_context_injection(temp_workspace):
    """Verifies that workspace files are automatically injected into the system prompt envelope."""
    vfs = TinyFilesystemMiddleware(root_dir=temp_workspace)
    
    # Seed a single file to generate layout tracking lines
    mock_file = temp_workspace / "test_run.json"
    mock_file.write_text("{}")
    
    # Mock an incoming pipeline execution frame matching LangChain's request contract
    mock_request = ModelRequest(
        messages=[],
        system_message=SystemMessage(content="You are a helpful assistant."),
        model="mock-model"
    )
    
    # Simulating the exact internal callback execution executed by the LangChain framework harness
    captured_request = None
    def dummy_handler(req: ModelRequest):
        nonlocal captured_request
        captured_request = req
        return None
        
    vfs.wrap_model_call(mock_request, dummy_handler)
    
    assert captured_request is not None
    final_prompt = captured_request.system_message.content
    
    # Assert that original prompt boundaries are safely maintained alongside the new injection block
    assert "You are a helpful assistant." in final_prompt
    assert "CRITICAL: You are running within a sandboxed virtual filesystem." in final_prompt
    assert "--- LIVE WORKSPACE INVENTORY ---" in final_prompt
    assert "/test_run.json" in final_prompt