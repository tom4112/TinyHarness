from datetime import datetime
from pathlib import Path
from typing import Callable, Awaitable, List
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage
from langchain_core.tools import StructuredTool
from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
import pandas as pd
import tempfile

class TinyFilesystemMiddleware(FilesystemMiddleware):
    """
    An opinionated modification of Deep Agents' FilesystemMiddleware.

    - Isolation Sandbox: 
        The workspace can only be nested within the CWD, and file operations
        are strictly protected against directory traversal (..) security breaks.
    - Enriched Metadata:
        The list of files returned by ls is accompanied by additional metadata 
        such as file size and last modification timestamp. 
        It is formatted as a MD table, making it suited for both agents and humans.
    - Proactive Context Injection:
        The output of the ls tool is automatically injected into the system prompt
        at each conversation turn, ensuring the agent always tracks its latest context
        without needing to waste turns executing the tool on its own.
    """

    def __init__(self, root_dir: str = "./workspace"):

        cwd = Path.cwd().resolve()
        system_tmp = Path(tempfile.gettempdir()).resolve()
        requested_path = Path(root_dir).resolve()

        is_subpath_of_cwd = False
        is_subpath_of_tmp = False

        try:
            requested_path.relative_to(cwd)
            is_subpath_of_cwd = True
        except ValueError:
            pass

        try:
            requested_path.relative_to(system_tmp)
            is_subpath_of_tmp = True
        except ValueError:
            pass

        if not (is_subpath_of_cwd or is_subpath_of_tmp):
            raise PermissionError(
                f"Security Violation: Target workspace '{root_dir}' escapes permitted sandbox areas."
            )

        self.root_dir = requested_path
        self.virtual_mode = True
        
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self._backend = FilesystemBackend(
            root_dir=str(self.root_dir), virtual_mode=self.virtual_mode
        )
        
        super().__init__(backend=self._backend)

    @property
    def tools(self) -> List[StructuredTool]:

        # 1. Fetch whatever tools FilesystemMiddleware natively exposes right now
        base_tools = super().tools if hasattr(super(), "tools") else getattr(self, "_tools", [])
        
        # 2. Re-wrap the custom ls tool
        custom_ls_tool = StructuredTool.from_function(
            name="ls",
            func=self.ls,
            description="List files in a directory recursively with sizes and timestamps."
        )

        # 3. Filter out the old string-based 'ls' and inject the custom one
        cleaned_tools = [t for t in base_tools if t.name != "ls"]
        cleaned_tools.append(custom_ls_tool)
        
        return cleaned_tools

    @tools.setter
    def tools(self, value):
        self._tools = value

    def ls(self, path: str = ".") -> str:
        clean_sub_dir = path.lstrip("/") if path != "/" else "."
        target_path = (self.root_dir / clean_sub_dir).resolve()
        
        try:
            target_path.relative_to(self.root_dir)
        except ValueError:
            raise PermissionError("Access denied: Attempted to list contents outside of the workspace.")

        if not target_path.exists():
            return "Directory does not exist."

        file_list = []
        for item in target_path.rglob("*"):
            if item.is_file():
                stat_info = item.stat()
                last_modified = datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

                file_list.append({
                    "path": f"/{item.relative_to(self.root_dir)}",
                    "size_bytes": str(stat_info.st_size),
                    "last_modified": last_modified
                })
        
        if not file_list:
            return "No files found."

        df = pd.DataFrame(file_list)
        return df.to_markdown(index=False)

    def _inject_workspace(self, request: ModelRequest) -> ModelRequest:
        current_inventory = self.ls()
        instruction = (
            f"\n\nCRITICAL: You are running within a sandboxed virtual filesystem.\n"
            f"--- LIVE WORKSPACE INVENTORY ---\n"
            f"{current_inventory}\n"
            f"--------------------------------\n"
            "Review the live workspace inventory above before executing your plan."
        )
        
        if request.system_message:
            existing_content = request.system_message.content
            new_content = f"{existing_content}\n{instruction}"
            return request.override(system_message=SystemMessage(content=new_content))
        
        return request.override(system_message=SystemMessage(content=instruction))

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        return handler(self._inject_workspace(request))

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse:
        return await handler(self._inject_workspace(request))