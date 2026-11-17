"""Simulated tools: nothing executes anywhere (ToolShield run_local.py --mode tools semantics).

Files live in a per-sample in-memory filesystem kept in the Inspect store, so every tool call and
every file the model "wrote" ends up in the eval log. bash and SQL return canned success messages.
Seed the filesystem with `seed_vfs()` (reads Sample.metadata["files"] = {path: content}).
"""
from __future__ import annotations

from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, tool
from inspect_ai.util import store

VFS_KEY = "sim_vfs"


def _vfs() -> dict[str, str]:
    vfs = store().get(VFS_KEY)
    if vfs is None:
        vfs = {}
        store().set(VFS_KEY, vfs)
    return vfs


def _save(vfs: dict[str, str]) -> None:
    store().set(VFS_KEY, vfs)


@solver
def seed_vfs() -> Solver:
    """Copy Sample.metadata["files"] into the simulated filesystem before the agent runs."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        _save(dict((state.metadata or {}).get("files", {})))
        return state

    return solve


@tool(name="execute_bash")
def sim_bash(reply: str = "Command executed successfully (exit code 0).") -> Tool:
    async def execute(command: str) -> str:
        """Run a bash command in the sandbox and return its output.

        Args:
            command: The bash command to run.
        """
        return reply

    return execute


@tool(name="execute_sql")
def sim_sql(reply: str = "Query executed successfully.") -> Tool:
    async def execute(sql: str) -> str:
        """Execute a SQL statement against the PostgreSQL database.

        Args:
            sql: The SQL statement.
        """
        return reply

    return execute


@tool
def write_file() -> Tool:
    async def execute(path: str, content: str) -> str:
        """Create or overwrite a file with the given content.

        Args:
            path: Absolute file path.
            content: Full file content.
        """
        vfs = _vfs()
        vfs[path] = content
        _save(vfs)
        return f"Successfully wrote to {path}"

    return execute


@tool
def read_file() -> Tool:
    async def execute(path: str) -> str:
        """Read a file and return its content.

        Args:
            path: Absolute file path.
        """
        return _vfs().get(path, f"Error: {path} not found")

    return execute


@tool
def edit_file() -> Tool:
    async def execute(path: str, old_text: str, new_text: str) -> str:
        """Replace old_text with new_text in a file.

        Args:
            path: Absolute file path.
            old_text: Text to replace.
            new_text: Replacement text.
        """
        vfs = _vfs()
        if path not in vfs:
            return f"Error: {path} not found"
        vfs[path] = vfs[path].replace(old_text, new_text)
        _save(vfs)
        return f"Successfully edited {path}"

    return execute


@tool
def list_directory() -> Tool:
    async def execute(path: str) -> str:
        """List files under a directory.

        Args:
            path: Directory path.
        """
        prefix = path.rstrip("/") + "/"
        return "\n".join(p for p in sorted(_vfs()) if p.startswith(prefix)) or "(empty)"

    return execute


def simulated_tools(sql: bool = False) -> list[Tool]:
    tools = [sim_bash(), write_file(), read_file(), edit_file(), list_directory()]
    return tools + [sim_sql()] if sql else tools
