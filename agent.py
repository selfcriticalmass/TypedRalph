from __future__ import annotations

import json
import re
from typing import Any, Callable

from pydantic import BaseModel, Field

from config import AppConfig, ExecutionMode
from executor import ExecutionResult, SubprocessExecutor
from inference import BaseInferenceBackend
from schema import FunctionSchema
from search import SearchEngine


TraceCallback = Callable[[dict[str, Any]], None]


class AgentResult(BaseModel):
    success: bool
    mode: ExecutionMode
    iterations: int
    final_answer: str
    events: list[dict[str, Any]] = Field(default_factory=list)


class CodeActAgent:
    def __init__(
        self,
        *,
        config: AppConfig,
        backend: BaseInferenceBackend,
        executor: SubprocessExecutor,
        search_engine: SearchEngine | None = None,
        registry: list[FunctionSchema] | None = None,
        trace_callback: TraceCallback | None = None,
    ):
        self.config = config
        self.backend = backend
        self.executor = executor
        self.search_engine = search_engine
        self.registry = {function.name: function for function in registry or []}
        self.trace_callback = trace_callback
        self.events: list[dict[str, Any]] = []

    def run(self, prompt: str, mode: ExecutionMode | None = None) -> AgentResult:
        mode = mode or self.config.mode
        if mode == ExecutionMode.FREE:
            return self._run_free(prompt)
        return self._run_schema(prompt)

    def _run_free(self, prompt: str) -> AgentResult:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a CodeAct-style coding agent. Reply with exactly one JSON object and no markdown. "
                    'Schema: {"thought": str, "code": str, "done": bool, "final_answer": str}. '
                    "When more work is required, set done=false and provide Python in code. "
                    "Never use exec or eval in generated code."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        max_iterations = self.config.execution.max_iterations
        for iteration in range(1, max_iterations + 1):
            self._emit("phase", f"Free iteration {iteration}")
            response_text = self.backend.generate(messages).text
            self._emit("model", response_text)
            try:
                decision = _extract_json(response_text)
            except ValueError as exc:
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Your previous response was invalid JSON: {exc}",
                    }
                )
                self._emit("warning", str(exc))
                continue

            if decision.get("done"):
                final_answer = str(decision.get("final_answer") or "Completed")
                self._emit("result", final_answer)
                return AgentResult(
                    success=True,
                    mode=ExecutionMode.FREE,
                    iterations=iteration,
                    final_answer=final_answer,
                    events=self.events,
                )

            code = decision.get("code", "")
            if not code.strip():
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": "Provide Python code in the code field or finish explicitly.",
                    }
                )
                self._emit("warning", "Model response omitted executable code")
                continue

            self._emit("code", code)
            execution = self.executor.run_python(
                code, timeout=self.config.execution.free_mode_timeout
            )
            self._emit_execution(execution)

            messages.append({"role": "assistant", "content": response_text})
            messages.append(
                {"role": "user", "content": self._free_observation(execution)}
            )

        final_answer = f"Stopped after reaching the iteration cap of {max_iterations}."
        self._emit("result", final_answer)
        return AgentResult(
            success=False,
            mode=ExecutionMode.FREE,
            iterations=max_iterations,
            final_answer=final_answer,
            events=self.events,
        )

    def _run_schema(self, prompt: str) -> AgentResult:
        if not self.search_engine or not self.registry:
            registry_path = self.config.schema_config.registry_path
            raise ValueError(
                "Schema mode requires a loaded function registry"
                + (
                    f" (registry_path={registry_path!r})"
                    if registry_path
                    else " (registry_path is not set)"
                )
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a schema-constrained agent. Reply with exactly one JSON object and no markdown. "
                    'Allowed shapes: {"tool": "search", "query": str, "reason": str}, '
                    '{"tool": "execute", "function_name": str, "args": dict, "reason": str}, '
                    '{"tool": "finish", "final_answer": str}. '
                    "Do not generate Python. Use only search and execute before finishing."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Request: {prompt}\n\n"
                    f"Available functions: {', '.join(sorted(self.registry)) or 'none'}"
                ),
            },
        ]

        max_iterations = self.config.execution.max_iterations
        for iteration in range(1, max_iterations + 1):
            self._emit("phase", f"Schema iteration {iteration}")
            response_text = self.backend.generate(messages).text
            self._emit("model", response_text)
            try:
                decision = _extract_json(response_text)
            except ValueError as exc:
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Your previous response was invalid JSON: {exc}",
                    }
                )
                self._emit("warning", str(exc))
                continue

            tool = decision.get("tool")
            if tool == "finish":
                final_answer = str(decision.get("final_answer") or "Completed")
                self._emit("result", final_answer)
                return AgentResult(
                    success=True,
                    mode=ExecutionMode.SCHEMA,
                    iterations=iteration,
                    final_answer=final_answer,
                    events=self.events,
                )

            if tool == "search":
                query = str(decision.get("query") or prompt)
                matches = self.search_engine.search(query)
                payload = [match.model_dump() for match in matches]
                self._emit("search", json.dumps(payload, indent=2))
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": f"search({query!r}) returned:\n{json.dumps(payload, indent=2)}",
                    }
                )
                continue

            if tool == "execute":
                function_name = decision.get("function_name")
                args = decision.get("args") or {}
                function = self.registry.get(function_name)
                if function is None:
                    observation = f"execute failed: unknown function '{function_name}'"
                    self._emit("warning", observation)
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": observation})
                    continue
                try:
                    execution = self.executor.execute_function(
                        function,
                        args,
                        timeout=self.config.execution.schema_mode_timeout,
                    )
                except Exception as exc:
                    observation = f"execute failed before subprocess start: {exc}"
                    self._emit("warning", observation)
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": observation})
                    continue

                self._emit_execution(execution)
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"execute(function_name={function_name!r}, args={json.dumps(args)}) returned:\n"
                            f"{self._schema_observation(execution)}"
                        ),
                    }
                )
                continue

            observation = (
                f"Unsupported tool decision: {tool!r}. Use search, execute, or finish."
            )
            self._emit("warning", observation)
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": observation})

        final_answer = f"Stopped after reaching the iteration cap of {max_iterations}."
        self._emit("result", final_answer)
        return AgentResult(
            success=False,
            mode=ExecutionMode.SCHEMA,
            iterations=max_iterations,
            final_answer=final_answer,
            events=self.events,
        )

    def _free_observation(self, execution: ExecutionResult) -> str:
        return (
            f"Execution status: {execution.status}\n"
            f"Exit code: {execution.exit_code}\n"
            f"Stdout:\n{execution.stdout or '<empty>'}\n\n"
            f"Stderr:\n{execution.stderr or '<empty>'}\n\n"
            f"Artifacts: {execution.artifact_paths or []}"
        )

    def _schema_observation(self, execution: ExecutionResult) -> str:
        payload = {
            "status": execution.status,
            "exit_code": execution.exit_code,
            "stdout": execution.stdout,
            "stderr": execution.stderr,
            "result": execution.result,
            "result_text": execution.result_text,
            "artifacts": execution.artifact_paths,
        }
        return json.dumps(payload, indent=2, default=str)

    def _emit_execution(self, execution: ExecutionResult) -> None:
        content = json.dumps(
            {
                "status": execution.status,
                "exit_code": execution.exit_code,
                "stdout": execution.stdout,
                "stderr": execution.stderr,
                "result": execution.result,
                "result_text": execution.result_text,
                "artifacts": execution.artifact_paths,
                "duration_seconds": execution.duration_seconds,
            },
            indent=2,
            default=str,
        )
        self._emit("execution", content)

    def _emit(self, event_type: str, content: str) -> None:
        event = {"type": event_type, "content": content}
        self.events.append(event)
        if self.trace_callback:
            self.trace_callback(event)


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1)
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed
