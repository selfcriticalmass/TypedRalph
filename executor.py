from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from config import ExecutionConfig
from schema import FunctionSchema


class ExecutionResult(BaseModel):
    status: str
    exit_code: int | None
    stdout: str
    stderr: str
    result: Any = None
    result_text: str | None = None
    duration_seconds: float
    artifact_paths: list[str] = Field(default_factory=list)
    command: list[str] = Field(default_factory=list)


class SubprocessExecutor:
    def __init__(self, config: ExecutionConfig):
        self.config = config

    def run_python(
        self,
        code: str,
        *,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        timeout = timeout or self.config.free_mode_timeout
        with self._prepare_run_dir("free") as run_dir:
            script_path = run_dir / "generated_code.py"
            script_path.write_text(code)
            command = [sys.executable, str(script_path)]
            result = self._run_subprocess(command, run_dir, timeout, env)
            result.artifact_paths = self._collect_artifacts(run_dir, {script_path})
            if result.stdout.strip() and not result.result_text:
                result.result_text = result.stdout.strip()
            return result

    def execute_function(
        self,
        function: FunctionSchema,
        args: dict[str, Any],
        *,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        timeout = timeout or self.config.schema_mode_timeout
        validated_args = function.validate_args(args)

        with self._prepare_run_dir("schema") as run_dir:
            payload_path = run_dir / "payload.json"
            runner_path = run_dir / "runner.py"
            result_path = run_dir / "result.json"
            payload_path.write_text(
                json.dumps(
                    {
                        "function": function.callable_reference(),
                        "args": validated_args,
                        "result_path": str(result_path),
                    }
                )
            )
            runner_path.write_text(_function_runner_source())

            command = [sys.executable, str(runner_path), str(payload_path)]
            result = self._run_subprocess(command, run_dir, timeout, env)
            if result_path.exists():
                payload = json.loads(result_path.read_text())
                result.status = payload.get("status", result.status)
                result.result = payload.get("result")
                result.result_text = payload.get("result_text")
                if payload.get("error"):
                    result.stderr = (result.stderr + "\n" + payload["error"]).strip()
            result.artifact_paths = self._collect_artifacts(
                run_dir, {payload_path, runner_path, result_path}
            )
            return result

    @contextmanager
    def _prepare_run_dir(self, prefix: str):
        artifact_root = Path(self.config.artifact_dir)
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"codeact-{prefix}-", dir=artifact_root
        ) as temp_dir:
            yield Path(temp_dir)

    def _resource_limiter(self):
        if os.name != "posix":
            return None

        memory_limit = self.config.memory_limit_mb
        cpu_limit = self.config.cpu_time_limit_seconds

        def limit_resources():
            import resource

            if memory_limit:
                bytes_limit = memory_limit * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
            if cpu_limit:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))

        return limit_resources

    def _base_env(self, env: dict[str, str] | None = None) -> dict[str, str]:
        base = os.environ.copy()
        base["CODEACT_WORKSPACE_ROOT"] = self.config.workspace_root
        if env:
            base.update(env)
        return base

    def _run_subprocess(
        self,
        command: list[str],
        run_dir: Path,
        timeout: int,
        env: dict[str, str] | None,
    ) -> ExecutionResult:
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=run_dir,
                env=self._base_env(env),
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=self._resource_limiter(),
            )
            status = "success" if completed.returncode == 0 else "error"
            return ExecutionResult(
                status=status,
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_seconds=time.perf_counter() - start,
                command=command,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                status="timeout",
                exit_code=None,
                stdout=exc.stdout or "",
                stderr=exc.stderr or f"Execution timed out after {timeout} seconds",
                duration_seconds=time.perf_counter() - start,
                command=command,
            )

    def _collect_artifacts(self, run_dir: Path, ignored_paths: set[Path]) -> list[str]:
        ignored = {path.name for path in ignored_paths}
        artifacts: list[str] = []
        for path in run_dir.rglob("*"):
            if path.is_file() and path.name not in ignored:
                artifacts.append(str(path.relative_to(run_dir)))
        return artifacts


def _function_runner_source() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import importlib
        import importlib.util
        import json
        import traceback
        from pathlib import Path
        import sys


        def _load_module(module_name: str, source_file: str | None):
            try:
                return importlib.import_module(module_name)
            except Exception:
                if not source_file:
                    raise
                spec = importlib.util.spec_from_file_location(module_name, source_file)
                if spec is None or spec.loader is None:
                    raise
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module


        def _resolve_qualname(root, qualname: str):
            current = root
            for part in qualname.split('.'):
                current = getattr(current, part)
            return current


        def _json_safe(value):
            try:
                json.dumps(value)
                return value, None
            except TypeError:
                return None, repr(value)


        payload = json.loads(Path(sys.argv[1]).read_text())
        result_path = Path(payload['result_path'])
        function_payload = payload['function']

        try:
            module = _load_module(function_payload['module_name'], function_payload.get('source_file'))
            target = _resolve_qualname(module, function_payload['qualname'])
            result = target(**payload['args'])
            json_result, result_text = _json_safe(result)
            result_path.write_text(
                json.dumps(
                    {
                        'status': 'success',
                        'result': json_result,
                        'result_text': result_text,
                    }
                )
            )
        except Exception:
            error = traceback.format_exc()
            result_path.write_text(json.dumps({'status': 'error', 'error': error}))
            print(error, file=sys.stderr)
            raise
        """
    )
