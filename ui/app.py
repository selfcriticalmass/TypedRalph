from __future__ import annotations

from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
)

from agent import CodeActAgent
from config import AppConfig, BackendType, ExecutionMode, load_config, save_config
from diagnostics import DiagnosticReport, run_housekeeping, run_smoke_tests
from executor import SubprocessExecutor
from inference import BackendCapabilities, build_backend
from schema import FunctionMatch, FunctionSchema, load_function_registry
from search import SearchEngine


class ApiKeyScreen(ModalScreen[str | None]):
    CSS = """
    ApiKeyScreen {
        align: center middle;
    }

    #dialog {
        width: 70;
        height: 12;
        padding: 1 2;
        background: $panel;
        border: round $primary;
    }

    #runtime_api_key {
        margin: 1 0;
    }

    #dialog_buttons {
        height: auto;
        align-horizontal: right;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("API key required for the selected backend")
            yield Input(
                password=True, placeholder="Enter API key", id="runtime_api_key"
            )
            with Horizontal(id="dialog_buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Use Key", variant="primary", id="save")

    @on(Button.Pressed, "#cancel")
    def cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#save")
    def save(self) -> None:
        self.dismiss(self.query_one("#runtime_api_key", Input).value or None)


class CodeActApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #warning_bar {
        color: $warning;
        padding: 0 1;
        height: auto;
    }

    #status_bar {
        padding: 0 1;
        background: $boost;
        height: auto;
    }

    #body {
        height: 1fr;
    }

    #sidebar {
        width: 38;
        border: solid $panel;
    }

    #trace_panel {
        border: solid $panel;
        width: 1fr;
    }

    #schema_filter {
        margin: 0 1 1 1;
    }

    #schema_table {
        height: 1fr;
    }

    #trace_log {
        height: 1fr;
    }

    #controls {
        layout: vertical;
        height: auto;
        padding: 1;
        border-top: solid $panel;
    }

    #toolbar {
        height: auto;
        margin-bottom: 1;
    }

    #prompt_bar {
        height: auto;
    }

    #prompt_input {
        width: 1fr;
    }

    #model_input {
        width: 28;
        margin-right: 1;
    }

    Select {
        width: 20;
        margin-right: 1;
    }

    Button {
        margin-right: 1;
    }
    """

    BINDINGS = [("ctrl+r", "run_prompt", "Run prompt")]

    def __init__(self, config_path: str | None = None):
        super().__init__()
        self.config_path = config_path
        self.config: AppConfig = load_config(config_path)
        self.backend = build_backend(self.config)
        self.capabilities: BackendCapabilities = self.backend.capabilities()
        self.registry: list[FunctionSchema] = []
        self.search_engine: SearchEngine | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="warning_bar")
        yield Static("", id="status_bar")
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Label("Schema Browser")
                yield Input(
                    placeholder="Search registered functions", id="schema_filter"
                )
                yield DataTable(id="schema_table")
            with Vertical(id="trace_panel"):
                yield Label("Agent Trace")
                yield RichLog(id="trace_log", wrap=True, highlight=False, markup=False)
        with Vertical(id="controls"):
            with Horizontal(id="toolbar"):
                yield Select(
                    [
                        ("Free", ExecutionMode.FREE.value),
                        ("Schema", ExecutionMode.SCHEMA.value),
                    ],
                    value=self.config.mode.value,
                    id="mode_select",
                )
                yield Select(
                    [
                        ("Local Gemma", BackendType.LOCAL_GEMMA.value),
                        ("OpenRouter", BackendType.OPENROUTER.value),
                    ],
                    value=self.config.inference.backend.value,
                    id="backend_select",
                )
                yield Input(
                    value=self.config.inference.model_name,
                    placeholder="Model id",
                    id="model_input",
                )
                yield Button("Housekeeping", id="housekeeping_button")
                yield Button("Smoke", id="smoke_button")
            with Horizontal(id="prompt_bar"):
                yield Input(placeholder="Enter a task for the agent", id="prompt_input")
                yield Button("Run", variant="primary", id="run_button")
        yield Footer()

    def on_mount(self) -> None:
        self._setup_schema_table()
        self._reload_registry()
        self._refresh_status()
        if self.config.inference.bootstrap_on_startup:
            self._prepare_backend_on_startup()

    def _setup_schema_table(self) -> None:
        table = self.query_one("#schema_table", DataTable)
        table.add_columns("Function", "Signature", "Lex", "Sem", "Fuse")
        table.cursor_type = "row"

    def _refresh_status(self) -> None:
        model_label = self.capabilities.local_model_path or self.capabilities.model_name
        readiness = "ready"
        if self.capabilities.bootstrap_required:
            readiness = "bootstrap"
        elif self.capabilities.model_available is False:
            readiness = "missing"
        logits_label = self.capabilities.logits_mode
        status = (
            f"Mode: {self.config.mode.value} | "
            f"Backend: {self.config.inference.backend.value} | "
            f"Model: {model_label} | "
            f"Ready: {readiness} | "
            f"Logits: {logits_label} | "
            f"CFG: {self.capabilities.cfg_state}"
        )
        self.query_one("#status_bar", Static).update(status)
        self.query_one("#warning_bar", Static).update(self._combined_warning())

    def _reload_backend(self) -> None:
        self.backend = build_backend(self.config)
        self.capabilities = self.backend.capabilities()
        self._refresh_status()

    def _config_save_path(self) -> str:
        if self.config_path:
            return self.config_path
        return str(Path(self.config.execution.workspace_root) / "config.json")

    def _reload_config_from_disk(self) -> None:
        config_path = self._config_save_path()
        path = Path(config_path)
        if not path.exists():
            return

        runtime_api_key = self.config.inference.runtime_api_key
        runtime_hf_token = self.config.inference.runtime_hf_token
        loaded = load_config(config_path)
        loaded.inference.runtime_api_key = runtime_api_key
        loaded.inference.runtime_hf_token = runtime_hf_token
        self.config = loaded

        self.query_one("#mode_select", Select).value = self.config.mode.value
        self.query_one(
            "#backend_select", Select
        ).value = self.config.inference.backend.value
        self.query_one("#model_input", Input).value = self.config.inference.model_name

        self.backend = build_backend(self.config)
        self.capabilities = self.backend.capabilities()
        self._reload_registry()
        self._refresh_status()

    def _sync_model_name_from_ui(self) -> None:
        model_value = self.query_one("#model_input", Input).value.strip()
        previous_name = self.config.inference.model_name
        self.config.inference.apply_model_name(model_value)
        if self.config.inference.model_name != previous_name:
            self._reload_backend()

    def _set_capabilities(self, capabilities: BackendCapabilities) -> None:
        self.capabilities = capabilities
        self._refresh_status()

    @work(thread=True, exclusive=True)
    def _prepare_backend_on_startup(self) -> None:
        try:
            self.backend.prepare(
                progress_callback=lambda message: self.call_from_thread(
                    self._append_trace,
                    "system",
                    message,
                )
            )
            self.call_from_thread(self._set_capabilities, self.backend.capabilities())
        except Exception as exc:
            self.call_from_thread(self._append_trace, "warning", str(exc))

    def _reload_registry(self) -> None:
        registry_path = self.config.schema_config.registry_path
        if not registry_path:
            self.registry = []
            self.search_engine = None
            self._update_schema_table([])
            return
        try:
            self.registry = load_function_registry(registry_path)
            self.search_engine = SearchEngine(self.registry, self.config.retrieval)
            self._update_schema_table(self.search_engine.list_functions())
            self.query_one("#warning_bar", Static).update(self._combined_warning())
        except Exception as exc:
            self.registry = []
            self.search_engine = None
            self._append_trace("warning", f"Registry load failed: {exc}")
            self._update_schema_table([])
            self.query_one("#warning_bar", Static).update(self._combined_warning())

    def _update_schema_table(self, rows: list[FunctionMatch]) -> None:
        table = self.query_one("#schema_table", DataTable)
        table.clear(columns=False)
        for row in rows:
            table.add_row(
                row.function_name,
                row.signature_summary,
                f"{row.lexical_score:.3f}",
                f"{row.semantic_score:.3f}",
                f"{row.fused_score:.3f}",
            )

    def _append_trace(self, event_type: str, content: str) -> None:
        log = self.query_one("#trace_log", RichLog)
        log.write(f"[{event_type.upper()}]\n{content}\n")

    def _append_report(self, report: DiagnosticReport) -> None:
        if report.config_updated and report.config_path:
            self._append_trace("system", f"Saved config to {report.config_path}")
        for check in report.checks:
            self._append_trace(
                check.status,
                f"{check.name}: {check.summary}"
                + (f"\n{check.details}" if check.details else ""),
            )

    def _combined_warning(self) -> str:
        warnings = []
        if self.capabilities.warning:
            warnings.append(self.capabilities.warning)
        if self.search_engine and self.search_engine.semantic_error:
            warnings.append(
                f"Semantic retrieval unavailable: {self.search_engine.semantic_error}"
            )
        return " | ".join(warnings)

    def action_run_prompt(self) -> None:
        self._start_run()

    @on(Button.Pressed, "#run_button")
    def on_run_button(self) -> None:
        self._start_run()

    @on(Button.Pressed, "#housekeeping_button")
    def on_housekeeping_button(self) -> None:
        self._start_housekeeping()

    @on(Button.Pressed, "#smoke_button")
    def on_smoke_button(self) -> None:
        self._start_smoke_tests()

    @on(Input.Submitted, "#model_input")
    def on_model_submit(self) -> None:
        self._sync_model_name_from_ui()
        save_config(self.config, self._config_save_path())
        self._append_trace(
            "system", f"Saved model selection to {self._config_save_path()}"
        )

    @on(Input.Submitted, "#prompt_input")
    def on_prompt_submit(self) -> None:
        self._start_run()

    @on(Input.Changed, "#schema_filter")
    def on_schema_filter_changed(self, event: Input.Changed) -> None:
        if not self.search_engine:
            return
        query = event.value.strip()
        if not query:
            self._update_schema_table(self.search_engine.list_functions())
            return
        self._update_schema_table(self.search_engine.search(query))

    @on(Select.Changed, "#mode_select")
    def on_mode_changed(self, event: Select.Changed) -> None:
        self.config.mode = ExecutionMode(event.value)
        self._refresh_status()

    @on(Select.Changed, "#backend_select")
    def on_backend_changed(self, event: Select.Changed) -> None:
        self.config.inference.backend = BackendType(event.value)
        self._reload_backend()

    def _start_run(self) -> None:
        self._reload_config_from_disk()
        self._sync_model_name_from_ui()
        prompt = self.query_one("#prompt_input", Input).value.strip()
        if not prompt:
            return
        if (
            self.config.inference.requires_api_key()
            and not self.config.inference.resolve_api_key(self.config.dotenv_values())
        ):
            self.push_screen(ApiKeyScreen(), self._handle_runtime_key)
            return
        self._run_prompt(prompt)

    def _start_housekeeping(self) -> None:
        self._sync_model_name_from_ui()
        self._run_housekeeping()

    def _start_smoke_tests(self) -> None:
        self._sync_model_name_from_ui()
        self._run_smoke_tests()

    def _handle_runtime_key(self, value: str | None) -> None:
        if value:
            self.config.inference.runtime_api_key = value
            self._reload_backend()
            prompt = self.query_one("#prompt_input", Input).value.strip()
            if prompt:
                self._run_prompt(prompt)

    @work(thread=True, exclusive=True)
    def _run_housekeeping(self) -> None:
        self.call_from_thread(
            self._append_trace, "system", "Starting housekeeping checks"
        )
        try:
            report = run_housekeeping(
                self.config,
                config_path=self._config_save_path(),
                progress_callback=lambda message: self.call_from_thread(
                    self._append_trace,
                    "system",
                    message,
                ),
            )
            self.backend = build_backend(self.config)
            self.call_from_thread(self._set_capabilities, self.backend.capabilities())
            self.call_from_thread(self._append_report, report)
        except Exception as exc:
            self.call_from_thread(self._append_trace, "error", str(exc))

    @work(thread=True, exclusive=True)
    def _run_smoke_tests(self) -> None:
        self.call_from_thread(self._append_trace, "system", "Starting smoke tests")
        try:
            report = run_smoke_tests(
                self.config,
                progress_callback=lambda message: self.call_from_thread(
                    self._append_trace,
                    "system",
                    message,
                ),
            )
            self.call_from_thread(self._append_report, report)
        except Exception as exc:
            self.call_from_thread(self._append_trace, "error", str(exc))

    @work(thread=True, exclusive=True)
    def _run_prompt(self, prompt: str) -> None:
        self.call_from_thread(self._append_trace, "user", prompt)
        try:
            self.backend.prepare(
                progress_callback=lambda message: self.call_from_thread(
                    self._append_trace,
                    "system",
                    message,
                )
            )
            self.call_from_thread(self._set_capabilities, self.backend.capabilities())
        except Exception as exc:
            self.call_from_thread(self._append_trace, "error", str(exc))
            return
        executor = SubprocessExecutor(self.config.execution)
        agent = CodeActAgent(
            config=self.config,
            backend=self.backend,
            executor=executor,
            search_engine=self.search_engine,
            registry=self.registry,
            trace_callback=lambda event: self.call_from_thread(
                self._append_trace,
                event["type"],
                event["content"],
            ),
        )
        try:
            result = agent.run(prompt, self.config.mode)
            self.call_from_thread(self._append_trace, "final", result.final_answer)
        except Exception as exc:
            self.call_from_thread(self._append_trace, "error", str(exc))


def main() -> None:
    config_path = str(Path("config.json")) if Path("config.json").exists() else None
    CodeActApp(config_path=config_path).run()


if __name__ == "__main__":
    main()
