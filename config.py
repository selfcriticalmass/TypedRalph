from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExecutionMode(str, Enum):
    FREE = "free"
    SCHEMA = "schema"


class BackendType(str, Enum):
    LOCAL_GEMMA = "local_gemma"
    OPENROUTER = "openrouter"


class UIConfig(BaseModel):
    theme: str = "textual-dark"
    log_verbosity: str = "info"
    show_timestamps: bool = True


class InferenceConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    backend: BackendType = BackendType.LOCAL_GEMMA
    model_name: str = "google/gemma-4-31B-it"
    model_revision: str | None = None
    local_model_id: str = "google/gemma-4-31B-it"
    local_cache_root: str = ".venv/huggingface"
    download_if_missing: bool = True
    bootstrap_on_startup: bool = False
    hf_token_env_var: str = "HF_TOKEN"
    runtime_hf_token: str | None = Field(default=None, repr=False)
    base_url: str = "https://openrouter.ai/api/v1"
    base_url_env_var: str = "OPENROUTER_BASE_URL"
    api_key_env_var: str = "OPENROUTER_API_KEY"
    runtime_api_key: str | None = Field(default=None, repr=False)
    use_keyring: bool = False
    keyring_service: str = "codeact-agent"
    keyring_username: str = "openrouter"
    openrouter_site_url: str | None = None
    openrouter_app_name: str = "codeact-agent"
    logits_required_for_cfg: bool = True
    cfg_enabled: bool = True
    context_length: int = 8192
    temperature: float = 0.2
    max_tokens: int = 768
    stop_tokens: list[str] = Field(default_factory=list)
    full_logits_override: bool | None = None
    supports_custom_logits_override: bool | None = None
    device_map: str = "auto"
    torch_dtype: str = "auto"
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    extra_options: dict[str, Any] = Field(default_factory=dict)

    def requires_api_key(self) -> bool:
        return self.backend == BackendType.OPENROUTER

    def requires_local_model(self) -> bool:
        return self.backend == BackendType.LOCAL_GEMMA

    def resolve_api_key(
        self, dotenv_values: dict[str, str] | None = None
    ) -> str | None:
        dotenv_values = dotenv_values or {}
        for key_name in (self.api_key_env_var, "CODEACT_LLM_API_KEY"):
            value = os.getenv(key_name) or dotenv_values.get(key_name)
            if value:
                return value
        if self.runtime_api_key:
            return self.runtime_api_key
        if self.use_keyring:
            try:
                import keyring

                return keyring.get_password(self.keyring_service, self.keyring_username)
            except Exception:
                return None
        return None

    def resolve_base_url(
        self, dotenv_values: dict[str, str] | None = None
    ) -> str | None:
        dotenv_values = dotenv_values or {}
        return (
            os.getenv(self.base_url_env_var)
            or dotenv_values.get(self.base_url_env_var)
            or os.getenv("CODEACT_LLM_BASE_URL")
            or dotenv_values.get("CODEACT_LLM_BASE_URL")
            or self.base_url
        )

    def resolve_hf_token(
        self, dotenv_values: dict[str, str] | None = None
    ) -> str | None:
        dotenv_values = dotenv_values or {}
        for key_name in (
            self.hf_token_env_var,
            "HUGGING_FACE_HUB_TOKEN",
            "HF_HUB_TOKEN",
            "CODEACT_HF_TOKEN",
        ):
            value = os.getenv(key_name) or dotenv_values.get(key_name)
            if value:
                return value
        return self.runtime_hf_token


class RetrievalConfig(BaseModel):
    embedding_model: str = "BAAI/bge-m3"
    cache_path: str = "cache/vectors.parquet"
    top_k: int = 5
    rrf_k: int = 60
    semantic_enabled: bool = True
    batch_size: int = 8
    max_length: int = 2048


class ExecutionConfig(BaseModel):
    workspace_root: str = Field(default_factory=lambda: str(Path.cwd()))
    artifact_dir: str = "artifacts"
    free_mode_timeout: int = 30
    schema_mode_timeout: int = 15
    max_iterations: int = 8
    memory_limit_mb: int | None = 1024
    cpu_time_limit_seconds: int | None = 30


class SchemaConfig(BaseModel):
    registry_path: str | None = None
    auto_reload: bool = False
    validation_strictness: str = "strict"


class AppConfig(BaseModel):
    ui: UIConfig = Field(default_factory=UIConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    schema: SchemaConfig = Field(default_factory=SchemaConfig)
    mode: ExecutionMode = ExecutionMode.FREE

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        if path is None:
            return cls()
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        payload = json.loads(config_path.read_text())
        return cls.model_validate(payload)

    def dotenv_values(self) -> dict[str, str]:
        return load_dotenv_values(Path(self.execution.workspace_root) / ".env")


def load_dotenv_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        from dotenv import dotenv_values

        values = dotenv_values(path)
        return {
            key: value for key, value in values.items() if key and value is not None
        }
    except Exception:
        return {}


def load_config(path: str | Path | None = None) -> AppConfig:
    return AppConfig.load(path)
