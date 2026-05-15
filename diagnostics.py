from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from config import AppConfig, BackendType, load_config, save_config
from embed import BgeM3Embedder
from inference import (
    LocalGemmaBackend,
    OpenRouterBackend,
    build_backend,
    mark_openrouter_probe_result,
)


ProgressCallback = Callable[[str], None]


class DiagnosticCheck(BaseModel):
    name: str
    status: str
    summary: str
    details: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class DiagnosticReport(BaseModel):
    kind: str
    checks: list[DiagnosticCheck] = Field(default_factory=list)
    config_updated: bool = False
    config_path: str | None = None


def run_housekeeping(
    config: AppConfig,
    *,
    config_path: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> DiagnosticReport:
    checks: list[DiagnosticCheck] = []

    _emit(progress_callback, "Checking CUDA availability")
    checks.append(_cuda_check())

    _emit(progress_callback, "Checking local Gemma cache state")
    local_backend = LocalGemmaBackend(
        _config_for_backend(config, BackendType.LOCAL_GEMMA)
    )
    local_status = local_backend.inspect_model()
    checks.append(
        DiagnosticCheck(
            name="local_model",
            status="pass" if local_status.available else "warn",
            summary=(
                f"Local model is available at {local_status.local_model_path}"
                if local_status.available
                else "Local model is not cached yet"
            ),
            details=local_status.detail,
            data=local_status.model_dump(),
        )
    )

    _emit(progress_callback, "Checking OpenRouter access and logprobs support")
    openrouter_backend = OpenRouterBackend(
        _config_for_backend(config, BackendType.OPENROUTER)
    )
    if not config.inference.resolve_api_key(config.dotenv_values()):
        checks.append(
            DiagnosticCheck(
                name="openrouter",
                status="skip",
                summary="OpenRouter check skipped because no API key is configured",
            )
        )
    else:
        probe = openrouter_backend.probe_logprobs_support()
        if probe["accessible"]:
            mark_openrouter_probe_result(
                config,
                logprobs_supported=probe["logprobs_supported"],
                notes=probe["notes"],
            )
            checks.append(
                DiagnosticCheck(
                    name="openrouter",
                    status="pass" if probe["logprobs_supported"] else "warn",
                    summary=(
                        "OpenRouter is reachable and returned logprobs metadata"
                        if probe["logprobs_supported"]
                        else "OpenRouter is reachable but did not return usable logprobs metadata"
                    ),
                    details=probe["notes"],
                    data=probe,
                )
            )
        else:
            mark_openrouter_probe_result(
                config,
                logprobs_supported=False,
                notes=probe["notes"],
            )
            checks.append(
                DiagnosticCheck(
                    name="openrouter",
                    status="fail",
                    summary="OpenRouter probe failed",
                    details=probe["notes"],
                    data=probe,
                )
            )

    report = DiagnosticReport(kind="housekeeping", checks=checks)
    if config_path:
        save_config(config, config_path)
        report.config_updated = True
        report.config_path = config_path
    return report


def run_smoke_tests(
    config: AppConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> DiagnosticReport:
    checks: list[DiagnosticCheck] = []

    _emit(progress_callback, "Running BGE-M3 smoke test")
    checks.append(_bge_smoke_test(config))

    _emit(progress_callback, "Running local Gemma smoke test")
    checks.append(_backend_smoke_test(config, BackendType.LOCAL_GEMMA))

    _emit(progress_callback, "Running OpenRouter smoke test")
    if not config.inference.resolve_api_key(config.dotenv_values()):
        checks.append(
            DiagnosticCheck(
                name="openrouter_smoke",
                status="skip",
                summary="OpenRouter smoke test skipped because no API key is configured",
            )
        )
    else:
        checks.append(_backend_smoke_test(config, BackendType.OPENROUTER))

    return DiagnosticReport(kind="smoke", checks=checks)


def _backend_smoke_test(
    config: AppConfig, backend_type: BackendType
) -> DiagnosticCheck:
    backend_config = _config_for_backend(config, backend_type)
    backend = build_backend(backend_config)
    name = f"{backend_type.value}_smoke"
    try:
        backend.prepare()
        text = backend.smoke_test()
        return DiagnosticCheck(
            name=name,
            status="pass",
            summary=f"{backend_type.value} smoke test completed",
            details=text,
        )
    except Exception as exc:
        return DiagnosticCheck(
            name=name,
            status="fail",
            summary=f"{backend_type.value} smoke test failed",
            details=str(exc),
        )


def _bge_smoke_test(config: AppConfig) -> DiagnosticCheck:
    try:
        embedder = BgeM3Embedder(config.retrieval)
        dense, sparse = embedder.embed_texts(["CodeAct embedding smoke test"])[0]
        return DiagnosticCheck(
            name="bge_smoke",
            status="pass",
            summary="BGE-M3 smoke test completed",
            details=f"dense_dim={len(dense)}, sparse_terms={len(sparse)}",
        )
    except Exception as exc:
        return DiagnosticCheck(
            name="bge_smoke",
            status="fail",
            summary="BGE-M3 smoke test failed",
            details=str(exc),
        )


def _cuda_check() -> DiagnosticCheck:
    try:
        import torch

        available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if available else 0
        devices = [torch.cuda.get_device_name(index) for index in range(device_count)]
        return DiagnosticCheck(
            name="cuda",
            status="pass" if available else "warn",
            summary="CUDA is available" if available else "CUDA is not available",
            details=", ".join(devices) if devices else None,
            data={
                "available": available,
                "device_count": device_count,
                "devices": devices,
            },
        )
    except Exception as exc:
        return DiagnosticCheck(
            name="cuda",
            status="fail",
            summary="CUDA check failed",
            details=str(exc),
        )


def _config_for_backend(config: AppConfig, backend_type: BackendType) -> AppConfig:
    cloned = copy.deepcopy(config)
    cloned.inference.backend = backend_type
    return cloned


def _emit(progress_callback: ProgressCallback | None, message: str) -> None:
    if progress_callback:
        progress_callback(message)


def _print_report(report: DiagnosticReport) -> None:
    print(json.dumps(report.model_dump(mode="json"), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CodeAct housekeeping and smoke tests"
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--housekeeping", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config and Path(args.config).exists() else None
    config = load_config(config_path)
    ran_any = False
    if args.housekeeping or not args.smoke:
        report = run_housekeeping(
            config,
            config_path=None if args.no_save else (args.config or "config.json"),
        )
        _print_report(report)
        ran_any = True
    if args.smoke:
        report = run_smoke_tests(config)
        _print_report(report)
        ran_any = True
    if not ran_any:
        parser.error("No diagnostic action selected")
