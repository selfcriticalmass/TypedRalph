from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from typing import Any, Callable

from pydantic import BaseModel

from config import AppConfig, BackendType


Message = dict[str, str]
TokenCallback = Callable[[str], None]
ProgressCallback = Callable[[str], None]


class BackendCapabilities(BaseModel):
    backend: str
    model_name: str
    logits_mode: str = "unavailable"
    full_logits_available: bool
    full_vocab_logits: bool
    supports_custom_logits_processors: bool
    cfg_state: str
    model_available: bool | None = None
    bootstrap_required: bool = False
    local_model_path: str | None = None
    warning: str | None = None


class InferenceResponse(BaseModel):
    text: str
    finish_reason: str | None = None
    raw: Any = None


class LocalModelStatus(BaseModel):
    model_id: str
    available: bool
    local_model_path: str | None = None
    bootstrap_required: bool = False
    detail: str | None = None


class BaseInferenceBackend:
    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.config = app_config.inference
        self._dotenv_values = app_config.dotenv_values()

    def capabilities(self) -> BackendCapabilities:
        raise NotImplementedError

    def prepare(self, progress_callback: ProgressCallback | None = None) -> None:
        return None

    def smoke_test(self) -> str:
        response = self.generate(
            [{"role": "user", "content": "Reply with exactly READY."}]
        )
        return response.text.strip()

    def generate(
        self,
        messages: list[Message],
        *,
        stream_handler: TokenCallback | None = None,
    ) -> InferenceResponse:
        raise NotImplementedError

    def api_key(self) -> str | None:
        return self.config.resolve_api_key(self._dotenv_values)

    def hf_token(self) -> str | None:
        return self.config.resolve_hf_token(self._dotenv_values)

    def _emit_progress(
        self, progress_callback: ProgressCallback | None, message: str
    ) -> None:
        if progress_callback:
            progress_callback(message)


class LocalGemmaBackend(BaseInferenceBackend):
    def __init__(self, app_config: AppConfig):
        super().__init__(app_config)
        self._processor = None
        self._model = None
        self._model_status: LocalModelStatus | None = None

    def capabilities(self) -> BackendCapabilities:
        status = self.inspect_model()
        full_logits = (
            self.config.full_logits_override
            if self.config.full_logits_override is not None
            else True
        )
        supports_custom = (
            self.config.supports_custom_logits_override
            if self.config.supports_custom_logits_override is not None
            else True
        )
        cfg_state = "enabled" if self.config.cfg_enabled and full_logits else "disabled"
        warnings: list[str] = []
        if not status.available:
            warnings.append(
                "Local Gemma model not found in the uv-managed Hugging Face cache. "
                "Running the agent will download it into .venv if downloads are enabled."
            )
        if self.config.cfg_enabled and not full_logits:
            warnings.append(
                "CFG disabled: the local backend is not exposing full logits for generation steps."
            )
        return BackendCapabilities(
            backend=self.config.backend.value,
            model_name=self.config.local_model_id,
            logits_mode="full" if full_logits else "unavailable",
            full_logits_available=full_logits,
            full_vocab_logits=full_logits,
            supports_custom_logits_processors=supports_custom,
            cfg_state=cfg_state,
            model_available=status.available,
            bootstrap_required=status.bootstrap_required,
            local_model_path=status.local_model_path,
            warning=" | ".join(warnings) or None,
        )

    def inspect_model(self) -> LocalModelStatus:
        if self._model_status is not None:
            return self._model_status

        snapshot_path = self._resolve_cached_snapshot(local_files_only=True)
        if snapshot_path is not None:
            self._model_status = LocalModelStatus(
                model_id=self.config.local_model_id,
                available=True,
                local_model_path=str(snapshot_path),
                bootstrap_required=False,
            )
            return self._model_status

        self._model_status = LocalModelStatus(
            model_id=self.config.local_model_id,
            available=False,
            local_model_path=str(self._cache_root()),
            bootstrap_required=self.config.download_if_missing,
            detail="Model snapshot was not detected in the uv-managed Hugging Face cache.",
        )
        return self._model_status

    def prepare(self, progress_callback: ProgressCallback | None = None) -> None:
        status = self.inspect_model()
        if not status.available:
            if not self.config.download_if_missing:
                raise RuntimeError(
                    "Local Gemma model is unavailable and download_if_missing is disabled."
                )
            self._emit_progress(
                progress_callback,
                f"Downloading {self.config.local_model_id} into {self._cache_root()}",
            )
            self._download_model()
            self._model_status = None
            status = self.inspect_model()
            if not status.available:
                raise RuntimeError(
                    "Local Gemma bootstrap finished without a usable cached snapshot"
                )

        self._ensure_client(progress_callback)

    def generate(
        self,
        messages: list[Message],
        *,
        stream_handler: TokenCallback | None = None,
    ) -> InferenceResponse:
        self.prepare()
        text_adapter = self._text_adapter()
        model = self._model
        inputs = self._build_inputs(messages)
        generation_kwargs = self._generation_kwargs()

        if stream_handler:
            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(
                text_adapter,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            worker = Thread(
                target=model.generate,
                kwargs={**inputs, **generation_kwargs, "streamer": streamer},
                daemon=True,
            )
            worker.start()
            text_parts: list[str] = []
            for token in streamer:
                if token:
                    text_parts.append(token)
                    stream_handler(token)
            worker.join()
            return InferenceResponse(text="".join(text_parts), finish_reason="stop")

        outputs = model.generate(
            **inputs,
            **generation_kwargs,
            return_dict_in_generate=True,
            output_scores=True,
        )
        prompt_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs.sequences[0][prompt_length:]
        text = text_adapter.decode(generated_tokens, skip_special_tokens=True)
        return InferenceResponse(
            text=text,
            finish_reason="stop",
            raw={"scores_available": bool(outputs.scores)},
        )

    def _ensure_client(self, progress_callback: ProgressCallback | None = None) -> None:
        if self._processor is not None and self._model is not None:
            return

        status = self.inspect_model()
        if not status.available or not status.local_model_path:
            raise RuntimeError("Local Gemma model path is unavailable")

        self._emit_progress(
            progress_callback, f"Loading processor from {status.local_model_path}"
        )
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_path = status.local_model_path
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=True,
        )
        text_adapter = self._text_adapter()
        if text_adapter.pad_token_id is None and text_adapter.eos_token is not None:
            text_adapter.pad_token = text_adapter.eos_token

        model_kwargs = {
            "device_map": self.config.device_map,
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": True,
            **self.config.extra_options,
        }
        torch_dtype = _resolve_torch_dtype(self.config.torch_dtype)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        self._emit_progress(
            progress_callback, f"Loading model weights from {status.local_model_path}"
        )
        self._model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    def _download_model(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=self.config.local_model_id,
            revision=self.config.model_revision,
            cache_dir=str(self._cache_root()),
            token=self.hf_token(),
            resume_download=True,
        )

    def _build_inputs(self, messages: list[Message]) -> dict[str, Any]:
        processor = self._processor
        text_adapter = self._text_adapter()
        model = self._model
        if hasattr(processor, "apply_chat_template"):
            rendered = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        elif hasattr(text_adapter, "apply_chat_template"):
            rendered = text_adapter.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = render_messages(messages)

        try:
            inputs = processor(text=rendered, return_tensors="pt")
        except TypeError:
            inputs = processor(rendered, return_tensors="pt")

        if getattr(model, "hf_device_map", None):
            return inputs
        model_device = getattr(model, "device", None)
        if model_device is None:
            return inputs
        return {key: value.to(model_device) for key, value in inputs.items()}

    def _generation_kwargs(self) -> dict[str, Any]:
        do_sample = self.config.temperature > 0
        text_adapter = self._text_adapter()
        kwargs = {
            "max_new_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "do_sample": do_sample,
            "pad_token_id": text_adapter.pad_token_id,
            "eos_token_id": text_adapter.eos_token_id,
            "logits_processor": self._build_logits_processors(),
        }
        if not do_sample:
            kwargs.pop("temperature")
        return kwargs

    def _build_logits_processors(self):
        from transformers import LogitsProcessorList

        # Local transformers inference is the place where CFG-style weighting hooks belong.
        return LogitsProcessorList()

    def _cache_root(self) -> Path:
        cache_root = _resolve_workspace_path(
            self.app_config.execution.workspace_root,
            self.config.local_cache_root,
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root

    def _resolve_cached_snapshot(self, *, local_files_only: bool) -> Path | None:
        from huggingface_hub import snapshot_download

        try:
            snapshot_path = snapshot_download(
                repo_id=self.config.local_model_id,
                revision=self.config.model_revision,
                cache_dir=str(self._cache_root()),
                token=self.hf_token(),
                local_files_only=local_files_only,
                resume_download=True,
            )
        except Exception:
            return None

        path = Path(snapshot_path)
        return path if path.exists() else None

    def _text_adapter(self):
        if self._processor is None:
            raise RuntimeError("Local Gemma processor is not loaded")
        return getattr(self._processor, "tokenizer", self._processor)


class OpenRouterBackend(BaseInferenceBackend):
    def __init__(self, app_config: AppConfig):
        super().__init__(app_config)
        self._client = None

    def capabilities(self) -> BackendCapabilities:
        logits_mode = self._logits_mode()
        full_logits = logits_mode == "full"
        supports_custom = (
            self.config.supports_custom_logits_override
            if self.config.supports_custom_logits_override is not None
            else False
        )
        if self.config.cfg_enabled:
            if logits_mode == "full":
                cfg_state = "enabled"
            elif logits_mode == "partial":
                cfg_state = "partial"
            elif logits_mode == "unverified":
                cfg_state = "unverified"
            else:
                cfg_state = "disabled"
        else:
            cfg_state = "disabled"
        warning = None
        if self.config.cfg_enabled:
            if logits_mode == "partial":
                warning = (
                    "OpenRouter logprobs probe passed for this model. CFG may be usable in a limited form, "
                    "but provider-side custom logits processors are still unavailable."
                )
            elif logits_mode == "unverified":
                warning = "OpenRouter CFG support is unverified. Run Housekeeping to probe logprobs support for the current model."
            elif logits_mode == "unavailable":
                warning = "CFG disabled: the last OpenRouter housekeeping probe did not find usable logprobs support for this model."
        return BackendCapabilities(
            backend=self.config.backend.value,
            model_name=self.config.model_name,
            logits_mode=logits_mode,
            full_logits_available=full_logits,
            full_vocab_logits=full_logits,
            supports_custom_logits_processors=supports_custom,
            cfg_state=cfg_state,
            model_available=True,
            bootstrap_required=False,
            warning=warning,
        )

    def generate(
        self,
        messages: list[Message],
        *,
        stream_handler: TokenCallback | None = None,
    ) -> InferenceResponse:
        client = self._ensure_client()
        if stream_handler:
            stream = client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )
            text_parts: list[str] = []
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    text_parts.append(token)
                    stream_handler(token)
            return InferenceResponse(text="".join(text_parts), raw=None)

        response = client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=False,
        )
        choice = response.choices[0]
        return InferenceResponse(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason,
            raw=response,
        )

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        api_key = self.api_key()
        if not api_key:
            raise ValueError(
                "OpenRouter API key is required for the openrouter backend"
            )

        from openai import OpenAI

        headers = {"X-Title": self.config.openrouter_app_name}
        if self.config.openrouter_site_url:
            headers["HTTP-Referer"] = self.config.openrouter_site_url

        self._client = OpenAI(
            api_key=api_key,
            base_url=self.config.resolve_base_url(self._dotenv_values),
            default_headers=headers,
        )
        return self._client

    def probe_logprobs_support(self) -> dict[str, Any]:
        client = self._ensure_client()
        try:
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "Reply with exactly READY."}],
                temperature=0,
                max_tokens=4,
                logprobs=True,
                top_logprobs=5,
                stream=False,
            )
        except Exception as exc:
            return {
                "accessible": False,
                "logprobs_supported": False,
                "notes": str(exc),
            }

        choice = response.choices[0]
        logprobs = getattr(choice, "logprobs", None)
        content = getattr(logprobs, "content", None) if logprobs is not None else None
        supported = bool(content)
        notes = None
        if supported:
            top_count = len(getattr(content[0], "top_logprobs", []) or [])
            notes = f"Received token logprobs with {top_count} top-logprob entries on the first generated token."
        else:
            notes = "Request succeeded but the response did not include usable logprobs content."

        return {
            "accessible": True,
            "logprobs_supported": supported,
            "notes": notes,
        }

    def _logits_mode(self) -> str:
        if self.config.full_logits_override is True:
            return "full"
        if self.config.full_logits_override is False:
            return "unavailable"
        if self.config.openrouter_logprobs_supported is True:
            return "partial"
        if self.config.openrouter_logprobs_supported is False:
            return "unavailable"
        return "unverified"


def _cfg_state(cfg_enabled: bool, full_logits: bool) -> str:
    return "enabled" if cfg_enabled and full_logits else "disabled"


def _resolve_workspace_path(workspace_root: str, relative_or_absolute: str) -> Path:
    path = Path(relative_or_absolute)
    if path.is_absolute():
        return path
    return Path(workspace_root) / path


def _resolve_torch_dtype(name: str):
    normalized = name.lower()
    if normalized == "auto":
        return "auto"

    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[normalized]


def render_messages(messages: list[Message]) -> str:
    parts = []
    for message in messages:
        parts.append(f"{message['role'].upper()}:\n{message['content']}")
    parts.append("ASSISTANT:\n")
    return "\n\n".join(parts)


def build_backend(config: AppConfig) -> BaseInferenceBackend:
    if config.inference.backend == BackendType.LOCAL_GEMMA:
        return LocalGemmaBackend(config)
    if config.inference.backend == BackendType.OPENROUTER:
        return OpenRouterBackend(config)
    raise ValueError(f"Unsupported backend: {config.inference.backend}")


def mark_openrouter_probe_result(
    config: AppConfig,
    *,
    logprobs_supported: bool,
    notes: str,
) -> None:
    config.inference.openrouter_logprobs_supported = logprobs_supported
    config.inference.openrouter_cfg_checked_at = datetime.now(timezone.utc).isoformat()
    config.inference.openrouter_cfg_checked_model = config.inference.model_name
    config.inference.openrouter_cfg_probe_notes = notes
