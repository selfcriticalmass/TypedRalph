from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, create_model


def _render_annotation(annotation: Any) -> str:
    if annotation is inspect._empty:
        return "Any"
    if isinstance(annotation, str):
        return annotation
    rendered = getattr(annotation, "__name__", None)
    if rendered:
        return rendered
    return str(annotation).replace("typing.", "")


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


class FunctionSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Unique function name")
    callable: Callable[..., Any]
    input_schema: dict[str, str] = Field(default_factory=dict)
    output_schema: str = "Any"
    docstring: str = ""
    tags: list[str] = Field(default_factory=list)
    module_name: str
    qualname: str
    source_file: str | None = None

    @classmethod
    def from_callable(
        cls,
        target: Callable[..., Any],
        *,
        name: str | None = None,
        tags: list[str] | None = None,
        docstring: str | None = None,
    ) -> "FunctionSchema":
        signature = inspect.signature(target)
        input_schema: dict[str, str] = {}
        for parameter in signature.parameters.values():
            if parameter.kind in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.VAR_POSITIONAL,
            }:
                raise ValueError(
                    f"Function '{target.__name__}' uses unsupported parameter kind '{parameter.kind}'. "
                    "Schema mode expects keyword-callable functions."
                )
            input_schema[parameter.name] = _render_annotation(parameter.annotation)

        module = inspect.getmodule(target)
        qualname = getattr(target, "__qualname__", target.__name__)
        if "<locals>" in qualname:
            raise ValueError(
                f"Function '{target.__name__}' must be defined at module scope for subprocess execution"
            )

        return cls(
            name=name or target.__name__,
            callable=target,
            input_schema=input_schema,
            output_schema=_render_annotation(signature.return_annotation),
            docstring=docstring
            if docstring is not None
            else inspect.getdoc(target) or "",
            tags=tags or [],
            module_name=module.__name__ if module else "__main__",
            qualname=qualname,
            source_file=inspect.getsourcefile(target),
        )

    def input_model(self):
        signature = inspect.signature(self.callable)
        field_definitions: dict[str, tuple[Any, Any]] = {}
        for parameter in signature.parameters.values():
            annotation = (
                parameter.annotation
                if parameter.annotation is not inspect._empty
                else Any
            )
            default = (
                parameter.default if parameter.default is not inspect._empty else ...
            )
            field_definitions[parameter.name] = (annotation, default)
        return create_model(f"{self.name.title()}Input", **field_definitions)

    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        model = self.input_model()
        validated = model.model_validate(args)
        return validated.model_dump()

    def signature_summary(self) -> str:
        pairs = ", ".join(f"{key}: {value}" for key, value in self.input_schema.items())
        return f"{self.name}({pairs}) -> {self.output_schema}"

    def signature_text(self) -> str:
        parts = [
            self.name,
            self.signature_summary(),
            self.output_schema,
            " ".join(self.tags),
        ]
        return " ".join(part for part in parts if part)

    def docstring_summary(self, limit: int = 140) -> str:
        summary = _first_line(self.docstring)
        if len(summary) <= limit:
            return summary
        return summary[: limit - 3] + "..."

    def callable_reference(self) -> dict[str, str | None]:
        return {
            "module_name": self.module_name,
            "qualname": self.qualname,
            "source_file": self.source_file,
        }


class FunctionMatch(BaseModel):
    function_name: str
    signature_summary: str
    docstring_summary: str
    lexical_score: float
    semantic_score: float
    fused_score: float
    tags: list[str] = Field(default_factory=list)


def _load_module_from_file(path: Path):
    module_name = f"codeact_registry_{hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load registry module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coerce_registry_entry(entry: Any) -> FunctionSchema:
    if isinstance(entry, FunctionSchema):
        return entry
    if callable(entry):
        return FunctionSchema.from_callable(entry)
    if isinstance(entry, dict):
        return FunctionSchema.model_validate(entry)
    raise TypeError(f"Unsupported registry entry: {type(entry)!r}")


def load_function_registry(registry_path: str) -> list[FunctionSchema]:
    path = Path(registry_path)
    if path.exists():
        module = _load_module_from_file(path)
        entries = getattr(module, "FUNCTION_REGISTRY", None)
        if entries is None:
            raise AttributeError(f"{path} does not define FUNCTION_REGISTRY")
    else:
        module_path, _, attribute = registry_path.partition(":")
        module = importlib.import_module(module_path)
        attribute_name = attribute or "FUNCTION_REGISTRY"
        entries = getattr(module, attribute_name)

    if callable(entries):
        entries = entries()

    return [_coerce_registry_entry(entry) for entry in entries]


def registry_schema_hash(functions: list[FunctionSchema]) -> str:
    normalized = [
        {
            "name": function.name,
            "signature": function.signature_summary(),
            "docstring": function.docstring,
            "tags": function.tags,
            "reference": function.callable_reference(),
        }
        for function in sorted(functions, key=lambda item: item.name)
    ]
    payload = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
