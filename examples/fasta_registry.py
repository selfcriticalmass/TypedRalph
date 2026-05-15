from __future__ import annotations

from schema import FunctionSchema

from examples.fasta_tools import (
    convert_interleaved_fasta_to_sequential,
    validate_fasta_file,
)


FUNCTION_REGISTRY = [
    FunctionSchema.from_callable(
        convert_interleaved_fasta_to_sequential,
        tags=["fasta", "convert", "interleaved", "sequential"],
        docstring=(
            "Convert an interleaved FASTA alignment into a standard sequential FASTA file. "
            "Use this when a FASTA file is split into blank-line-separated sequence chunks and you need one full sequence per record."
        ),
    ),
    FunctionSchema.from_callable(
        validate_fasta_file,
        tags=["fasta", "validate", "sequence", "quality-control"],
        docstring=(
            "Validate a FASTA file and report whether its sequences contain only legal symbols for a given alphabet. "
            "This is useful for checking peptide, DNA, or RNA FASTA inputs before downstream processing."
        ),
    ),
]
