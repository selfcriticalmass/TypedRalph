from __future__ import annotations

from pathlib import Path


DNA_ALPHABET = set("ACGTRYSWKMBDHVN-?")
RNA_ALPHABET = set("ACGURYSWKMBDHVN-?")
PROTEIN_ALPHABET = set("ABCDEFGHIKLMNOPQRSTUVWYZXJ*-?")


def convert_interleaved_fasta_to_sequential(
    input_path: str,
    output_path: str,
    line_width: int = 80,
) -> dict[str, object]:
    """Convert a simple interleaved FASTA alignment into sequential FASTA.

    This expects a common interleaved layout where the first block contains
    headers plus the first sequence chunk for each record, and later blocks are
    blank-line separated chunk groups listed in the same sequence order.
    """

    records = _read_interleaved_fasta(Path(input_path))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n")
            for start in range(0, len(sequence), line_width):
                handle.write(sequence[start : start + line_width] + "\n")

    return {
        "status": "success",
        "input_path": str(Path(input_path)),
        "output_path": str(output),
        "record_count": len(records),
        "line_width": line_width,
    }


def validate_fasta_file(file_path: str, alphabet: str = "protein") -> dict[str, object]:
    """Validate that a FASTA file has headers, non-empty sequences, and legal symbols."""

    path = Path(file_path)
    records = _read_sequential_fasta(path)
    allowed = _alphabet_for(alphabet)
    errors: list[str] = []
    total_length = 0

    if not records:
        errors.append("File does not contain any FASTA records")

    for index, (header, sequence) in enumerate(records, start=1):
        total_length += len(sequence)
        if not header.strip():
            errors.append(f"Record {index} has an empty header")
        if not sequence:
            errors.append(f"Record {index} ({header}) has an empty sequence")
            continue

        illegal = sorted(
            {character for character in sequence if character not in allowed}
        )
        if illegal:
            joined = "".join(illegal)
            errors.append(
                f"Record {index} ({header}) contains illegal characters: {joined}"
            )

    return {
        "status": "valid" if not errors else "invalid",
        "file_path": str(path),
        "alphabet": alphabet,
        "record_count": len(records),
        "total_sequence_length": total_length,
        "errors": errors,
    }


def _alphabet_for(alphabet: str) -> set[str]:
    normalized = alphabet.strip().lower()
    if normalized == "dna":
        return DNA_ALPHABET
    if normalized == "rna":
        return RNA_ALPHABET
    if normalized == "protein":
        return PROTEIN_ALPHABET
    raise ValueError("alphabet must be one of: dna, rna, protein")


def _read_sequential_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_header: str | None = None
    current_sequence: list[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                records.append((current_header, "".join(current_sequence)))
            current_header = line[1:].strip()
            current_sequence = []
            continue
        if current_header is None:
            raise ValueError(
                f"Encountered sequence data before the first header in {path}"
            )
        current_sequence.append(_normalize_sequence_line(line))

    if current_header is not None:
        records.append((current_header, "".join(current_sequence)))
    return records


def _read_interleaved_fasta(path: Path) -> list[tuple[str, str]]:
    blocks = _split_blocks(path)
    if not blocks:
        raise ValueError(f"No FASTA content found in {path}")

    first_block = blocks[0]
    headers: list[str] = []
    sequences: list[list[str]] = []
    current_header: str | None = None

    for raw_line in first_block:
        line = raw_line.strip()
        if line.startswith(">"):
            current_header = line[1:].strip()
            headers.append(current_header)
            sequences.append([])
            continue
        if current_header is None:
            raise ValueError(
                "The first interleaved FASTA block must begin with headers"
            )
        sequences[-1].append(_normalize_sequence_line(line))

    if not headers:
        raise ValueError("No FASTA headers were found in the first interleaved block")

    for block_index, block in enumerate(blocks[1:], start=2):
        if any(line.lstrip().startswith(">") for line in block):
            raise ValueError(
                f"Interleaved block {block_index} unexpectedly contains FASTA headers"
            )
        if len(block) != len(headers):
            raise ValueError(
                f"Interleaved block {block_index} has {len(block)} lines for {len(headers)} records"
            )
        for index, raw_line in enumerate(block):
            sequences[index].append(_normalize_sequence_line(raw_line.strip()))

    return [(header, "".join(parts)) for header, parts in zip(headers, sequences)]


def _split_blocks(path: Path) -> list[list[str]]:
    blocks: list[list[str]] = []
    current_block: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if current_block:
                blocks.append(current_block)
                current_block = []
            continue
        current_block.append(line)
    if current_block:
        blocks.append(current_block)
    return blocks


def _normalize_sequence_line(line: str) -> str:
    return "".join(line.split()).upper()
