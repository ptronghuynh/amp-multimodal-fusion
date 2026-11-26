"""Independently rerun the BLASTP cross-split audit on the fixed R1 FASTA files.

This is a new explicit configuration, NOT a reconstruction of the historical command.
Python 3.9+ standard library; NCBI BLAST+ required for a fresh run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
SPLIT_FILES = {s: ROOT / f"{s}_b40cov.fasta" for s in ("train", "val", "test")}
MANIFEST = ROOT / "sequence_splits_blast40cov_FINAL.csv"
EXPECTED_COUNTS = {"train": 4175, "val": 523, "test": 521}
FIELDS = ("qseqid", "sseqid", "pident", "length", "qlen", "slen", "evalue", "bitscore")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def normalize_id(value: str) -> str:
    return value.removeprefix("lcl|")


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    key = None
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith(">"):
            key = normalize_id(line[1:].split()[0])
            if key in records:
                raise ValueError(f"Duplicate FASTA ID: {key}")
            records[key] = ""
        elif line.strip():
            if key is None:
                raise ValueError(f"Sequence before header: {path}")
            records[key] += line.strip().upper()
    if any(not seq for seq in records.values()):
        raise ValueError(f"Empty sequence: {path}")
    return records


def validate_inputs() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    hashes = {}
    for line in (ROOT / "b40cov_manifest_sha256.txt").read_text().splitlines():
        if not line.strip():
            continue
        sha, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        actual = digest(ROOT / name)
        if actual != sha:
            raise ValueError(f"SHA-256 mismatch: {name}")
        hashes[name] = actual
    if set(hashes) != {p.name for p in SPLIT_FILES.values()} | {MANIFEST.name}:
        raise ValueError("Incomplete hash manifest")
    membership: dict[str, str] = {}
    sequences: dict[str, str] = {}
    for split, path in SPLIT_FILES.items():
        records = read_fasta(path)
        if len(records) != EXPECTED_COUNTS[split] or set(records) & set(sequences):
            raise ValueError(f"Invalid count or overlapping IDs: {split}")
        membership.update({key: split for key in records})
        sequences.update(records)
    with MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(sequences) or {r["sequence_id"]: r["split"] for r in rows} != membership:
        raise ValueError("FASTA membership differs from CSV split manifest")
    if len(set(sequences.values())) != len(sequences):
        raise ValueError("Duplicate sequences in fixed cohort")
    return membership, sequences, hashes


def refilter(path: Path, membership: dict[str, str], sequences: dict[str, str], hits: Path) -> dict:
    rows = cross_rows = qualifying = 0
    unique_pairs: set[tuple[str, str]] = set()
    direction_rows: Counter = Counter()
    with path.open(newline="", encoding="ascii") as src, hits.open("w", newline="", encoding="ascii") as dst:
        writer = csv.writer(dst, delimiter="\t")
        writer.writerow((*FIELDS, "query_split", "subject_split"))
        for row in csv.reader(src, delimiter="\t"):
            if not row:
                continue
            if len(row) != len(FIELDS):
                raise ValueError(f"Expected {len(FIELDS)} columns, got {len(row)} at row {rows + 1}")
            rows += 1
            q, s = normalize_id(row[0]), normalize_id(row[1])
            qs, ss = membership[q], membership[s]
            identity, aln, qlen, slen = float(row[2]), int(row[3]), int(row[4]), int(row[5])
            if qlen != len(sequences[q]) or slen != len(sequences[s]):
                raise ValueError(f"Sequence length mismatch at row {rows}")
            if qs == ss:
                continue
            cross_rows += 1
            if identity >= 40 and 5 * aln >= 4 * qlen and 5 * aln >= 4 * slen:
                qualifying += 1
                unique_pairs.add(tuple(sorted((q, s))))
                direction_rows[f"{qs}->{ss}"] += 1
                writer.writerow((*row, qs, ss))
    return {"alignment_rows": rows, "cross_split_alignment_rows": cross_rows,
            "qualifying_cross_split_alignment_rows": qualifying,
            "qualifying_unique_unordered_pairs": len(unique_pairs),
            "qualifying_rows_by_direction": dict(sorted(direction_rows.items())),
            "criteria": {"pident_min": 40, "alignment_length_over_query_length_min": 0.8,
                         "alignment_length_over_subject_length_min": 0.8}}


def run_logged(cmd: list[str], out: Path, commands: list[list[str]]) -> None:
    commands.append(cmd)
    (out / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
    with (out / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(cmd) + "\n")
        stream.flush()
        subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory (must not exist)")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--archived-tsv", type=Path, help="Only refilter a saved 8-column TSV; not a fresh BLASTP run")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    membership, sequences, hashes = validate_inputs()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    base = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "split_counts": EXPECTED_COUNTS,
            "input_sha256": hashes}
    if args.archived_tsv:
        source = args.archived_tsv.resolve()
        base.update({"mode": "archived_output_refilter_only", "fresh_blastp_run": False,
                     "input_tsv": str(source), "input_tsv_sha256": digest(source)})
    else:
        blastp, makeblastdb = shutil.which("blastp"), shutil.which("makeblastdb")
        if not blastp or not makeblastdb:
            raise SystemExit("NCBI BLAST+ executables are missing from PATH. No fresh audit was performed.")
        versions = {Path(exe).name: subprocess.check_output([exe, "-version"], text=True).strip()
                    for exe in (blastp, makeblastdb)}
        (out / "versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
        combined = out / "fixed_5219.fasta"
        with combined.open("w", encoding="ascii") as stream:
            for key, seq in sequences.items():
                stream.write(f">{key}\n{seq}\n")
        db = out / "fixed_5219_db"
        source = out / "all_vs_all_short_NEW.tsv"
        commands: list[list[str]] = []
        run_logged([makeblastdb, "-in", str(combined), "-dbtype", "prot", "-parse_seqids", "-out", str(db)], out, commands)
        run_logged([blastp, "-task", "blastp-short", "-query", str(combined), "-db", str(db),
                    "-matrix", "PAM30", "-gapopen", "9", "-gapextend", "1", "-word_size", "2",
                    "-evalue", "1000", "-comp_based_stats", "0", "-seg", "no", "-soft_masking", "false",
                    "-max_target_seqs", str(len(sequences)), "-num_threads", str(args.threads),
                    "-outfmt", "6 " + " ".join(FIELDS), "-out", str(source)], out, commands)
        base.update({"mode": "new_explicit_blastp_audit", "fresh_blastp_run": True,
                     "versions": versions, "commands": commands, "output_tsv_sha256": digest(source),
                     "scope": "all cross-split reported HSPs, not just test-to-train"})
    base.update(refilter(source, membership, sequences, out / "qualifying_cross_split.tsv"))
    (out / "audit_summary.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
    print(json.dumps(base, indent=2))


if __name__ == "__main__":
    main()
