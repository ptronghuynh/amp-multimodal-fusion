"""Independent test-to-train audit using the fixed split and explicit BLASTP settings."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from run_blastp_audit import FIELDS, SPLIT_FILES, EXPECTED_COUNTS, digest, refilter, run_logged, validate_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    membership, sequences, hashes = validate_inputs()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    blastp, makeblastdb = shutil.which('blastp'), shutil.which('makeblastdb')
    if not blastp or not makeblastdb:
        raise SystemExit('NCBI BLAST+ executables are missing from PATH.')
    versions = {Path(exe).name: subprocess.check_output([exe, '-version'], text=True).strip() for exe in (blastp, makeblastdb)}
    (out / 'versions.json').write_text(json.dumps(versions, indent=2), encoding='utf-8')
    commands: list[list[str]] = []
    db = out / 'train_b40cov_db'
    source = out / 'test_vs_train_short_NEW.tsv'
    run_logged([makeblastdb, '-in', str(SPLIT_FILES['train']), '-dbtype', 'prot', '-parse_seqids', '-out', str(db)], out, commands)
    run_logged([blastp, '-task', 'blastp-short', '-query', str(SPLIT_FILES['test']), '-db', str(db),
                '-matrix', 'PAM30', '-gapopen', '9', '-gapextend', '1', '-word_size', '2',
                '-evalue', '1000', '-comp_based_stats', '0', '-seg', 'no', '-soft_masking', 'false',
                '-max_target_seqs', str(EXPECTED_COUNTS['train']), '-num_threads', str(args.threads),
                '-outfmt', '6 ' + ' '.join(FIELDS), '-out', str(source)], out, commands)
    result = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'mode': 'new_explicit_test_to_train_blastp_audit',
              'fresh_blastp_run': True, 'scope': 'test queries against train database', 'split_counts': EXPECTED_COUNTS,
              'input_sha256': hashes, 'versions': versions, 'commands': commands, 'output_tsv_sha256': digest(source)}
    result.update(refilter(source, membership, sequences, out / 'qualifying_cross_split.tsv'))
    (out / 'audit_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
