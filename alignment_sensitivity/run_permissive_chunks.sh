#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
B=./ncbi-blast-2.16.0+/bin/blastp
for f in query_[0-9][0-9][0-9][0-9].fasta; do
  stem=${f%.fasta}
  /usr/bin/time -f 'elapsed=%e sec' "$B" -task blastp -query "$f" -db all_db -matrix BLOSUM62 -word_size 3 -gapopen 11 -gapextend 1 -evalue 1000 -comp_based_stats 0 -seg no -soft_masking false -max_target_seqs 5219 -num_threads 4 -outfmt '6 qseqid sseqid pident length qlen slen evalue bitscore qstart qend sstart send' -out "${stem}_permissive.tsv" 2>"${stem}_permissive.log"
  echo "completed ${stem} $(wc -l < "${stem}_permissive.tsv")" >> chunks_status.log
done
printf 'SUCCESS\n' >> chunks_status.log
