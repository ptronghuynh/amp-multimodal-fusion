#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
makeblastdb -in all_sequences_5219.fasta -dbtype prot -parse_seqids -out all_db
common=(-task blastp -db all_db -matrix BLOSUM62 -word_size 3 -gapopen 11 -gapextend 1 -seg no -soft_masking false -max_target_seqs 5219 -num_threads 4 -outfmt '6 qseqid sseqid pident length qlen slen evalue bitscore qstart qend sstart send')
blastp "${common[@]}" -query all_sequences_5219.fasta -evalue 10 -comp_based_stats 2 -out standard_blosum62.tsv
blastp "${common[@]}" -query test_queries.fasta -evalue 1000 -comp_based_stats 0 -out permissive_test_vs_all.tsv
: > chunks_status.log
for f in query_[0-9][0-9][0-9][0-9].fasta; do
  stem=${f%.fasta}
  blastp "${common[@]}" -query "$f" -evalue 1000 -comp_based_stats 0 -out "${stem}_permissive.tsv"
  printf 'completed %s %s\n' "$stem" "$(wc -l < "${stem}_permissive.tsv")" >> chunks_status.log
done
printf 'SUCCESS\n' >> chunks_status.log
