from pathlib import Path
import hashlib
import zipfile
import pandas as pd
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
ROOT = args.root
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
manifest = pd.read_csv(ROOT / 'sequence_splits_blast40cov_FINAL.csv')
split = manifest.set_index('sequence_id')['split'].to_dict()
for line in (ROOT / 'b40cov_manifest_sha256.txt').read_text().splitlines():
    digest, name = line.split(maxsplit=1)
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
names = ['q','s','identity','alnlen','qlen','slen','evalue','bitscore']
thresholds = [.5,.6,.7,.8,.9,1.0]
files = {
    'archived_other': ROOT / 'all_vs_all.tsv',
    'fresh_short': ROOT / 'BLASTP_R2_audit/fresh_2_16_0/all_vs_all_short_NEW.tsv',
    'fresh_test_train': ROOT / 'BLASTP_R2_audit/fresh_test_to_train_2_16_0/test_vs_train_short_NEW.tsv',
    'archived_test_train': ROOT / 'blast_test_vs_train_b40cov.tsv',
}
def audit(name, stream):
    total = cross = 0
    counts = {t:0 for t in thresholds}
    pairs = {t:set() for t in thresholds}
    directions = {t:set() for t in thresholds}
    hits = []
    for df in pd.read_csv(stream,sep='\t',names=names,chunksize=300000):
        total += len(df)
        df['qsplit'] = df.q.map(split)
        df['ssplit'] = df.s.map(split)
        assert df.qsplit.notna().all() and df.ssplit.notna().all(), name
        df = df[df.qsplit != df.ssplit].copy()
        cross += len(df)
        df = df[df.identity >= 40].copy()
        df['qcov'] = df.alnlen / df.qlen
        df['scov'] = df.alnlen / df.slen
        df['mincov'] = np.minimum(df.qcov,df.scov)
        for t in thresholds:
            sub = df[df.mincov >= t]
            counts[t] += len(sub)
            pairs[t].update(tuple(sorted(x)) for x in zip(sub.q,sub.s))
            directions[t].update(zip(sub.q,sub.s))
        if name == 'archived_other':
            hits.append(df[df.mincov >= .8])
    rows = [dict(source=name,identity_min=40,coverage_min=t,all_rows=total,cross_rows=cross,qualifying_rows=counts[t],unique_unordered_pairs=len(pairs[t]),unique_ordered_pairs=len(directions[t])) for t in thresholds]
    if hits:
        pd.concat(hits,ignore_index=True).to_csv(OUT/'archived_68_rows.csv',index=False)
    print(name, [(x['coverage_min'],x['qualifying_rows'],x['unique_unordered_pairs']) for x in rows],flush=True)
    return rows,pairs[.8]
results = []
for name,p in files.items():
    rows,pairs = audit(name,p)
    results += rows
    if name == 'archived_other':
        pairs_old = pairs
with zipfile.ZipFile(ROOT/'audit_out.zip') as z:
    with z.open('audit_out/all_vs_all_short.tsv') as f:
        rows,pairs = audit('archived_short',f)
        results += rows
pd.DataFrame(results).to_csv(OUT/'coverage_sensitivity.csv',index=False)
for name,path in [('archived_other',files['archived_other']),('fresh_short',files['fresh_short'])]:
    sub = []
    ids = set(x for p in pairs_old for x in p)
    for df in pd.read_csv(path,sep='\t',names=names,chunksize=300000):
        df = df[df.q.isin(ids)&df.s.isin(ids)].copy()
        keep = [tuple(sorted(x)) in pairs_old for x in zip(df.q,df.s)]
        df = df.loc[keep].copy()
        df['qcov'] = df.alnlen/df.qlen
        df['scov'] = df.alnlen/df.slen
        sub.append(df)
    out = pd.concat(sub,ignore_index=True)
    out.to_csv(OUT/f'{name}_38_pairs.csv',index=False)
    print(name,'38-pair alignment rows',len(out),'pairs',len(set(tuple(sorted(x)) for x in zip(out.q,out.s))),flush=True)
# Pair-level diagnosis: do not infer undocumented historical search options.
a = pd.read_csv(OUT/'archived_68_rows.csv')
b = pd.read_csv(OUT/'fresh_short_38_pairs.csv')
for df in [a,b]:
    df['pair'] = ['|'.join(sorted(x)) for x in zip(df.q,df.s)]
    df['mincov'] = df[['qcov','scov']].min(axis=1)
summary = []
for key, group in a.groupby('pair'):
    n = b[b.pair == key]
    assert not ((n.identity >= 40) & (n.mincov >= .8)).any()
    if len(n) == 0:
        reason = 'no reported HSP'
    elif (n.identity >= 40).any():
        reason = 'coverage below 0.8 for all identity-qualified HSPs'
    else:
        reason = 'no identity-qualified HSP'
    summary.append(dict(pair=key,archived_qualifying_rows=len(group),archived_max_min_coverage=group.mincov.max(),fresh_rows=len(n),fresh_max_min_coverage=n.mincov.max(),fresh_max_min_coverage_at_identity40=n.loc[n.identity>=40,'mincov'].max(),reason=reason))
s = pd.DataFrame(summary)
s.to_csv(OUT/'pair_diagnostic.csv',index=False)
print('Pair-level diagnosis:',s.reason.value_counts().to_dict())
