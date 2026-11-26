from pathlib import Path
import hashlib
import json
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'all_sequences_5219.fasta'
manifest = pd.read_csv(ROOT / 'original_manifest.csv').set_index('id')
partition = pd.read_csv(ROOT / 'sequence_splits_blast40cov_FINAL.csv').set_index('sequence_id')['split']
assert len(manifest) == len(partition) == 5219
assert manifest['split'].sort_index().equals(partition.sort_index())
sequences = {r.id: str(r.seq) for r in SeqIO.parse(DATA, 'fasta')}
assert set(sequences) == set(partition.index)
assert len(set(sequences.values())) == 5219
cols = 'q s identity alnlen qlen slen evalue bitscore qstart qend sstart send'.split()
cutoffs = [0.5,0.6,0.7,0.8,0.9,1.0]
seeds = [42,52,62,72,82]
models = ['Seq-only','Phys-only','Concat','Cross-attn','Concat-matched','Concat-active-matched']
contrasts = [('Concat','Seq-only'),('Cross-attn','Seq-only'),('Cross-attn','Concat'),('Cross-attn','Concat-matched'),('Cross-attn','Concat-active-matched')]


def read_alignments(paths):
    frames = []
    n = 0
    seen = set()
    for path in paths:
        for d in pd.read_csv(path, sep='\t', names=cols, chunksize=200000):
            n += len(d)
            seen.update(d.q.unique())
            d['qsplit'] = d.q.map(partition)
            d['ssplit'] = d.s.map(partition)
            assert d.qsplit.notna().all() and d.ssplit.notna().all()
            assert np.all(d.qlen == d.q.map(lambda q:len(sequences[q])))
            assert np.all(d.slen == d.s.map(lambda q:len(sequences[q])))
            d = d[(d.qsplit != d.ssplit) & (d.identity >= 40)].copy()
            frames.append(d)
    return pd.concat(frames, ignore_index=True), n, seen


def pairs_at(d, cov):
    hit = d[(d.alnlen >= cov*d.qlen - 1e-9)&(d.alnlen >= cov*d.slen - 1e-9)].copy()
    return hit, {tuple(sorted((q,s))) for q,s in zip(hit.q,hit.s)}


def categories(pairs):
    return Counter('/'.join(sorted((partition[a],partition[b]))) for a,b in pairs)


def endpoints(pairs):
    return {x for pair in pairs for x in pair if partition[x]=='test'}


std,n,seen = read_alignments([ROOT/'standard_blosum62.tsv'])
assert set(partition.index)-seen == {'AMP_91','AMP_807'}
std_hits,std_pairs=pairs_at(std,0.8)
with zipfile.ZipFile(ROOT/'VNUHCM_R2_Coverage_Audit_Evidence.zip') as z:
    a=pd.read_csv(z.open('archived_68_rows.csv'))
assert (a.qsplit==a.q.map(partition)).all() and (a.ssplit==a.s.map(partition)).all()
assert (a.qlen==a.q.map(lambda q:len(sequences[q]))).all() and (a.slen==a.s.map(lambda q:len(sequences[q]))).all()
a_hits,a_pairs=pairs_at(a,0.8)
assert len(a_hits)==68 and len(a_pairs)==38
ptest,np_test,pseen=read_alignments([ROOT/'permissive_test_vs_all.tsv'])
assert pseen == set(partition[partition=='test'].index)
pt_hits,pt_pairs=pairs_at(ptest,0.8)
searches={'set_A':(a_hits,a_pairs,68),'BLOSUM62_standard':(std_hits,std_pairs,n),'BLOSUM62_permissive_test_queries':(pt_hits,pt_pairs,np_test)}
paths = sorted(ROOT.glob('query_[0-9][0-9][0-9][0-9]_permissive.tsv'))
status = ROOT/'chunks_status.log'
if status.exists() and status.read_text().endswith('SUCCESS\n'):
    assert len(paths)==6
    full,n_full,fullseen=read_alignments(paths)
    assert fullseen==set(partition.index)
    full_hits,full_pairs=pairs_at(full,0.8)
    searches['BLOSUM62_permissive_all']=(full_hits,full_pairs,n_full)
    # A test-query-only search on the same database must reproduce the same qualifying records.
    assert full_pairs.issuperset(pt_pairs)
else:
    full=None

union=set().union(*(s[1] for s in searches.values()))
searches['union']=(None,union,None)
summary=[]
for name,(hits,pairs,records) in searches.items():
    c=categories(pairs);test=endpoints(pairs)
    summary.append(dict(search=name,records=records,qualifying_rows=len(hits) if hits is not None else None,unique_pairs=len(pairs),train_test=c['test/train'],val_test=c['test/val'],train_val=c['train/val'],test_endpoints=len(test),amp_test_endpoints=sum(manifest.loc[x,'label']==1 for x in test),nonamp_test_endpoints=sum(manifest.loc[x,'label']==0 for x in test)))
    if hits is not None:
        hits.to_csv(ROOT/f'{name}_qualifying_alignments.csv',index=False)
    pd.DataFrame([(a,b,partition[a],partition[b]) for a,b in sorted(pairs)],columns=['id_a','id_b','split_a','split_b']).to_csv(ROOT/f'{name}_qualifying_pairs.csv',index=False)
    pd.DataFrame({'id':sorted(test)}).assign(label=lambda df:df.id.map(manifest.label)).to_csv(ROOT/f'{name}_removed_test_ids.csv',index=False)
S=pd.DataFrame(summary)
S.to_csv(ROOT/'search_summary.csv',index=False)
print(S.to_string(index=False),flush=True)
print('standard / set A overlap',len(std_pairs&a_pairs),'A-only',len(a_pairs-std_pairs),'std-only',len(std_pairs-a_pairs),flush=True)
cv=[]
for name,df in [('BLOSUM62_standard',std)]+([('BLOSUM62_permissive_all',full)] if full is not None else []):
    for cov in cutoffs:
        hits,ps=pairs_at(df,cov)
        cv.append(dict(search=name,coverage=cov,unique_cross_pairs=len(ps),unique_test_train_pairs=categories(ps)['test/train'],unique_test_val_pairs=categories(ps)['test/val'],unique_train_val_pairs=categories(ps)['train/val']))
pd.DataFrame(cv).to_csv(ROOT/'coverage_sensitivity_blosum62.csv',index=False)

P=pd.read_csv(ROOT/'original_predictions.csv')
assert len(P)==6*5*(521+523)
assert set(P.model)==set(models) and set(P.seed)==set(seeds)
assert P.groupby(['model','seed','split']).size().value_counts().to_dict()=={521:30,523:30}
assert (P['label']==P.id.map(manifest.label)).all()
assert (P['split']==P.id.map(partition)).all()
T=P[P.split=='test']
assert len(T)==15630
assert T.groupby('id').size().eq(30).all()
piv=T.pivot(index='id',columns=['model','seed'],values='prob')
y=manifest.loc[piv.index,'label'].to_numpy()
assert len(piv)==521 and (y==1).sum()==271 and (y==0).sum()==250
assert np.isfinite(piv.to_numpy()).all()


def evaluate(name,drop):
    kept=~piv.index.isin(drop)
    yy=y[kept]; pp=piv.loc[kept]
    pos=np.flatnonzero(yy==1);neg=np.flatnonzero(yy==0)
    rows=[]
    score={}
    for model in models:
        for seed in seeds:
            p=pp[(model,seed)].to_numpy()
            # Tie-aware AUC as in the original full-cohort script.
            score[model,seed]=(p[pos,None]>p[None,neg]).astype(np.float32)+.5*(p[pos,None]==p[None,neg])
            au=roc_auc_score(yy,p);assert np.isclose(au,score[model,seed].mean(),atol=1e-7)
            tn,fp,fn,tp=confusion_matrix(yy,p>=.5,labels=[0,1]).ravel()
            rows.append(dict(cohort=name,n=len(yy),n_amp=len(pos),n_nonamp=len(neg),model=model,seed=seed,AUC=au,AP=average_precision_score(yy,p),MCC=matthews_corrcoef(yy,p>=.5),Sensitivity=tp/(tp+fn),Specificity=tn/(tn+fp),TP=tp,TN=tn,FP=fp,FN=fn))
    rng=np.random.default_rng(20261125)
    b=[]; dist={}
    for aa,bb in contrasts:
        d=np.mean([score[aa,sd]-score[bb,sd] for sd in seeds],axis=0,dtype=np.float64)
        vals=np.array([d[np.ix_(rng.integers(0,len(pos),size=len(pos)),rng.integers(0,len(neg),size=len(neg)))].mean() for _ in range(2000)])
        k=aa+' - '+bb;dist[k]=vals
        b.append(dict(cohort=name,n=len(yy),n_amp=len(pos),n_nonamp=len(neg),comparison=k,delta=float(d.mean()),bootstrap_sd=float(vals.std(ddof=1)),ci_low=float(np.percentile(vals,2.5)),ci_high=float(np.percentile(vals,97.5)),bootstrap_fraction_nonpositive=float(np.mean(vals<=0))))
    pd.DataFrame(dist).to_csv(ROOT/f'bootstrap_draws_{name}.csv',index=False)
    return pd.DataFrame(rows),pd.DataFrame(b)

cohorts={'full':set(),'standard_filtered':endpoints(std_pairs),'union_filtered':endpoints(union)}
if full is None:
    # Include the completed permissive test-query assessment, not the interrupted all-vs-all attempt.
    assert endpoints(pt_pairs).issubset(cohorts['union_filtered'])
results=[];boot=[]
for name,drop in cohorts.items():
    r,b=evaluate(name,drop);results.append(r);boot.append(b)
    print(b.to_string(index=False),flush=True)
R=pd.concat(results,ignore_index=True);B=pd.concat(boot,ignore_index=True)
original=pd.read_csv(ROOT/'original_bootstrap.csv')
actual=B[B.cohort=='full'].reset_index(drop=True)
assert actual.comparison.equals(original.comparison)
assert np.allclose(actual[['delta','bootstrap_sd','ci_low','ci_high','bootstrap_fraction_nonpositive']],original[['delta','bootstrap_sd','ci_low','ci_high','bootstrap_fraction_nonpositive']],atol=1e-10)
R.to_csv(ROOT/'per_seed_metrics_sensitivity.csv',index=False)
B.to_csv(ROOT/'paired_contrasts_sensitivity.csv',index=False)
M=R.groupby(['cohort','model','n','n_amp','n_nonamp'],sort=False)[['AUC','AP','MCC','Sensitivity','Specificity']].agg(['mean','std'])
M.columns=['_'.join(c) for c in M.columns]
M.reset_index().to_csv(ROOT/'metrics_summary_sensitivity.csv',index=False)
print(M.round(4).to_string(),flush=True)
print('PASS: exact full-cohort bootstrap reproduced and all masked cohorts evaluated',flush=True)
