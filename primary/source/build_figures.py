"""Reproduce the seven manuscript figures using the locked full-cohort evidence.

No model fitting or prediction modification is performed. Vector PDF and SVG are generated from common Matplotlib artists; SVGs can be
converted to EMF for Word using LibreOffice Draw.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_curve, precision_score, roc_auc_score,
    roc_curve,
)

MAIN = ['Seq-only', 'Phys-only', 'Concat', 'Cross-attn']
MODELS = MAIN + ['Concat-matched', 'Concat-active-matched']
SEEDS = [42, 52, 62, 72, 82]
PAIRS = [('Concat', 'Seq-only'), ('Cross-attn', 'Seq-only'),
         ('Cross-attn', 'Concat'), ('Cross-attn', 'Concat-matched'),
         ('Cross-attn', 'Concat-active-matched')]
COLORS = dict(zip(MAIN, ['#2878B5', '#8B8F94', '#D47718', '#27875E']))
LINESTYLES = dict(zip(MAIN, ['-', '--', '-', '-.']))
NAMES = ['KD hydropathy', 'Basic fraction', 'Acidic fraction', 'Aromatic fraction',
         'Hydrophobic fraction', 'Aliphatic fraction', 'Sulfur fraction']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.labelsize': 10, 'axes.titlesize': 10, 'axes.titleweight': 'semibold',
    'legend.fontsize': 8.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'path', 'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'xtick.major.size': 3.0, 'ytick.major.size': 3.0, 'axes.unicode_minus': True})


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(data: Path, out: Path):
    pred = pd.read_csv(data / 'sevenD_predictions.csv')
    manifest = pd.read_csv(data / 'full_manifest.csv')
    assert len(manifest) == manifest.id.nunique() == 5219
    assert manifest.groupby('split').size().to_dict() == {'test': 521, 'train': 4175, 'val': 523}
    assert manifest.groupby('split').label.sum().to_dict() == {'test': 271, 'train': 2134, 'val': 274}
    assert len(pred) == 30 * (523 + 521)
    assert not pred.duplicated(['model', 'seed', 'split', 'id']).any()
    assert set(pred.model) == set(MODELS) and set(pred.seed) == set(SEEDS)
    assert pred.prob.between(0, 1).all() and np.isfinite(pred.prob).all()
    for (model, seed, split), g in pred.groupby(['model', 'seed', 'split']):
        expected = manifest[manifest.split.eq(split)].set_index('id').label.sort_index()
        actual = g.set_index('id').label.sort_index()
        pd.testing.assert_series_equal(actual, expected, check_names=False)
    saved = pd.read_csv(data / 'verified_metrics.csv')
    records = []
    for (model, seed), g in pred[pred.split.eq('test')].groupby(['model', 'seed']):
        y = g.label.to_numpy(); prob = g.prob.to_numpy()
        for mode in ['fixed', 'validation_MCC']:
            row = saved[(saved.model.eq(model)) & (saved.seed.eq(seed)) & (saved['mode'].eq(mode))].iloc[0]
            th = float(row.threshold)
            if mode == 'fixed':
                assert th == 0.5
            else:
                val = pred[(pred.model.eq(model)) & (pred.seed.eq(seed)) & pred.split.eq('val')]
                vy = val.label.to_numpy(); vp = val.prob.to_numpy()
                candidates = np.unique(np.r_[0., vp, 1.])
                mcc = np.array([matthews_corrcoef(vy, vp >= t) for t in candidates])
                best = np.flatnonzero(mcc == mcc.max())
                selected = candidates[best[np.argmin(abs(candidates[best]-.5))]]
                assert abs(th - selected) < 1e-12, (model, seed, th, selected)
            binary = prob >= th
            tn, fp, fn, tp = confusion_matrix(y, binary, labels=[0,1]).ravel()
            sen = tp / (tp + fn); spe = tn / (tn + fp)
            metrics = dict(AUC=roc_auc_score(y, prob), AP=average_precision_score(y, prob),
                ACC=accuracy_score(y, binary), F1=f1_score(y, binary, zero_division=0),
                MCC=matthews_corrcoef(y, binary), Sensitivity=sen, Specificity=spe,
                Precision=precision_score(y, binary, zero_division=0),
                Balanced_accuracy=(sen+spe)/2, FPR=1-spe, FNR=1-sen,
                TP=tp, TN=tn, FP=fp, FN=fn, threshold=th)
            for k, v in metrics.items():
                assert abs(row[k] - v) < 1e-10, (model,seed,mode,k,row[k],v)
            records.append(dict(model=model, seed=int(seed), mode=mode, **metrics))
    metrics = pd.DataFrame(records)
    metrics.to_csv(out / 'figure_source_metrics.csv', index=False)
    summary = metrics.groupby(['model','mode'])[list(metrics.columns[3:])].agg(['mean','std'])
    summary.columns = ['_'.join(x) for x in summary.columns]
    stored = pd.read_csv(data / 'verified_summary.csv').set_index(['model','mode'])
    assert np.allclose(summary.values,stored.loc[summary.index,summary.columns].values,atol=1e-12,rtol=1e-12)
    summary.to_csv(out / 'figure_source_summary.csv')
    test = pred[pred.split.eq('test')]
    pivot = test.pivot(index='id', columns=['model','seed'], values='prob')
    y = test.drop_duplicates('id').set_index('id').loc[pivot.index].label.to_numpy()
    assert len(y) == 521 and y.sum() == 271
    pos = np.flatnonzero(y==1); neg = np.flatnonzero(y==0)
    score = {k: (pivot[k].to_numpy()[pos,None] > pivot[k].to_numpy()[None,neg]).astype(np.float32)
             + .5*(pivot[k].to_numpy()[pos,None] == pivot[k].to_numpy()[None,neg]) for k in pivot.columns}
    rng = np.random.default_rng(20261125)
    draws = {}; rows = []
    for a,b in PAIRS:
        diff = np.mean([score[a,s]-score[b,s] for s in SEEDS],axis=0,dtype=np.float64)
        key = a+' - '+b
        v = np.array([diff[np.ix_(rng.integers(0,len(pos),size=len(pos)),
                    rng.integers(0,len(neg),size=len(neg)))].mean() for _ in range(2000)])
        draws[key] = v
        rows.append(dict(comparison=key,delta=diff.mean(),bootstrap_sd=v.std(ddof=1),
            ci_low=np.percentile(v,2.5),ci_high=np.percentile(v,97.5),
            bootstrap_fraction_nonpositive=np.mean(v<=0)))
    boot = pd.DataFrame(rows).set_index('comparison')
    old = pd.read_csv(data / 'verified_bootstrap.csv').set_index('comparison')
    assert np.allclose(boot.values,old.loc[boot.index,boot.columns].values,atol=1e-12,rtol=1e-12)
    draws = pd.DataFrame(draws)
    assert np.allclose(draws.values,pd.read_csv(data/'bootstrap_resamples.csv')[draws.columns].values,atol=1e-12,rtol=1e-12)
    boot.to_csv(out/'figure_source_bootstrap.csv')
    draws.to_csv(out/'figure_source_bootstrap_resamples.csv',index=False)
    return metrics, summary, pivot, y, boot, draws


def save(fig, out, number):
    stem = out / f'Figure_{number:02d}'
    for ext in ['pdf','svg']:
        fig.savefig(stem.with_suffix('.'+ext), bbox_inches='tight', pad_inches=0.10,
                    metadata={'Creator':'Reproducible manuscript figure generator'})
    plt.close(fig)


def architecture(out):
    # Same four primary architecture categories and separate 64-D context as Table 2.
    # Only the graphic format/layout changes; all operations and dimensions are preserved.
    fig = plt.figure(figsize=(7.0,7.0)); ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis('off')
    navy = '#14304D'; red = '#AE2222'; ink = '#152738'; seq = '#E8F1FA'; phys = '#E9F2E7'; attn = '#FFF1DF'; output = '#E6F2EB'
    def panel(x,y,w,h,title):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.18,rounding_size=1.3',fc='#FFFFFF',ec='#8B9CAA',lw=0.9))
        ax.text(x+1.6,y+h-2.8,title,ha='left',va='center',weight='bold',color=navy,fontsize=10.4)
    def box(x,y,w,h,text,fc=seq,tc=ink,fs=9.2):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.16,rounding_size=0.8',fc=fc,ec='#7890A4',lw=0.8))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',color=tc,fontsize=fs,linespacing=1.13)
    def arrow(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,lw=0.9,color=ink,shrinkA=0,shrinkB=0))
    ax.text(50,98.3,'Verified downstream architectures',ha='center',weight='bold',fontsize=12.5,color=navy)
    ax.text(50,95.4,'7-D primary and archived reference; 64-D separate configuration',ha='center',fontsize=9.1,color=ink)
    panel(1,67.0,48,25.5,'(A) Seq-only'); panel(51,67.0,48,25.5,'(B) Phys-only')
    box(3,82,44,5,'Frozen ESM-2 mean pooling: s (320-D)'); box(53,82,44,5,'Physicochemical descriptors: p (d-D)',phys)
    for cx,x,fc,txt in [(25,3,seq,'7-D: Linear 320 → 1\n64-D: MLP 320 → 128 → 1'),(75,53,phys,'7-D: MLP 7 → 64 → 1\n64-D: MLP 64 → 128 → 1')]:
        box(x,74,44,6,txt,fc,red); arrow((cx,82),(cx,80)); arrow((cx,74),(cx,72.5)); box(x,68.4,44,4.1,'Logit z → sigmoid → AMP probability',output,fs=8.5)
    panel(1,45.0,98,20.2,'(C) Concat (early fusion)')
    box(3,54.5,20,5,'Pooled sequence\ns (320-D)',seq,fs=8.7);box(3,47.4,20,5,'Descriptors\np (d-D)',phys,fs=8.7)
    box(29,50.2,22,7,'x = [s; p]\n327-D / 384-D',seq,red,fs=9.2);box(57,50.2,23,7,'MLP\n(320+d) → 256 → 1',seq,red,fs=9.1);box(86,50.2,11,7,'Logit z\n↓ sigmoid',output,fs=8.6)
    for y0,yy in [(57.0,55.2),(49.9,52.3)]:
        ax.plot([23,26,26],[y0,y0,yy],color=ink,lw=.9);arrow((26,yy),(29,yy))
    arrow((51,53.7),(57,53.7));arrow((80,53.7),(86,53.7))
    panel(1,2.0,98,41.2,'(D) Proposed configuration (Cross-attn)')
    ax.text(50,36.8,'d = 7, H = 256 (primary)   |   d = 64, H = 128 (separate)',ha='center',fontsize=9.4,color=red)
    box(3,26.9,20,6.3,'Pooled sequence\ns (320-D)',seq,fs=8.7);box(3,18.0,20,6.3,'Descriptors\np (d-D)',phys,fs=8.7)
    box(28,26.9,20,6.3,'Linear 320 → H\nu: one token',attn,red,fs=8.8);box(28,18,20,6.3,'Linear d → H\nv: one token',attn,red,fs=8.8)
    box(54,18,43,15.2,'Shared multi-head attention (4 heads)\na = MHA(u, v, v)\nb = MHA(v, u, u)\nEach output: H-D; one key per query',attn,fs=9.1)
    for y0 in [30.05,21.15]:arrow((23,y0),(28,y0));arrow((48,y0),(54,y0))
    box(70,7.1,27,7.6,'Concatenate [a; b]\n512-D / 256-D (2H)',attn,red,fs=9.0)
    box(37,7.1,27,7.6,'MLP\n2H → H → 1',attn,red);box(3,7.1,28,7.6,'Logit z\n↓ sigmoid',output)
    arrow((83.5,18),(83.5,14.7));arrow((70,10.9),(64,10.9));arrow((37,10.9),(31,10.9))
    ax.text(50,4.0,'All MLPs use one ReLU hidden layer; ESM-2 is frozen. See Table 2.',ha='center',fontsize=8.6,color=ink)
    save(fig,out,1)


def lighten(c,weight=.14):
    return tuple(1-weight*(1-v) for v in to_rgb(c))


def create_plots(data,out,metrics,summary,piv,y,boot,draws):
    fig,ax=plt.subplots(figsize=(6.45,3.25));vals=[summary.loc[(m,'fixed'),'AUC_mean'] for m in MAIN];sds=[summary.loc[(m,'fixed'),'AUC_std'] for m in MAIN]
    ax.bar(range(4),vals,yerr=sds,width=.62,color=[COLORS[m] for m in MAIN],edgecolor='#303943',linewidth=.6,capsize=4,error_kw={'elinewidth':1.15,'capthick':1.1})
    ax.set(ylim=(0,1.04),ylabel='ROC-AUC',xticks=range(4),xticklabels=MAIN);ax.set_yticks(np.arange(0,1.01,.2));ax.grid(axis='y',color='#E3E8EC',lw=.65);ax.set_axisbelow(True)
    ax.axhline(.5,ls='--',color='#77838A',lw=.8)
    for i,(v,s) in enumerate(zip(vals,sds)):ax.text(i,v+s+.023,f'{v:.4f} ± {s:.4f}',ha='center',fontsize=9.0)
    fig.text(.5,.015,'521 test peptides · 5 seeds',ha='center',va='bottom',fontsize=8.5,color='#4B5965')
    fig.tight_layout(rect=[0,.075,1,1]);save(fig,out,2)
    fig,axs=plt.subplots(1,2,figsize=(6.7,3.15),sharey=True)
    for ax,k in zip(axs,['Cross-attn - Seq-only','Cross-attn - Concat']):
        v=draws[k];row=boot.loc[k];ax.hist(v,bins=32,color='#527FA9',edgecolor='white',linewidth=.35);ax.axvline(0,ls='--',color='#232B32',lw=1.1);ax.axvline(row.delta,color='#164B76',lw=1.4)
        ax.axvline(row.ci_low,color='#A34428',lw=1.1,ls=':');ax.axvline(row.ci_high,color='#A34428',lw=1.1,ls=':')
        ax.set_title(k.replace(' - ',' − '),pad=10);ax.set_xlabel('Paired ROC-AUC difference');ax.grid(axis='y',color='#E8ECEF',lw=.65);ax.set_axisbelow(True)
        ax.text(.5,.98,f'Δ = {row.delta:+.4f}\n95% CI [{row.ci_low:+.4f}, {row.ci_high:+.4f}]',transform=ax.transAxes,ha='center',va='top',fontsize=8.1,bbox=dict(boxstyle='round,pad=.2',fc='white',ec='#D0D8DD',lw=.5))
        ax.margins(x=.06);ax.set_ylim(0, max(ax.get_ylim()[1],300))
    axs[0].set_ylabel('Frequency');fig.text(.5,.01,'2,000 paired class-stratified resamples; dotted lines: 95% percentile limits',ha='center',fontsize=8.2,color='#4B5965');fig.tight_layout(rect=[0,.07,1,1]);save(fig,out,3)
    grid=np.linspace(0,1,401); curves_export=[]
    for typ,num in [('roc',4),('pr',5)]:
        fig,ax=plt.subplots(figsize=(5.9,4.45));curves=[]
        for name in MAIN:
            cs=[]
            for seed in SEEDS:
                score=piv[name,seed].to_numpy()
                if typ=='roc':xx,yy,_=roc_curve(y,score)
                else:yy,xx,_=precision_recall_curve(y,score);xx=xx[::-1];yy=yy[::-1]
                cs.append(np.interp(grid,xx,yy))
            mean=np.mean(cs,axis=0);std=np.std(cs,axis=0,ddof=1)
            curves.append((name,mean,std));curves_export.extend(dict(curve=typ,model=name,x=x,mean=m,sd=s) for x,m,s in zip(grid,mean,std))
        for name,mean,std in curves:ax.fill_between(grid,np.clip(mean-std,0,1),np.clip(mean+std,0,1),color=lighten(COLORS[name],.15),linewidth=0,zorder=1)
        for name,mean,std in curves:
            key='AUC' if typ=='roc' else 'AP';label=f'{name} ({key} {summary.loc[(name,"fixed"),key+"_mean"]:.4f})';ax.plot(grid,mean,color=COLORS[name],ls=LINESTYLES[name],lw=1.8,label=label,zorder=3)
        if typ=='roc':ax.plot([0,1],[0,1],'--',color='#77838A',lw=1,zorder=2);ax.set_xlabel('False positive rate');ax.set_ylabel('True positive rate');loc='lower right'
        else:ax.axhline(y.mean(),ls='--',color='#77838A',lw=1,zorder=2);ax.set_xlabel('Recall');ax.set_ylabel('Precision');loc='lower left'
        ax.set(xlim=(0,1),ylim=(0,1.025),xticks=np.arange(0,1.01,.2),yticks=np.arange(0,1.01,.2));ax.grid(color='#E5EAED',lw=.65);ax.set_axisbelow(True);ax.legend(loc=loc,framealpha=1,edgecolor='#D1D7DC',fancybox=False)
        fig.tight_layout();save(fig,out,num)
    pd.DataFrame(curves_export).to_csv(out/'figure_source_curves.csv',index=False)
    # Rectangles rather than imshow: the confusion matrices remain vector objects in PDF.
    fig,axs=plt.subplots(2,2,figsize=(6.45,5.75));counts=[]
    for ax,name in zip(axs.flat,MAIN):
        r=summary.loc[(name,'fixed')];mat=np.array([[r.TN_mean,r.FP_mean],[r.FN_mean,r.TP_mean]]);assert np.allclose(mat.sum(axis=1),[250,271]);pct=mat/np.array([250,271])[:,None]*100
        for i in range(2):
            for j in range(2):
                val=mat[i,j];co=plt.cm.Blues(.17+.78*val/271);ax.add_patch(Rectangle((j-.5,i-.5),1,1,fc=co,ec='white',lw=1.2));ax.text(j,i-.09,f'{val:.1f}',ha='center',va='center',color='white' if val>165 else '#162B3D',fontsize=12,weight='bold');ax.text(j,i+.20,f'{pct[i,j]:.1f}%',ha='center',va='center',color='white' if val>165 else '#162B3D',fontsize=9.5);counts.append(dict(model=name,true=['non-AMP','AMP'][i],predicted=['non-AMP','AMP'][j],mean_count=val,row_percent=pct[i,j]))
        ax.set(xlim=(-.5,1.5),ylim=(1.5,-.5),xticks=[0,1],yticks=[0,1],xticklabels=['non-AMP','AMP'],yticklabels=['non-AMP','AMP']);ax.set_aspect('equal');ax.set_title(name,pad=6);ax.set_xlabel('Predicted class',fontsize=9);ax.set_ylabel('True class',fontsize=9);ax.tick_params(length=0)
        for spine in ax.spines.values():spine.set_visible(False)
    fig.tight_layout(rect=[0,.035,1,1],h_pad=1.0,w_pad=.8);fig.text(.5,.01,'Mean counts and within-true-class percentages (threshold 0.5)',ha='center',fontsize=8.8,color='#4B5965');save(fig,out,6);pd.DataFrame(counts).to_csv(out/'figure_source_confusion.csv',index=False)
    a=pd.read_csv(data/'sevenD_attributions.csv');fc=[f'feature_{i}' for i in range(7)];assert len(a)==250 and a.groupby('seed').size().eq(50).all() and np.isfinite(a[fc]).all().all()
    for seed in SEEDS:
        test_probs=pd.read_csv(data/'sevenD_predictions.csv');test_probs=test_probs[test_probs.model.eq('Cross-attn') & test_probs.seed.eq(seed) & test_probs.split.eq('test') & test_probs.label.eq(1) & test_probs.prob.ge(.5)]
        top_ids=set(test_probs.nlargest(50,'prob').id);assert top_ids==set(a[a.seed.eq(seed)].id),seed
    order=a[fc].median().sort_values(ascending=False).index.tolist()[:5];names=[NAMES[int(k.split('_')[-1])] for k in order]
    saved=json.loads((data/'attribution_summary.json').read_text());assert names==saved['top_features'][:5]
    fig,ax=plt.subplots(figsize=(6.4,3.35));bp=ax.boxplot([a[k].to_numpy() for k in order],vert=False,patch_artist=True,tick_labels=names,showfliers=True,widths=.52,boxprops={'facecolor':'#D7E9F4','edgecolor':'#355B7C'},medianprops={'color':'#214E75','linewidth':1.45},whiskerprops={'color':'#355B7C'},capprops={'color':'#355B7C'},flierprops={'marker':'o','markersize':2.8,'markerfacecolor':'white','markeredgecolor':'#355B7C','markeredgewidth':.55});ax.invert_yaxis();ax.set_xlabel('Absolute gradient × input');ax.grid(axis='x',color='#E5EAED',lw=.65);ax.set_axisbelow(True);fig.tight_layout();save(fig,out,7)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    metrics,summary,piv,y,boot,draws=summarize(args.data,args.out)
    architecture(args.out);create_plots(args.data,args.out,metrics,summary,piv,y,boot,draws)
    hashes={p.name:sha(p) for p in sorted(args.out.glob('Figure_*')) if p.suffix in {'.svg', '.pdf'}}
    (args.out/'figure_validation.json').write_text(json.dumps({'fits':30,'test_predictions':15630,'validation_predictions':15690,'test_class_counts':[250,271],'seeds':SEEDS,'metric_rows_verified':len(metrics),'metrics_per_row':16,'bootstrap_draws_per_contrast':2000,'bootstrap_contrasts_verified':5,'prediction_sha256':sha(args.data/'sevenD_predictions.csv'),'figure_sha256':hashes},indent=2))
    print(summary.loc[(MAIN,'fixed'),['AUC_mean','AUC_std','AP_mean','Sensitivity_mean','Specificity_mean']]);print(boot);print('Generated 7 PDF/SVG vector figures; checked predictions, thresholds, bootstrap and attribution IDs.',flush=True)

if __name__=='__main__':main()
