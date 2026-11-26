"""Validate the complete full-cohort experiment and produce publication tables and plots."""
from pathlib import Path
import argparse, json, hashlib, io, zipfile, platform
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix, accuracy_score, f1_score, matthews_corrcoef, precision_score)
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
O=a.output; root=a.root
seeds=[42,52,62,72,82]; models=['Seq-only','Phys-only','Concat','Cross-attn','Concat-matched','Concat-active-matched']; main=models[:4]
M=pd.read_csv(O/'sevenD_metrics_all_seeds.csv'); P=pd.read_csv(O/'sevenD_predictions.csv'); T=P[P.split=='test']; V=P[P.split=='val']
assert M.shape[0]==60 and len(T)==30*521 and len(V)==30*523
assert not T.duplicated(['model','seed','id']).any() and not V.duplicated(['model','seed','id']).any()
manifest=pd.read_csv(O/'full_manifest.csv'); ids={sp:manifest[manifest.split==sp].id.tolist() for sp in ('train','val','test')}
assert all(set(g.id)==set(ids[sp]) for (name,seed,sp),g in P.groupby(['model','seed','split']))
lab=dict(zip(manifest.id,manifest.label)); assert all(P.label==P.id.map(lab))
S=np.load(O/'full_embeddings.npz',allow_pickle=False); F=np.load(O/'phys7_full.npz',allow_pickle=False); assert np.array_equal(S['sequence_id'],F['sequence_id'])
s=torch.from_numpy(S['sequence_embedding']); p=torch.from_numpy(F['phys7']); Y=F['label']; lookup={str(k):i for i,k in enumerate(S['sequence_id'])}
torch.set_num_threads(2)
class Seq(nn.Module):
 def __init__(self):super().__init__();self.fc=nn.Linear(320,1)
 def forward(self,s,p):return self.fc(s).squeeze(-1)
class Phys(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(7,64),nn.ReLU(),nn.Linear(64,1))
 def forward(self,s,p):return self.net(p).squeeze(-1)
class Concat(nn.Module):
 def __init__(self,h=256):super().__init__();self.fc=nn.Sequential(nn.Linear(327,h),nn.ReLU(),nn.Linear(h,1))
 def forward(self,s,p):return self.fc(torch.cat([s,p],1)).squeeze(-1)
class Cross(nn.Module):
 def __init__(self):
  super().__init__();self.seq_proj=nn.Linear(320,256);self.phys_proj=nn.Linear(7,256);self.attn=nn.MultiheadAttention(256,4,batch_first=True);self.fc=nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Linear(256,1))
 def forward(self,s,p):
  u=self.seq_proj(s).unsqueeze(1);v=self.phys_proj(p).unsqueeze(1);x,_=self.attn(u,v,v,need_weights=True);y,_=self.attn(v,u,u,need_weights=True);return self.fc(torch.cat([x.squeeze(1),y.squeeze(1)],1)).squeeze(-1)
 def collapse(self):
  h=256;v=self.attn.in_proj_weight[2*h:];vb=self.attn.in_proj_bias[2*h:];o=self.attn.out_proj.weight;ob=self.attn.out_proj.bias
  A=o@v;b=o@vb+ob;As=A@self.seq_proj.weight;bs=A@self.seq_proj.bias+b;Ap=A@self.phys_proj.weight;bp=A@self.phys_proj.bias+b;W=self.fc[0].weight;b0=self.fc[0].bias
  target=Concat()
  with torch.no_grad():
   target.fc[0].weight.copy_(torch.cat([W[:,h:]@As,W[:,:h]@Ap],1));target.fc[0].bias.copy_(b0+W[:,:h]@bp+W[:,h:]@bs);target.fc[2].weight.copy_(self.fc[2].weight);target.fc[2].bias.copy_(self.fc[2].bias)
  return target
factory={'Seq-only':Seq,'Phys-only':Phys,'Concat':Concat,'Cross-attn':Cross,'Concat-matched':lambda:Concat(1456),'Concat-active-matched':lambda:Concat(1056)}
checks=[]; extras=[]
for name in models:
 for seed in seeds:
  model=factory[name]();state=torch.load(O/f'7D_{name}_seed{seed}.pt',map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True);model.eval()
  n=sum(v.numel() for v in model.parameters()); assert n==int(M[(M.model==name)&(M.seed==seed)].params.iloc[0])
  for sp in ('val','test'):
   data=P[(P.model==name)&(P.seed==seed)&(P.split==sp)]; ix=[lookup[k] for k in data.id]; y=data.label.to_numpy(); prob=data.prob.to_numpy()
   with torch.no_grad(): new=torch.sigmoid(model(s[ix],p[ix])).numpy()
   err=float(np.max(np.abs(new-prob))); assert err<1e-5
   checks.append({'model':name,'seed':seed,'split':sp,'n':len(y),'max_prediction_error':err,'params':n})
   if sp=='test':
    for mode in ('fixed','validation_MCC'):
     row=M[(M.model==name)&(M.seed==seed)&(M['mode']==mode)].iloc[0];th=row.threshold;pred=prob>=th;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
     for k,v in [('AUC',roc_auc_score(y,prob)),('AP',average_precision_score(y,prob)),('ACC',accuracy_score(y,pred)),('F1',f1_score(y,pred,zero_division=0)),('MCC',matthews_corrcoef(y,pred))]:assert abs(row[k]-v)<1e-10,(name,seed,k,row[k],v)
     extras.append(dict(row.to_dict(),Precision=precision_score(y,pred,zero_division=0),Balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,FPR=fp/(tn+fp),FNR=fn/(tp+fn)))
Q=pd.DataFrame(extras);Q.to_csv(O/'verified_metrics.csv',index=False); pd.DataFrame(checks).to_csv(O/'checkpoint_prediction_verification.csv',index=False)
cols=['AUC','AP','ACC','F1','MCC','Sensitivity','Specificity','Precision','Balanced_accuracy','FPR','FNR','TP','TN','FP','FN','threshold','seconds']
summary=Q.groupby(['model','mode'])[cols].agg(['mean','std']);summary.columns=['_'.join(x) for x in summary.columns];summary.to_csv(O/'verified_summary.csv')
# Functional collapse: check both the recovered archive and newly fitted checkpoints on all inputs.
with zipfile.ZipFile(root/'checkpoints.zip') as z:
 key=next(k for k in z.namelist() if k.endswith('/cross_attn.pt') or k=='cross_attn.pt');old=torch.load(io.BytesIO(z.read(key)),map_location='cpu',weights_only=True)
red=[]
for name,state in [('archived',old)]+[(str(seed),torch.load(O/f'7D_Cross-attn_seed{seed}.pt',map_location='cpu',weights_only=True)) for seed in seeds]:
 cross=Cross();cross.load_state_dict(state,strict=True);cross.eval();small=cross.collapse().eval();diff=[]
 with torch.no_grad():
  for j in range(0,len(s),256):diff.append((cross(s[j:j+256],p[j:j+256])-small(s[j:j+256],p[j:j+256])).abs().max().item())
  u=cross.seq_proj(s[:64]).unsqueeze(1);v=cross.phys_proj(p[:64]).unsqueeze(1);_,w=cross.attn(u,v,v,need_weights=True,average_attn_weights=False);werr=(w-1).abs().max().item()
 cross.zero_grad();cross(s[:64],p[:64]).sum().backward();grad=cross.attn.in_proj_weight.grad[:512].abs().max().item();gb=cross.attn.in_proj_bias.grad[:512].abs().max().item()
 assert max(diff)<1e-4 and werr==grad==gb==0
 red.append({'checkpoint':name,'n_inputs':len(s),'max_abs_logit_difference':max(diff),'attention_weight_error':werr,'QK_weight_gradient_max':grad,'QK_bias_gradient_max':gb,'original_parameters':478977,'collapsed_parameters':84225})
pd.DataFrame(red).to_csv(O/'functional_reduction_full.csv',index=False)
# Recompute paired bootstrap, keeping test examples aligned across all five seeds.
piv=T.pivot(index='id',columns=['model','seed'],values='prob'); y=T.drop_duplicates('id').set_index('id').loc[piv.index].label.to_numpy();pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
assert len(pos)==271 and len(neg)==250
rng=np.random.default_rng(20261125);pairs=[('Concat','Seq-only'),('Cross-attn','Seq-only'),('Cross-attn','Concat'),('Cross-attn','Concat-matched'),('Cross-attn','Concat-active-matched')]
score={k:(piv[k].to_numpy()[pos,None]>piv[k].to_numpy()[None,neg]).astype(np.float32)+.5*(piv[k].to_numpy()[pos,None]==piv[k].to_numpy()[None,neg]) for k in piv.columns}
boot=[]; dist={}
for aa,bb in pairs:
 d=np.mean([score[aa,sd]-score[bb,sd] for sd in seeds],axis=0,dtype=np.float64); vals=np.array([d[np.ix_(rng.integers(0,len(pos),size=len(pos)),rng.integers(0,len(neg),size=len(neg)))].mean() for _ in range(2000)])
 k=aa+' - '+bb;dist[k]=vals;boot.append({'comparison':k,'delta':float(d.mean()),'bootstrap_sd':float(vals.std(ddof=1)),'ci_low':float(np.percentile(vals,2.5)),'ci_high':float(np.percentile(vals,97.5)),'bootstrap_fraction_nonpositive':float(np.mean(vals<=0))})
B=pd.DataFrame(boot);B.to_csv(O/'verified_bootstrap.csv',index=False);pd.DataFrame(dist).to_csv(O/'bootstrap_resamples.csv',index=False)
oldboot=pd.read_csv(O/'sevenD_bootstrap.csv');assert np.allclose(B[['delta','ci_low','ci_high']],oldboot[['delta','ci_low','ci_high']],atol=1e-10)
# Scientific plots. Curves show arithmetic averages across seed-wise interpolated curves, not pooled predictions.
plt.rcParams.update({'font.size':10,'axes.labelsize':10,'axes.titlesize':11,'legend.fontsize':9,'figure.dpi':130,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False})
colors=dict(zip(main,['#4C78A8','#A5A5A5','#F28E2B','#59A14F']))
def save(fig,n):fig.savefig(O/n,dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)
fig,ax=plt.subplots(figsize=(6.8,3.8)); q=summary.loc[(main,'fixed'),:] if False else summary
vals=[summary.loc[(name,'fixed'),'AUC_mean'] for name in main];sd=[summary.loc[(name,'fixed'),'AUC_std'] for name in main]
ax.bar(main,vals,yerr=sd,capsize=4,color=[colors[m] for m in main],edgecolor='black',linewidth=.5);ax.set_ylim(.5,1.0);ax.set_ylabel('ROC-AUC');ax.grid(axis='y',alpha=.2)
for i,(v,z) in enumerate(zip(vals,sd)):ax.text(i,v+z+.013,f'{v:.4f}',ha='center',fontsize=9)
fig.tight_layout();save(fig,'Figure_02_full.png')
fig,axs=plt.subplots(1,2,figsize=(7.5,3.1));
for ax,k in zip(axs,['Cross-attn - Seq-only','Cross-attn - Concat']):
 ax.hist(dist[k],bins=32,color='#4C78A8',alpha=.85,edgecolor='white');ax.axvline(0,ls='--',color='black',lw=1);ax.set_title(k);ax.set_xlabel('Paired ROC-AUC difference');ax.set_ylabel('Frequency')
fig.tight_layout();save(fig,'Figure_03_full.png')
for typ,filename in [('roc','Figure_04_full.png'),('pr','Figure_05_full.png')]:
 fig,ax=plt.subplots(figsize=(5.7,4.4));grid=np.linspace(0,1,401)
 for name in main:
  curves=[]
  for seed in seeds:
   v=piv[name,seed].to_numpy()
   if typ=='roc':xx,yy,_=roc_curve(y,v)
   else:
    yy,xx,_=precision_recall_curve(y,v);xx=xx[::-1];yy=yy[::-1]
   curves.append(np.interp(grid,xx,yy))
  mean=np.mean(curves,axis=0);std=np.std(curves,axis=0,ddof=1);key='AUC' if typ=='roc' else 'AP';label=f'{name} ({key} {summary.loc[(name,"fixed"),key+"_mean"]:.4f})';ax.plot(grid,mean,color=colors[name],lw=1.8,label=label);ax.fill_between(grid,np.clip(mean-std,0,1),np.clip(mean+std,0,1),color=colors[name],alpha=.09)
 if typ=='roc':ax.plot([0,1],[0,1],'--',color='gray',lw=1);ax.set_xlabel('False positive rate');ax.set_ylabel('True positive rate');ax.legend(loc='lower right')
 else:ax.axhline(y.mean(),ls='--',color='gray',lw=1);ax.set_xlabel('Recall');ax.set_ylabel('Precision');ax.legend(loc='lower left')
 ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.grid(alpha=.15);fig.tight_layout();save(fig,filename)
fig,axs=plt.subplots(2,2,figsize=(6.7,5.9));
for ax,name in zip(axs.flat,main):
 r=summary.loc[name,'fixed'];matrix=np.array([[r.TN_mean,r.FP_mean],[r.FN_mean,r.TP_mean]]);im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=271)
 for i in range(2):
  for j in range(2):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='white' if matrix[i,j]>135 else '#172b4d',fontsize=12)
 ax.set_title(name);ax.set_xticks([0,1],['non-AMP','AMP']);ax.set_yticks([0,1],['non-AMP','AMP']);ax.set_xlabel('Predicted');ax.set_ylabel('True')
fig.tight_layout();save(fig,'Figure_06_full.png')
A=pd.read_csv(O/'sevenD_attributions.csv');fcols=[f'feature_{i}' for i in range(7)];names=['KD hydropathy','Basic fraction','Acidic fraction','Aromatic fraction','Hydrophobic fraction','Aliphatic fraction','Sulfur fraction'];order=A[fcols].median().sort_values(ascending=False).index.tolist();top=order[:5];fig,ax=plt.subplots(figsize=(7,3.9));ax.boxplot([A[c] for c in top],tick_labels=[names[int(c.split('_')[-1])] for c in top],showfliers=True,patch_artist=True,boxprops={'facecolor':'#D8E9F5'},medianprops={'color':'#1E4976'});ax.tick_params(axis='x',rotation=20);ax.set_ylabel('Absolute gradient × input');fig.tight_layout();save(fig,'Figure_07_full.png')
profiles=A.groupby('seed')[fcols].median();pairs_corr=[(pearsonr(profiles.iloc[i],profiles.iloc[j]).statistic,spearmanr(profiles.iloc[i],profiles.iloc[j]).statistic) for i in range(len(profiles)) for j in range(i+1,len(profiles))];corr=np.array(pairs_corr);attr_summary={'n':len(A),'seed_counts':A.groupby('seed').size().to_dict(),'top_features':[names[int(c.split('_')[-1])] for c in order],'pearson_mean':float(corr[:,0].mean()),'pearson_sd':float(corr[:,0].std(ddof=1)),'spearman_mean':float(corr[:,1].mean()),'spearman_sd':float(corr[:,1].std(ddof=1))}
(O/'attribution_summary.json').write_text(json.dumps(attr_summary,indent=2));(O/'verification_summary.json').write_text(json.dumps({'fits':30,'test_predictions':len(T),'val_predictions':len(V),'max_prediction_error':max(x['max_prediction_error'] for x in checks),'embedding_shape':list(s.shape),'python':platform.python_version(),'torch':str(torch.__version__),'machine':platform.machine(),'processor':platform.processor(),'test_positive':271,'test_negative':250,'new_checkpoints_collapsed':5},indent=2));print(summary.loc[(slice(None),'fixed'),['AUC_mean','AUC_std','AP_mean','MCC_mean','Sensitivity_mean','Specificity_mean']].round(6));print(B.to_string(index=False));print(attr_summary);print('Verification complete',flush=True)
