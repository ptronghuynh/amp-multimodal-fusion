from pathlib import Path
import io, json, zipfile, hashlib, time, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix, accuracy_score, f1_score
from Bio import SeqIO
from torch.utils.data import DataLoader, TensorDataset

import argparse
ap=argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--output', type=Path, required=True)
ap.add_argument('--cache', type=Path)
ap.add_argument('--phys-only', action='store_true')
args=ap.parse_args()
ROOT=args.root; OUT=args.output; OUT.mkdir(parents=True, exist_ok=True)
for line in (ROOT/'b40cov_manifest_sha256.txt').read_text().splitlines():
 digest, name=line.split(maxsplit=1)
 assert hashlib.sha256((ROOT/name.strip()).read_bytes()).hexdigest()==digest, name
torch.set_num_threads(2); torch.set_num_interop_threads(1)
SEEDS=[42,52,62,72,82]
# Exact feature definitions extracted from source notebook cell 88.
from phys7 import physchem_features as feature7, PHYS_NAMES as names7
records=[]
for split in ['train','val','test']:
 for r in SeqIO.parse(ROOT/f'{split}_b40cov.fasta','fasta'):
  seq=str(r.seq); records.append((r.id,seq,split,int(r.id.startswith('AMP_'))))
sub=pd.DataFrame(records,columns=['id','sequence','split','label'])
assert len(sub)==sub.id.nunique()==sub.sequence.nunique()==5219
manifest=pd.read_csv(ROOT/'sequence_splits_blast40cov_FINAL.csv')
assert dict(zip(manifest.sequence_id,manifest.split))==dict(zip(sub.id,sub.split))
assert sub.groupby('split').size().to_dict()=={'train':4175,'val':523,'test':521}
assert sub.groupby('split').label.sum().to_dict()=={'train':2134,'val':274,'test':271}
assert all(set(seq)<=set('ACDEFGHIKLMNPQRSTVWY') for seq in sub.sequence)
sub[['id','split','label']].to_csv(OUT/'full_manifest.csv',index=False)
P7=torch.stack([feature7(seq) for seq in sub.sequence]); Y=torch.tensor(sub.label.values,dtype=torch.float32)
assert P7.shape==(5219,7) and torch.isfinite(P7).all()
idx={sp:torch.tensor(np.flatnonzero(sub.split.values==sp)) for sp in ['train','val','test']}
np.savez_compressed(OUT/'phys7_full.npz',sequence_id=sub.id.values.astype(str),phys7=P7.numpy(),label=Y.numpy(),feature_names=np.array(names7))
if args.phys_only:
 # Empty sequence argument: Phys-only ignores it, no synthetic ESM embeddings are used.
 S=torch.empty((5219,0))
else:
 assert args.cache is not None, 'Need complete, genuine ESM-2 cache.'
 cache=np.load(args.cache,allow_pickle=True).item()
 missing=[seq for seq in sub.sequence if seq not in cache]
 assert not missing, f'Cache missing {len(missing)} sequences; full comparison not started.'
 S=torch.tensor(np.stack([cache[seq] for seq in sub.sequence]),dtype=torch.float32)
 assert S.shape==(5219,320) and torch.isfinite(S).all()
 provenance={'cache':args.cache.name,'sha256':hashlib.sha256(args.cache.read_bytes()).hexdigest(),'mode':'regenerated_from_supplied_ESM2_checkpoint','extraction_time_measured':True}
 reference=Path(__file__).resolve().parent/'cache_subset_inputs.npz'
 if reference.exists():
  with np.load(reference,allow_pickle=False) as ref:
   ids=ref['sequence_id'].tolist(); lookup=dict(zip(sub.id,range(len(sub))))
   x=S.numpy()[[lookup[i] for i in ids]]; y=ref['sequence_embedding']
   cosine=(x*y).sum(1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1))
   provenance.update({'reference_overlap':len(ids),'reference_max_abs_difference':float(np.max(np.abs(x-y))),'reference_mean_abs_difference':float(np.mean(np.abs(x-y))),'reference_min_cosine':float(cosine.min())})
   (OUT/'embedding_provenance.json').write_text(json.dumps(provenance,indent=2))
   assert np.isfinite(cosine).all() and cosine.min()>0.999, 'Encoder mismatch with recovered R2 vectors; inspect before training.'
 np.savez_compressed(OUT/'full_embeddings.npz',sequence_id=sub.id.values.astype(str),sequence_embedding=S.numpy())
 (OUT/'embedding_provenance.json').write_text(json.dumps(provenance,indent=2))
print('Full partition counts',sub.groupby(['split','label']).size().to_dict(),flush=True)
class Seq(nn.Module):
 def __init__(self,h=128):
  super().__init__(); self.net=nn.Sequential(nn.Linear(320,h),nn.ReLU(),nn.Linear(h,1))
 def forward(self,s,p):return self.net(s).squeeze(-1)
class Phys(nn.Module):
 def __init__(self,d=64,h=128):
  super().__init__(); self.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,1))
 def forward(self,s,p):return self.net(p).squeeze(-1)
class Concat(nn.Module):
 def __init__(self,d=64,h=256):
  super().__init__(); self.fc=nn.Sequential(nn.Linear(320+d,h),nn.ReLU(),nn.Linear(h,1))
 def forward(self,s,p):return self.fc(torch.cat([s,p],dim=1)).squeeze(-1)
class Cross(nn.Module):
 def __init__(self,d=64,h=128,fc=128,heads=4):
  super().__init__();self.seq_proj=nn.Linear(320,h);self.phys_proj=nn.Linear(d,h);self.attn=nn.MultiheadAttention(h,heads,batch_first=True);self.fc=nn.Sequential(nn.Linear(2*h,fc),nn.ReLU(),nn.Linear(fc,1))
 def forward(self,s,p):
  s=self.seq_proj(s).unsqueeze(1);p=self.phys_proj(p).unsqueeze(1)
  a,_=self.attn(s,p,p,need_weights=True);b,_=self.attn(p,s,s,need_weights=True)
  return self.fc(torch.cat([a.squeeze(1),b.squeeze(1)],dim=-1)).squeeze(-1)
 def collapse(self):
  h=self.seq_proj.out_features; d=self.phys_proj.in_features; w=self.fc[0].out_features
  # A one-key attention block is just the V affine map followed by the output affine map.
  v=self.attn.in_proj_weight[2*h:];vb=self.attn.in_proj_bias[2*h:];o=self.attn.out_proj.weight;ob=self.attn.out_proj.bias
  A=o@v; b=o@vb+ob
  As=A@self.seq_proj.weight; bs=A@self.seq_proj.bias+b
  Ap=A@self.phys_proj.weight; bp=A@self.phys_proj.bias+b
  W=self.fc[0].weight; b0=self.fc[0].bias
  target=Concat(d,w)
  with torch.no_grad():
   target.fc[0].weight.copy_(torch.cat([W[:,h:]@As,W[:,:h]@Ap],dim=1));target.fc[0].bias.copy_(b0+W[:,:h]@bp+W[:,h:]@bs)
   target.fc[2].weight.copy_(self.fc[2].weight);target.fc[2].bias.copy_(self.fc[2].bias)
  return target

def count(m):return sum(x.numel() for x in m.parameters() if x.requires_grad)
class SeqLinear(nn.Module):
 def __init__(self):
  super().__init__();self.fc=nn.Linear(320,1)
 def forward(self,s,p):return self.fc(s).squeeze(-1)
P=P7
factories={'Seq-only':lambda:SeqLinear(),'Phys-only':lambda:Phys(7,64),'Concat':lambda:Concat(7,256),'Concat-matched':lambda:Concat(7,1456),'Concat-active-matched':lambda:Concat(7,1056),'Cross-attn':lambda:Cross(7,256,256,4)}

expected={'Seq-only':321,'Phys-only':577,'Concat':84225,'Concat-matched':479025,'Concat-active-matched':347425,'Cross-attn':478977}
for name,build in factories.items():
 model=build(); assert count(model)==expected[name],(name,count(model))
 assert model(torch.zeros((2,320)),torch.zeros((2,7))).shape==(2,)
if args.phys_only: factories={'Phys-only':factories['Phys-only']}
(OUT/'training_config.json').write_text(json.dumps({'seeds':SEEDS,'models':list(factories),'train':4175,'validation':523,'test':521,'optimizer':'Adam','learning_rate':0.001,'batch_size':64,'max_epochs':12,'early_stopping_patience':4,'checkpoint_selection':'minimum validation loss','loss':'BCEWithLogitsLoss','phys7_standardization':False,'device':'cpu','torch':torch.__version__,'source':'R2 Reproducibility/source/run_sevenD.py; full audited partition; regenerated frozen ESM-2 embeddings','phys_only':args.phys_only},indent=2))
def metrics(y,prob,threshold=.5):
 pred=prob>=threshold; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return {'AUC':roc_auc_score(y,prob),'AP':average_precision_score(y,prob),'ACC':accuracy_score(y,pred),'F1':f1_score(y,pred,zero_division=0),'MCC':matthews_corrcoef(y,pred),'Sensitivity':tp/(tp+fn),'Specificity':tn/(tn+fp),'TP':int(tp),'TN':int(tn),'FP':int(fp),'FN':int(fn)}

def predict(model,x):
 model.eval()
 with torch.no_grad():return torch.sigmoid(model(S[x],P[x])).numpy()

def threshold_mcc(y,prob):
 candidates=np.unique(np.r_[0.,prob,1.]); scores=np.array([matthews_corrcoef(y,prob>=t) for t in candidates]); best=np.flatnonzero(scores==scores.max()); return float(candidates[best[np.argmin(np.abs(candidates[best]-.5))]])
rows=[]; preds=[]; attr=[]; history=[]; starts=time.time()
for seed in SEEDS:
 for name,build in factories.items():
  run_start=time.perf_counter()
  random.seed(seed); np.random.seed(seed);torch.manual_seed(seed)
  m=build(); opt=torch.optim.Adam(m.parameters(),lr=1e-3); loss=nn.BCEWithLogitsLoss()
  gen=torch.Generator().manual_seed(seed); loader=DataLoader(TensorDataset(S[idx['train']],P[idx['train']],Y[idx['train']]),batch_size=64,shuffle=True,generator=gen)
  best=float('inf'); best_state=None; wait=0; best_epoch=0
  for epoch in range(1,13):
   m.train()
   for ss,pp,yy in loader:
    opt.zero_grad(); z=m(ss,pp); l=loss(z,yy);l.backward();opt.step()
   m.eval()
   with torch.no_grad():vl=loss(m(S[idx['val']],P[idx['val']]),Y[idx['val']]).item()
   history.append({'model':name,'seed':seed,'epoch':epoch,'val_loss':vl})
   if vl<best:best=vl;best_state={k:v.detach().clone() for k,v in m.state_dict().items()};best_epoch=epoch;wait=0
   else:wait+=1
   if wait>=4:break
  m.load_state_dict(best_state); pv=predict(m,idx['val']); pt=predict(m,idx['test']); yv=Y[idx['val']].numpy(); yt=Y[idx['test']].numpy(); t=threshold_mcc(yv,pv)
  for mode,thr in [('fixed',.5),('validation_MCC',t)]:
   rows.append({'model':name,'seed':seed,'mode':mode,'params':count(m),'epoch':best_epoch,'threshold':thr,'seconds':time.perf_counter()-run_start,**metrics(yt,pt,thr)})
  for sp,probs in [('val',pv),('test',pt)]:
   for k,j in enumerate(idx[sp].tolist()):preds.append({'model':name,'seed':seed,'split':sp,'id':sub.id.iloc[j],'label':int(Y[j]),'prob':float(probs[k])})
  if name=='Cross-attn':
   px=P[idx['test']].detach().clone().requires_grad_(True); z=m(S[idx['test']],px); grad=torch.autograd.grad(z.sum(),px)[0]; aa=(grad*px).detach().abs().numpy(); tp=np.flatnonzero((yt==1)&(pt>=.5)); top=tp[np.argsort(-pt[tp])[:50]]
   for j in top:
    attr.append({'seed':seed,'id':sub.id.iloc[idx['test'][j].item()],**{f'feature_{i}':float(aa[j,i]) for i in range(7)}})
  print(seed,name,count(m),f'AUC={rows[-2]["AUC"]:.5f}',f'AP={rows[-2]["AP"]:.5f}',f'MCC={rows[-2]["MCC"]:.5f}',flush=True)
  torch.save(best_state,OUT/f'7D_{name.replace(" ","_")}_seed{seed}.pt')
  pd.DataFrame(rows).to_csv(OUT/'sevenD_metrics_all_seeds.csv',index=False)
  pd.DataFrame(preds).to_csv(OUT/'sevenD_predictions.csv',index=False)
  pd.DataFrame(history).to_csv(OUT/'training_history.csv',index=False)
R=pd.DataFrame(rows);R.to_csv(OUT/'sevenD_metrics_all_seeds.csv',index=False);F=pd.DataFrame(preds);F.to_csv(OUT/'sevenD_predictions.csv',index=False);pd.DataFrame(attr).to_csv(OUT/'sevenD_attributions.csv',index=False)
metrics_cols=['AUC','AP','ACC','F1','MCC','Sensitivity','Specificity','threshold','seconds']
agg=R.groupby(['model','mode'])[metrics_cols].agg(['mean','std']);agg.columns=['_'.join(x) for x in agg.columns];agg.to_csv(OUT/'sevenD_summary.csv')
if not args.phys_only:
 # Paired, class-stratified bootstrap of per-seed AUC differences (average over the same five seeds).
 test=F[F.split=='test']; piv=test.pivot(index='id',columns=['model','seed'],values='prob'); labs=test.drop_duplicates('id').set_index('id').loc[piv.index].label.to_numpy();pos=np.flatnonzero(labs==1);neg=np.flatnonzero(labs==0)
 rng=np.random.default_rng(20261125);pairs=[('Concat','Seq-only'),('Cross-attn','Seq-only'),('Cross-attn','Concat'),('Cross-attn','Concat-matched'),('Cross-attn','Concat-active-matched')];boot=[]
 # AUC for bootstrap samples via pairwise score comparison, avoids repeated sorting.
 auc_pairs={k:(piv[k].to_numpy()[pos,None]>piv[k].to_numpy()[None,neg]).astype(float)+.5*(piv[k].to_numpy()[pos,None]==piv[k].to_numpy()[None,neg]) for k in piv.columns}
 for a,b in pairs:
  d=np.mean([auc_pairs[(a,s)]-auc_pairs[(b,s)] for s in SEEDS],axis=0); ds=[]
  for j in range(2000):
   ii=rng.integers(0,len(pos),size=len(pos));jj=rng.integers(0,len(neg),size=len(neg));ds.append(d[np.ix_(ii,jj)].mean())
  boot.append({'comparison':a+' - '+b,'delta':float(d.mean()),'ci_low':float(np.percentile(ds,2.5)),'ci_high':float(np.percentile(ds,97.5)),'bootstrap_sd':float(np.std(ds,ddof=1)),'bootstrap_support_positive':float(np.mean(np.array(ds)>0)), 'bootstrap_fraction_nonpositive':float(np.mean(np.array(ds)<=0))})
 pd.DataFrame(boot).to_csv(OUT/'sevenD_bootstrap.csv',index=False)
 print('Elapsed seconds',time.time()-starts);print(agg.round(4).to_string());print(pd.DataFrame(boot).to_string(index=False))

print(agg.round(6).to_string(),flush=True)
(OUT/'status.json').write_text(json.dumps({'completed_fits':len(R)//2,'full_comparison_completed':not args.phys_only,'elapsed_seconds':time.time()-starts},indent=2))
