from pathlib import Path
import io, json, zipfile, hashlib, time, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix, accuracy_score, f1_score
from Bio import SeqIO
from torch.utils.data import DataLoader, TensorDataset

BASE=Path(__file__).resolve().parent.parent; ROOT=BASE/'inputs'; OUT=BASE/'outputs'; OUT.mkdir(exist_ok=True)
torch.set_num_threads(2); torch.set_num_interop_threads(1)
SEEDS=[42,52,62,72,82]
nb=json.loads((ROOT/'Xulydataset.ipynb').read_text(encoding='utf-8'))
# Source cell 88 is execution 136: seven features active before the archived final training.
ns={'np':np,'torch':torch,'nn':nn};exec(''.join(nb['cells'][88]['source']),ns)
feature7=ns['physchem_features']; names7=ns['PHYS_NAMES']
# Cell 164 is the 64D implementation used for AMPDiscover; function computes 20+16+28.
ns64={'np':np,'torch':torch,'nn':nn};exec(''.join(nb['cells'][164]['source']),ns64)
feature64=ns64['physchem_features_64']
with zipfile.ZipFile(ROOT/'checkpoints_amplify.zip') as z:
 cache=np.load(io.BytesIO(z.read('checkpoints_amplify/cache/esm2_t6_8M_amplify_cache.npy')),allow_pickle=True).item()
records=[]
for split in ['train','val','test']:
 for r in SeqIO.parse(ROOT/f'{split}_b40cov.fasta','fasta'):
  seq=str(r.seq); records.append((r.id,seq,split,int(r.id.startswith('AMP_')),seq in cache))
df=pd.DataFrame(records,columns=['id','sequence','split','label','cached'])
assert len(df)==5219 and df.sequence.nunique()==5219
# Use intersection fixed by cache availability, not outcome or predictions.
sub=df[df.cached].copy().reset_index(drop=True)
print('Subset counts',sub.groupby(['split','label']).size().to_dict(),flush=True)
sub[['id','split','label']].to_csv(OUT/'cache_subset_manifest.csv',index=False)
S=torch.tensor(np.stack([cache[s] for s in sub.sequence]),dtype=torch.float32)
P7=torch.stack([feature7(s) for s in sub.sequence]); P64=torch.stack([feature64(s) for s in sub.sequence]); Y=torch.tensor(sub.label.values,dtype=torch.float32)
idx={sp:torch.tensor(np.flatnonzero(sub.split.values==sp)) for sp in ['train','val','test']}
assert torch.isfinite(S).all() and torch.isfinite(P64).all()
# Train-only standardization of 64D features for new controlled experiment.
mu=P64[idx['train']].mean(0); sd=P64[idx['train']].std(0,unbiased=False); sd[sd<1e-8]=1; P=(P64-mu)/sd
np.savez_compressed(OUT/'cache_subset_inputs.npz',sequence_embedding=S.numpy(),phys64=P64.numpy(),phys7=P7.numpy(),label=Y.numpy(),split=sub.split.values.astype(str),sequence_id=sub.id.values.astype(str),mean=mu.numpy(),std=sd.numpy())
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
# Direct verification with archived checkpoint. No predictive performance is inferred from this forward test.
archive=Cross(7,256,256,4)
with zipfile.ZipFile(ROOT/'checkpoints.zip') as z:
 states={Path(f).stem:torch.load(io.BytesIO(z.read(f)),map_location='cpu',weights_only=True) for f in z.namelist() if f.endswith('.pt')}
archive.load_state_dict(states['cross_attn'],strict=True);archive.eval()
with torch.no_grad():
 collapsed=archive.collapse().eval(); a=archive(S,P7);b=collapsed(S,P7)
 delta=(a-b).abs().max().item(); assert delta<1e-4,delta
# Query/key gradients and one-token weights directly measured.
archive.zero_grad();archive(S[:64],P7[:64]).sum().backward();h=256
qkgrad=archive.attn.in_proj_weight.grad[:2*h].abs().max().item();qkbias=archive.attn.in_proj_bias.grad[:2*h].abs().max().item()
with torch.no_grad():
 s=archive.seq_proj(S[:64]).unsqueeze(1);p=archive.phys_proj(P7[:64]).unsqueeze(1);_,aw=archive.attn(s,p,p,need_weights=True,average_attn_weights=False)
 awerr=(aw-1).abs().max().item()
verify={'checkpoint_parameter_counts':{k:sum(t.numel() for t in v.values()) for k,v in states.items()},'checkpoint_tensor_shapes':{k:{n:list(t.shape) for n,t in v.items()} for k,v in states.items()},'cross_collapsed_params':count(collapsed),'cross_max_abs_logit_difference':delta,'qk_max_abs_weight_gradient':qkgrad,'qk_max_abs_bias_gradient':qkbias,'max_attention_weight_deviation_from_one':awerr,'input_vectors_checked':len(S),'torch_version':torch.__version__,'cached_embeddings':len(cache),'covered_sequences':len(sub),'seeds':SEEDS}
(OUT/'checkpoint_verification.json').write_text(json.dumps(verify,indent=2)); print('Verification',json.dumps({k:v for k,v in verify.items() if k!='checkpoint_tensor_shapes'}),flush=True)
# Full test/val/train subset, same feature vectors and same training settings for all models.
factories={'Seq-only':lambda:Seq(),'Phys-only':lambda:Phys(),'Concat':lambda:Concat(),'Concat-matched':lambda:Concat(h=385),'Concat-active-matched':lambda:Concat(h=299),'Cross-attn':lambda:Cross()}

def metrics(y,prob,threshold=.5):
 pred=prob>=threshold; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return {'AUC':roc_auc_score(y,prob),'AP':average_precision_score(y,prob),'ACC':accuracy_score(y,pred),'F1':f1_score(y,pred,zero_division=0),'MCC':matthews_corrcoef(y,pred),'Sensitivity':tp/(tp+fn),'Specificity':tn/(tn+fp),'TP':int(tp),'TN':int(tn),'FP':int(fp),'FN':int(fn)}

def predict(model,x):
 model.eval()
 with torch.no_grad():return torch.sigmoid(model(S[x],P[x])).numpy()

def threshold_mcc(y,prob):
 candidates=np.unique(np.r_[0.,prob,1.]); scores=np.array([matthews_corrcoef(y,prob>=t) for t in candidates]); best=np.flatnonzero(scores==scores.max()); return float(candidates[best[np.argmin(np.abs(candidates[best]-.5))]])
rows=[]; preds=[]; attr=[]; starts=time.time()
for seed in SEEDS:
 for name,build in factories.items():
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
   if vl<best:best=vl;best_state={k:v.detach().clone() for k,v in m.state_dict().items()};best_epoch=epoch;wait=0
   else:wait+=1
   if wait>=4:break
  m.load_state_dict(best_state); pv=predict(m,idx['val']); pt=predict(m,idx['test']); yv=Y[idx['val']].numpy(); yt=Y[idx['test']].numpy(); t=threshold_mcc(yv,pv)
  for mode,thr in [('fixed',.5),('validation_MCC',t)]:
   rows.append({'model':name,'seed':seed,'mode':mode,'params':count(m),'epoch':best_epoch,'threshold':thr,**metrics(yt,pt,thr)})
  for sp,probs in [('val',pv),('test',pt)]:
   for k,j in enumerate(idx[sp].tolist()):preds.append({'model':name,'seed':seed,'split':sp,'id':sub.id.iloc[j],'label':int(Y[j]),'prob':float(probs[k])})
  if name=='Cross-attn':
   px=P[idx['test']].detach().clone().requires_grad_(True); z=m(S[idx['test']],px); grad=torch.autograd.grad(z.sum(),px)[0]; aa=(grad*px).detach().abs().numpy(); tp=np.flatnonzero((yt==1)&(pt>=.5)); top=tp[np.argsort(-pt[tp])[:50]]
   for j in top:
    attr.append({'seed':seed,'id':sub.id.iloc[idx['test'][j].item()],**{f'feature_{i}':float(aa[j,i]) for i in range(64)}})
  print(seed,name,count(m),f'AUC={rows[-2]["AUC"]:.5f}',f'AP={rows[-2]["AP"]:.5f}',f'MCC={rows[-2]["MCC"]:.5f}',flush=True)
  torch.save(best_state,OUT/f'{name.replace(" ","_")}_seed{seed}.pt')
R=pd.DataFrame(rows);R.to_csv(OUT/'supplementary_metrics_all_seeds.csv',index=False);F=pd.DataFrame(preds);F.to_csv(OUT/'supplementary_predictions.csv',index=False);pd.DataFrame(attr).to_csv(OUT/'supplementary_attributions.csv',index=False)
metrics_cols=['AUC','AP','ACC','F1','MCC','Sensitivity','Specificity','threshold']
agg=R.groupby(['model','mode'])[metrics_cols].agg(['mean','std']);agg.columns=['_'.join(x) for x in agg.columns];agg.to_csv(OUT/'supplementary_summary.csv')
# Paired, class-stratified bootstrap of per-seed AUC differences (average over the same five seeds).
test=F[F.split=='test']; piv=test.pivot(index='id',columns=['model','seed'],values='prob'); labs=test.drop_duplicates('id').set_index('id').loc[piv.index].label.to_numpy();pos=np.flatnonzero(labs==1);neg=np.flatnonzero(labs==0)
rng=np.random.default_rng(20261125);pairs=[('Concat','Seq-only'),('Cross-attn','Concat'),('Cross-attn','Concat-matched'),('Cross-attn','Concat-active-matched')];boot=[]
# AUC for bootstrap samples via pairwise score comparison, avoids repeated sorting.
auc_pairs={k:(piv[k].to_numpy()[pos,None]>piv[k].to_numpy()[None,neg]).astype(float)+.5*(piv[k].to_numpy()[pos,None]==piv[k].to_numpy()[None,neg]) for k in piv.columns}
for a,b in pairs:
 d=np.mean([auc_pairs[(a,s)]-auc_pairs[(b,s)] for s in SEEDS],axis=0); ds=[]
 for j in range(2000):
  ii=rng.integers(0,len(pos),size=len(pos));jj=rng.integers(0,len(neg),size=len(neg));ds.append(d[np.ix_(ii,jj)].mean())
 boot.append({'comparison':a+' - '+b,'delta':float(d.mean()),'ci_low':float(np.percentile(ds,2.5)),'ci_high':float(np.percentile(ds,97.5)),'bootstrap_support_positive':float(np.mean(np.array(ds)>0))})
pd.DataFrame(boot).to_csv(OUT/'supplementary_bootstrap.csv',index=False)
print('Elapsed seconds',time.time()-starts);print(agg.round(4).to_string());print(pd.DataFrame(boot).to_string(index=False))
