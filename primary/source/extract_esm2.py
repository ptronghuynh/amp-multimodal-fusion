"""ESM-2 inference using Meta checkpoint, numerically verified against archived embeddings."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from Bio import SeqIO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
    cfg = checkpoint['cfg']['model']
    assert (cfg.encoder_layers, cfg.encoder_embed_dim, cfg.encoder_attention_heads) == (6, 320, 20)
    assert cfg.token_dropout and cfg.use_rotary_embeddings
    prefix = 'encoder.sentence_encoder.'
    weights = {k[len(prefix):]: v.float() for k, v in checkpoint['model'].items() if k.startswith(prefix)}
    vocab = ['<cls>', '<pad>', '<eos>', '<unk>'] + list('LAGVSERTIDPKQNFYM HWCXB UZO.-'.replace(' ', '')) + ['<null_1>', '<mask>']
    assert len(vocab) == 33 and len(set(vocab)) == 33
    token_ids = {s: i for i, s in enumerate(vocab)}
    def linear(x, name):
        return F.linear(x, weights[name+'.weight'], weights[name+'.bias'])
    def norm(x, name):
        return F.layer_norm(x, (320,), weights[name+'.weight'], weights[name+'.bias'], 1e-5)
    def rotate(x, c, s):
        u, v = x.chunk(2, dim=-1)
        return x*c + torch.cat((-v, u), dim=-1)*s
    @torch.inference_mode()
    def encode(seqs):
        b, t = len(seqs), max(map(len, seqs))+2
        tok = torch.full((b,t), 1, dtype=torch.long)
        for i, seq in enumerate(seqs):
            tok[i, :len(seq)+2] = torch.tensor([0]+[token_ids[c] for c in seq]+[2])
        valid = tok != 1
        x = F.embedding(tok, weights['embed_tokens.weight'])*0.88
        x = x.masked_fill(~valid[...,None], 0.)
        for i in range(6):
            p = f'layers.{i}.'
            a = p+'self_attn.'
            z = norm(x, p+'self_attn_layer_norm')
            q,k,v = [linear(z,a+n+'_proj').reshape(b,t,20,16).transpose(1,2) for n in ('q','k','v')]
            angle = torch.outer(torch.arange(t,dtype=torch.float32),weights[a+'rot_emb.inv_freq'])
            angle = torch.cat((angle,angle),dim=-1)[None,None,:,:]
            q,k = rotate(q,angle.cos(),angle.sin()),rotate(k,angle.cos(),angle.sin())
            out = F.scaled_dot_product_attention(q,k,v,attn_mask=valid[:,None,None,:],dropout_p=0.)
            x = x + linear(out.transpose(1,2).reshape(b,t,320),a+'out_proj')
            x = x + linear(F.gelu(linear(norm(x,p+'final_layer_norm'),p+'fc1')),p+'fc2')
        x = norm(x,'emb_layer_norm_after')
        return np.stack([x[i,1:len(s)+1].mean(0).numpy() for i,s in enumerate(seqs)])
    ref = np.load(args.reference, allow_pickle=False)
    sample = ref['sequence'][:32].tolist()
    pred = encode(sample)
    print('Pilot maximum absolute difference:', float(np.max(np.abs(pred-ref['embedding'][:32]))), flush=True)
    assert np.max(np.abs(pred-ref['embedding'][:32])) < 1e-4, 'ESM2 reproduction mismatch'
    records = [(r.id,str(r.seq),sp,int(r.id.startswith('AMP_'))) for sp in ('train','val','test') for r in SeqIO.parse(args.root/f'{sp}_b40cov.fasta','fasta')]
    assert len(records) == len({r[1] for r in records}) == 5219
    sequences = sorted((r[1] for r in records),key=len)
    cache = {}
    start = time.perf_counter()
    batch = 64
    for i in range(0,len(sequences),batch):
        seqs = sequences[i:i+batch]
        cache.update(zip(seqs,encode(seqs)))
        if i % 640 == 0 or i+batch >= len(sequences):
            print(f'Encoded {len(cache)}/5219 in {time.perf_counter()-start:.2f} s',flush=True)
    duration = time.perf_counter()-start
    arr = np.stack([cache[s] for s in ref['sequence']])
    cosine = (arr*ref['embedding']).sum(1)/(np.linalg.norm(arr,axis=1)*np.linalg.norm(ref['embedding'],axis=1))
    max_error = float(np.max(np.abs(arr-ref['embedding'])))
    assert cosine.min()>0.999999 and max_error<1e-4
    np.save(args.output/'esm2_cache_full.npy',cache)
    np.savez_compressed(args.output/'full_embeddings.npz',sequence_id=np.array([r[0] for r in records]),sequence_embedding=np.stack([cache[r[1]] for r in records]))
    meta = {'model':'esm2_t6_8M_UR50D','checkpoint_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),'sequences':len(cache),'dimension':320,'layer':6,'pooling':'mean over residues, excluding BOS/EOS and padding','device':'CPU','threads':4,'batch_size':batch,'extraction_loop_seconds':duration,'reference_overlap':len(arr),'reference_max_abs_difference':max_error,'reference_min_cosine':float(cosine.min()),'torch_version':str(torch.__version__),'implementation':'checkpoint-backed PyTorch inference, validated against the archived fair-esm embedding caches'}
    (args.output/'extraction_provenance.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':
    main()
