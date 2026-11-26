"""Measure cached-representation inference using the trained 64-D subset models.
No training, device substitution, or comparison to undocumented timing conditions.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import platform
import time
import zipfile
import numpy as np
import torch
from torch import nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence_zip', type=Path)
    parser.add_argument('--out', type=Path, default=Path('runtime_cpu'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with zipfile.ZipFile(args.evidence_zip) as archive:
        # Use the exact model-class definitions from the reproducibility script,
        # without executing its training, data-loading or plotting sections.
        source = archive.read('source/run_analysis.py').decode('utf-8')
        definitions = source[source.index('class Seq(nn.Module):'):source.index('def count(m):')]
        namespace = {'torch': torch, 'nn': nn}
        exec(definitions, namespace)
        inp = archive.read('outputs/cache_subset_inputs.npz')
        with np.load(io.BytesIO(inp), allow_pickle=False) as data:
            s = torch.from_numpy(data['sequence_embedding'].copy())
            p = torch.from_numpy(((data['phys64'] - data['mean']) / data['std']).copy())
            assert s.shape == (507, 320) and p.shape == (507, 64)
            assert torch.isfinite(s).all() and torch.isfinite(p).all()
        models = {}
        checkpoints = {}
        for label, cls in [('Seq-only', 'Seq'), ('Concat', 'Concat'), ('Cross-attn', 'Cross')]:
            path = f'outputs/{label}_seed42.pt'
            state = archive.read(path)
            model = namespace[cls]()
            model.load_state_dict(torch.load(io.BytesIO(state), map_location='cpu', weights_only=True), strict=True)
            model.eval()
            models[label] = model
            checkpoints[label] = {'path': path, 'sha256': hashlib.sha256(state).hexdigest(), 'parameters': sum(x.numel() for x in model.parameters()), 'file_bytes': len(state)}
        assert [checkpoints[x]['parameters'] for x in models] == [41217, 98817, 148481]
    metadata = {'cpu': next((l.split(':', 1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')), platform.processor()), 'device': 'cpu', 'threads': 1, 'interop_threads': 1, 'python': platform.python_version(), 'pytorch': torch.__version__, 'dtype': 'float32', 'seed': 42, 'inputs': 507, 'input_sha256': hashlib.sha256(inp).hexdigest(), 'model_definitions_sha256': hashlib.sha256(definitions.encode()).hexdigest(), 'checkpoints': checkpoints, 'warmup_calls': 100, 'repetitions': 5, 'calls_per_repetition': 507, 'batch_sizes': [1, 64], 'scope': 'model forward plus sigmoid; preconstructed CPU tensors; no embedding extraction, feature computation, model loading, transfer, or I/O'}
    measurements = []
    rng = np.random.default_rng(42)
    with torch.inference_mode():
        for bs in [1,64]:
            # Each pass visits all 507 inputs once per batch position;
            # modular indexing gives fixed-size batches without padding.
            batches = []
            for j in range(507):
                indices = (j * bs + np.arange(bs)) % 507
                batches.append((s[indices], p[indices]))
            for model in models.values():
                for ss, pp in batches[:100]:
                    torch.sigmoid(model(ss, pp))
            for rep in range(5):
                for label in rng.permutation(list(models)):
                    model = models[label]
                    t0 = time.perf_counter_ns()
                    for ss, pp in batches:
                        out = torch.sigmoid(model(ss, pp))
                    sec = (time.perf_counter_ns()-t0)/1e9
                    assert out.shape == (bs,) and torch.isfinite(out).all()
                    measurements.append({'model': str(label), 'batch_size': bs, 'repetition': rep+1, 'calls': len(batches), 'peptide_evaluations': len(batches)*bs, 'elapsed_s': sec, 'ms_per_peptide': 1000*sec/(len(batches)*bs), 'peptides_per_s': len(batches)*bs/sec})
    summary = []
    for label in models:
        for bs in [1,64]:
            r = [v for v in measurements if v['model']==label and v['batch_size']==bs]
            summary.append({'model': label,'batch_size': bs,'latency_ms_per_peptide_mean': float(np.mean([v['ms_per_peptide'] for v in r])),'latency_ms_per_peptide_sd': float(np.std([v['ms_per_peptide'] for v in r], ddof=1)),'throughput_peptides_per_s_mean': float(np.mean([v['peptides_per_s'] for v in r])),'throughput_peptides_per_s_sd': float(np.std([v['peptides_per_s'] for v in r],ddof=1))})
    for file, rows in [('runtime_cpu_replicates.csv',measurements), ('runtime_cpu_summary.csv',summary)]:
        with (args.out/file).open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.out/'runtime_cpu_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({'metadata': metadata, 'summary':summary},indent=2))

if __name__ == '__main__':
    main()
