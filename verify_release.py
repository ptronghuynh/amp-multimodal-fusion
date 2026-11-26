"""Check integrity and numerical correspondence with manuscript 4734."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
SEEDS = {42, 52, 62, 72, 82}
MODELS = ['Seq-only', 'Phys-only', 'Concat', 'Cross-attn', 'Concat-matched', 'Concat-active-matched']


def sha(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / 'tables' / f'{name}.csv', dtype=str)


def fmt(mean: float, sd: float) -> str:
    return f'{mean:.4f} ± {sd:.4f}'


def mean_sd(values: pd.Series) -> str:
    return fmt(values.mean(), values.std(ddof=1))


def close_string(observed: str, expected: str) -> None:
    assert observed.replace('−', '-').strip() == expected.replace('−', '-').strip(), (observed, expected)


def metrics(pred: pd.DataFrame) -> dict:
    y, p = pred.label.to_numpy(), pred.prob.to_numpy()
    b = p >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, b, labels=[0, 1]).ravel()
    return dict(AUC=roc_auc_score(y, p), AP=average_precision_score(y, p),
                ACC=accuracy_score(y, b), F1=f1_score(y, b), MCC=matthews_corrcoef(y, b),
                Sensitivity=tp/(tp+fn), Specificity=tn/(tn+fp))


def main() -> None:
    for line in (ROOT / 'SHA256SUMS.txt').read_text().splitlines():
        expected, rel = line.split('  ', 1)
        assert sha(ROOT / rel) == expected, rel
    print('PASS: snapshot SHA-256 checksums')
    primary = ROOT / 'primary' / 'outputs'
    manifest = pd.read_csv(primary / 'full_manifest.csv')
    assert manifest.id.is_unique and len(manifest) == 5219
    assert manifest.groupby('split').size().to_dict() == {'train':4175,'val':523,'test':521}
    predictions = pd.read_csv(primary / 'sevenD_predictions.csv')
    assert len(predictions) == 30 * (521 + 523)
    assert set(predictions.model) == set(MODELS) and set(predictions.seed) == SEEDS
    assert predictions.groupby(['model','seed','split']).size().value_counts().to_dict() == {521:30,523:30}
    assert predictions.groupby(['model','seed','split']).id.nunique().value_counts().to_dict() == {521:30,523:30}
    indexed = manifest.set_index('id')
    assert predictions.id.map(indexed.label).equals(predictions.label)
    assert predictions.id.map(indexed.split).equals(predictions.split)
    assert len(list(primary.glob('7D_*_seed*.pt'))) == 30
    test = predictions[predictions.split.eq('test')]
    assert np.isfinite(test.prob).all()
    per_seed = pd.read_csv(ROOT/'alignment_sensitivity/per_seed_metrics_sensitivity.csv')
    for cohort, count in [('full',521), ('standard_filtered',502), ('union_filtered',496)]:
        records = per_seed[per_seed.cohort.eq(cohort)]
        assert len(records) == 30 and records.set_index(['model','seed']).index.is_unique
        assert set(records.model) == set(MODELS) and set(records.seed) == SEEDS
        assert records.n.eq(count).all() and records[['TP','TN','FP','FN']].sum(axis=1).eq(count).all()
    union = per_seed[per_seed.cohort.eq('union_filtered')]
    for _, row in table('Table_S14').iterrows():
        group = union[union.model.eq(row['Model'])]
        for column, metric in [('ROC-AUC','AUC'),('AP','AP'),('MCC','MCC'),('Sensitivity','Sensitivity'),('Specificity','Specificity')]:
            close_string(row[column],mean_sd(group[metric]))
    print('PASS: Table_S14 from published per-seed metrics')
    for data, filename, mapping in [
        (test, 'Table_3', {'ROC-AUC':'AUC','AP':'AP','ACC':'ACC','F1':'F1','MCC':'MCC'}),
        (test, 'Table_6', {'ROC-AUC':'AUC','AP':'AP'}),
    ]:
        rows = []
        for (m,seed), g in data.groupby(['model','seed']): rows.append(dict(model=m,seed=seed,**metrics(g)))
        result = pd.DataFrame(rows)
        for _, row in table(filename).iterrows():
            g = result[result.model.eq(row['Model'])]
            for column, metric in mapping.items():close_string(row[column],mean_sd(g[metric]))
        print(f'PASS: {filename} recomputed from sequence predictions')
    # Compare the stored resampling distributions with the reported intervals,
    # and independently check mean ROC-AUC differences using saved probabilities.
    for cohort in ['full', 'standard_filtered', 'union_filtered']:
        drawsfile = f'bootstrap_draws_{cohort}.csv'
        g = pd.read_csv(ROOT/'alignment_sensitivity/paired_contrasts_sensitivity.csv').query('cohort == @cohort')
        draws = pd.read_csv(ROOT/'alignment_sensitivity'/drawsfile)
        assert len(g)==5 and len(draws)==2000
        auc = per_seed[per_seed.cohort.eq(cohort)].set_index(['model','seed']).AUC.to_dict()
        if cohort == 'full':
            for key, group in test.groupby(['model','seed']):
                assert np.isclose(auc[key], roc_auc_score(group.label, group.prob))
        for r in g.itertuples():
            a,b = r.comparison.split(' - ')
            delta = np.mean([auc[a,s]-auc[b,s] for s in SEEDS])
            z = draws[r.comparison].to_numpy()
            assert np.allclose([delta,z.std(ddof=1),*np.percentile(z,[2.5,97.5]),np.mean(z<=0)], [r.delta,r.bootstrap_sd,r.ci_low,r.ci_high,r.bootstrap_fraction_nonpositive],atol=1e-10,rtol=1e-9)
            assert r.ci_low*r.ci_high>0
            if cohort=='full':
                row = table('Table_4').loc[lambda t:t.iloc[:,0].str.replace('−','-',regex=False).eq(r.comparison)].iloc[0]
                close_string(row.iloc[1], f'{r.delta:+.4f} ± {r.bootstrap_sd:.4f}')
                close_string(row.iloc[2], f'[{r.ci_low:.4f}, {r.ci_high:.4f}]')
                close_string(row.iloc[3], f'{r.bootstrap_fraction_nonpositive:.4f}')
            else:
                row = table('Table_S13').loc[lambda t:t.iloc[:,0].str.replace('−','-',regex=False).eq(r.comparison)].iloc[0]
                col = 1 if cohort=='standard_filtered' else 3
                close_string(row.iloc[col], f'{r.delta:+.4f} ± {r.bootstrap_sd:.4f}')
                close_string(row.iloc[col+1], f'[{r.ci_low:.4f}, {r.ci_high:.4f}]')
        print(f'PASS: {cohort}, five paired contrasts and stored bootstrap draws')
    ext = pd.read_csv(ROOT/'external/table7_per_seed_verified.csv')
    assert len(ext)==40 and ext.groupby(['dataset','model']).size().eq(5).all()
    assert ext.groupby('dataset').n_test.unique().map(len).eq(1).all()
    assert ext.groupby('dataset').n_test.first().to_dict()=={'AMPlify':1617,'AMPDiscover':15327}
    assert (ext[['TP','TN','FP','FN']].sum(axis=1)==ext.n_test).all()
    assert np.allclose(ext.Sensitivity,ext.TP/(ext.TP+ext.FN))
    assert np.allclose(ext.Specificity,ext.TN/(ext.TN+ext.FP))
    den = np.sqrt((ext.TP+ext.FP)*(ext.TP+ext.FN)*(ext.TN+ext.FP)*(ext.TN+ext.FN))
    assert np.allclose(ext.MCC,(ext.TP*ext.TN-ext.FP*ext.FN)/den)
    names = {'Seq-only':'seq_only','Phys-only':'phys_only','Concat':'concat','Cross-attn':'cross_attn'}
    for _, row in table('Table_7').iterrows():
        g=ext[ext.dataset.eq(row.Dataset)&ext.model.eq(names[row.Model])]
        for m in ('AUC','AP','MCC','Sensitivity','Specificity'):close_string(row[m],mean_sd(g[m]))
    print('PASS: Table 7 aggregates from 40 per-seed metric records')
    assert len(list((ROOT/'tables').glob('*.csv')))==23
    assert len(list((ROOT/'figures').glob('*.svg')))==7
    print('PASS: 23 reported tables, 7 final SVGs, 30 primary fitted checkpoints')


if __name__=='__main__':
    main()
