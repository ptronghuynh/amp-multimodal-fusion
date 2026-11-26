# Homology-Aware Comparison of PLM-Physicochemical Fusion Reveals Sensitivity-Specificity Trade-offs in AMP Prediction

Reproducibility repository for manuscript 4734, revised submission (2026-11-26).

## Study design

The primary experiment uses a fixed collection of 5,219 peptide sequences (4,175 training, 523 validation, 521 test). Six configurations (Seq-only, Phys-only, Concat, Cross-attn, Concat-matched, Concat-active-matched) are evaluated over seeds 42, 52, 62, 72 and 82. The frozen encoder is ESM-2 t6 8M, with 320-dimensional residue-mean-pooled representations. The seven-coordinate primary physicochemical input is not standardized. Training uses Adam (learning rate 0.001), BCEWithLogitsLoss, batch size 64, up to 12 epochs and validation-loss early stopping (patience 4).

The alignment sensitivity analysis applies 40% identity and bidirectional alignment-length/full-sequence-length coverage of at least 0.8. The PAM30 searches found zero qualifying pairs under those settings; the complementary BLOSUM62 standard and permissive searches found 35 and 36 cross-partition pairs. Their union with comparison set A contains 47 pairs. Excluding test sequences directly linked to training or validation sequences yields test cohorts of 502 (standard BLOSUM62) and 496 (union), without retraining. All five paired ROC-AUC contrasts retain their direction and 95% percentile intervals excluding zero. The test sensitivity analysis does not remove train-validation relationships or establish universal absence of homology.

## Verify the published results

From the repository root, use Python 3.11 or later in an isolated environment.

```sh
python -m pip install numpy pandas scikit-learn
python verify_release.py
```

The verifier checks the release hashes, sample counts, 30 fitted states, primary metrics from predictions and filtered metrics from published per-seed summaries, five paired contrasts in each of three cohorts, and Table 7 from 40 per-seed external evaluations. The 23 canonical manuscript tables are under `tables/` and the seven final SVGs under `figures/`. The primary predictions and split manifests use study-assigned identifiers. Alignment evidence in this release is limited to study-level and per-run statistics. Pair records, coordinates and individual exclusion masks are held under controlled access. The thirty PyTorch files under `primary/outputs/` contain fitted model parameters.

## Original data and permissions

| Source | Access and attribution |
|---|---|
| AMPDiscover / starPep | Download the general AMP training, test and external sets from the [authors' dataset page](https://biocom-ampdiscover.cicese.mx/dataset). The authors provide them for reproducibility and method testing. Public redistribution of the underlying compilation could not be confirmed. |
| DBAASP | Data were obtained from the Database of Antimicrobial Activity and Structure of Peptides. Its [terms](https://www.dbaasp.org/terms-and-conditions) permit redistribution with acknowledgement. Cite Pirtskhalava et al., [DBAASP v3](https://doi.org/10.1093/nar/gkaa991). |
| DRAMP | The [DRAMP database](https://dramp.cpu-bioinfor.com/) states CC BY 4.0, requests original-study citations for general and clinical records, and directs users to obtain authorization for patented AMPs. |
| LAMP | Follow the database link in the [LAMP2 publication](https://doi.org/10.1093/database/baaa061). A public redistribution grant covering the exact source records used here could not be confirmed. |
| YADAMP | See Piotto et al., [YADAMP](https://doi.org/10.1016/j.ijantimicag.2011.12.003). A public redistribution grant for the underlying records could not be confirmed. |
| AMPlify | The [authors' project](https://github.com/bcgsc/AMPlify) provides the original resource. The reported external evaluation is available here as per-seed metrics. |
| ESM-2 | Retrieve the [original frozen encoder](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt). The upstream MIT notice is in `primary/inputs/ESM-LICENSE.txt` and the expected weight digest is in `primary/outputs/extraction_provenance.json`. |

The mixed-source sequences, sequence-indexed embeddings, cached inputs and original evidence archives are held separately under controlled access. The public files give no onward licence to those materials. For the exact study partitions or reference inputs, request authorized access from the corresponding author through the journal. A current upstream download may differ from the locked study snapshot. The repository provides no accession-to-study-ID crosswalk sufficient to reconstruct the original FASTA files from live databases alone.

## Reproduce the primary experiment with authorized inputs

1. Place the original train, validation and test FASTA files beside `primary/inputs/sequence_splits_blast40cov_FINAL.csv`. From that directory, run `sha256sum -c b40cov_manifest_sha256.txt` to verify all four files.
2. Place the verified encoder in `primary/inputs/esm2_t6_8M_UR50D.pt`, the authorized reference in `primary/source/verified_partial_embeddings.npz`, and the original functional-check states in `primary/inputs/checkpoints.zip`. The optional `primary/source/cache_subset_inputs.npz` supports a further embedding comparison.
3. From the root, install the recorded dependencies with `python -m pip install -r primary/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`. Use a fresh output directory for each run.

```bash
python primary/source/extract_esm2.py --root primary/inputs --weights primary/inputs/esm2_t6_8M_UR50D.pt --reference primary/source/verified_partial_embeddings.npz --output primary/outputs_reproduced
python primary/source/run_full_verified.py --root primary/inputs --output primary/outputs_reproduced --cache primary/outputs_reproduced/esm2_cache_full.npy
python primary/source/postprocess_full.py --root primary/inputs --output primary/outputs_reproduced
python primary/source/build_figures.py --data primary/outputs --out figures_reproduced
```

`primary/source/feature_provenance.json` identifies the original notebook and feature definition. Alignment scripts are under `alignment_sensitivity/`. Running the original sensitivity pipeline also requires the authorized FASTA inputs and original coverage evidence ZIP. The recorded CPU timing experiment uses `python runtime/benchmark_cpu.py runtime/cached_model_inputs.zip --out runtime_reproduced` with the separately obtained input archive. The aggregate external values can be independently checked from the public metrics. The source snapshot lacks the per-peptide prediction files for the 40 external evaluations and the original AMPlify split CSVs. Historical HemoPI-3 and supplementary comparisons are reported at the level supported by the supplied tables.

The public release contains the original measurements. No experiments were rerun to prepare it.

The shape check in external notebook cell 165 reads its example from the locally supplied training set. The original cell embedded a database sequence. The feature definition and recorded results are unchanged.
