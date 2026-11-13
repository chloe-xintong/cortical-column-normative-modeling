#!/usr/bin/env python3
"""Prepare, validate, fit and compare Figure 5 Option 2-FS baselines.

Description
-----------
Prepare, validate, fit and compare Figure 5 Option 2-FS baselines.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``stage`` command-line input
- ``--output`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure5/scripts/option2_fs_baselines.py --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESULTS = Path('/mnt/newStor/paros/paros_WORK/xintong/results')
REFERENCE = RESULTS / 'figure5_age_prediction/outputs'
NATIVE = Path('/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer')
O2 = RESULTS / 'qc/o2_fs'
MODELS = ('o2fs_ico4_md', 'o2fs_ico4_md_length')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(command, env=None):
    subprocess.run([str(v) for v in command], check=True, env=env)


def scalar(path, *, allow_nan=False):
    values = np.asarray(nib.load(str(path)).dataobj).reshape(-1)
    if (values.shape != (163842,) or np.isinf(values).any()
            or (not allow_nan and np.isnan(values).any())):
        raise ValueError(f'Invalid fsaverage scalar: {path}, {values.shape}')
    return values


def subjects():
    s = pd.read_csv(REFERENCE / 'features/subject_order.csv', dtype={'subject_id': str})
    if len(s) != 196 or s.subject_id.duplicated().any():
        raise ValueError('Expected exactly 196 unique reference subjects')
    if not np.array_equal(s.row_index, np.arange(196)):
        raise ValueError('Noncanonical reference row order')
    return s


def validate_folds(ids):
    outer = json.loads((REFERENCE / 'folds/outer_folds.json').read_text())
    inner = json.loads((REFERENCE / 'folds/inner_folds_by_outer_fold.json').read_text())
    if len(outer['folds']) != 5:
        raise ValueError('Expected five outer folds')
    tested = []
    for fold in outer['folds']:
        train, test = fold['train_subject_ids'], fold['test_subject_ids']
        if set(train) & set(test) or sorted(train + test) != sorted(ids):
            raise ValueError('Invalid outer partition')
        splits = inner['outer_folds'][str(fold['outer_fold'])]['folds']
        if len(splits) != 5:
            raise ValueError('Expected five inner folds')
        validated = []
        for split in splits:
            a, b = split['train_subject_ids'], split['validation_subject_ids']
            if set(a) & set(b) or sorted(a + b) != sorted(train):
                raise ValueError('Invalid inner partition or leakage')
            validated.extend(b)
        if sorted(validated) != sorted(train):
            raise ValueError('Inner validation coverage mismatch')
        tested.extend(test)
    if sorted(tested) != sorted(ids):
        raise ValueError('Outer OOF coverage mismatch')


def prepare(args):
    subject = subjects().iloc[args.subject_index].subject_id
    target = args.output / 'source' / subject
    target.mkdir(parents=True, exist_ok=True)
    fs_home = Path('/home/apps/freesurfer')
    env = os.environ.copy()
    env.update(FREESURFER_HOME=str(fs_home), PATH=f'{fs_home}/bin:' + env['PATH'])
    with tempfile.TemporaryDirectory(prefix='f5-o2fs-') as temp:
        sd = Path(temp)
        (sd / subject).symlink_to(NATIVE / subject)
        (sd / 'fsaverage').symlink_to(fs_home / 'subjects/fsaverage')
        env['SUBJECTS_DIR'] = str(sd)
        for hemi in ('lh', 'rh'):
            existing = O2 / subject / 'fsaverage' / hemi
            # Existing MD is read-only; missing subjects are generated in this experiment.
            md_dir = existing if (existing / 'vol2surf_manifest.json').exists() else target / hemi / 'md'
            if not (md_dir / 'vol2surf_manifest.json').exists():
                run([sys.executable, REPO / 'analysis_results/figure2/o2_fs/run_vol2surf.py',
                     '--subject-id', subject, '--subjects-dir', sd,
                     '--metric-image', Path('/mnt/newStor/paros/paros_WORK/xintong/data/processed/dwi') / subject / f'{subject}_md.nii.gz',
                     '--reg', RESULTS / 'fsaverage_result/columns' / subject / 'registration' / f'{subject}_DWI2T1.dat',
                     '--template', 'fsaverage', '--hemisphere', hemi,
                     '--depths', ','.join(f'{x:.2f}' for x in np.linspace(0, 1, 21)),
                     '--output-dir', md_dir], env)
            white, wf = fsio.read_geometry(NATIVE / subject / 'surf' / f'{hemi}.white')
            pial, pf = fsio.read_geometry(NATIVE / subject / 'surf' / f'{hemi}.pial')
            if white.shape != pial.shape or not np.array_equal(wf, pf):
                raise ValueError('Native white/pial vertex correspondence failed')
            length = np.linalg.norm(pial - white, axis=1)
            if not np.isfinite(length).all() or np.any(length < 0):
                raise ValueError('Invalid native column lengths')
            native_path = target / f'{hemi}.native_column_length.mgh'
            nib.save(nib.MGHImage(length.astype(np.float32).reshape(-1, 1, 1), np.eye(4)), native_path)
            native_spots = np.unique([0, len(length) // 2, len(length) - 1])
            saved_native = np.asarray(nib.load(native_path).dataobj).reshape(-1)
            if not np.allclose(saved_native[native_spots],
                               np.linalg.norm(pial[native_spots] - white[native_spots], axis=1)):
                raise ValueError('Native Euclidean length verification failed')
            mapped = target / f'{hemi}.fsaverage_column_length.mgh'
            command = ['mri_surf2surf', '--srcsubject', subject, '--trgsubject', 'fsaverage',
                       '--hemi', hemi, '--sval', native_path, '--tval', mapped,
                       '--surfreg', 'sphere.reg', '--mapmethod', 'nnfr', '--noreshape']
            with (target / f'{hemi}.length_mapping.log').open('w') as log:
                subprocess.run([str(x) for x in command], env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            scalar(mapped)
            (target / f'{hemi}.sources.json').write_text(json.dumps({
                'subject_id': subject, 'hemisphere': hemi, 'md_directory': str(md_dir),
                'length_definition': 'Euclidean native pial minus corresponding native white vertex',
                'mapping_command': [str(x) for x in command],
                'native_length_sha256': digest(native_path), 'mapped_length_sha256': digest(mapped),
                'native_surface_sha256': {str(NATIVE / subject / 'surf' / f'{hemi}.{v}'):
                    digest(NATIVE / subject / 'surf' / f'{hemi}.{v}') for v in ('white', 'pial', 'sphere.reg')},
                'native_length_spot_checks': int(len(native_spots)),
            }, indent=2) + '\n')


def build(args):
    s = subjects()
    validate_folds(s.subject_id.tolist())
    f = args.output / 'features'
    (f / 'matrices').mkdir(parents=True, exist_ok=True)
    (f / 'feature_metadata').mkdir(exist_ok=True)
    s.to_csv(f / 'subject_order.csv', index=False)
    nodes = pd.read_csv(REFERENCE / 'template/ico4_vertex_order.csv')
    if not nodes.in_analysis_mask.isin([True, False]).all():
        raise ValueError('Invalid analysis mask')
    nodes = nodes.loc[nodes.in_analysis_mask].reset_index(drop=True)
    length_meta = pd.read_csv(REFERENCE / 'features/feature_metadata/features_ico4_column_length.csv')
    md_meta = pd.read_csv(REFERENCE / 'features/feature_metadata/features_ico4_md.csv')
    keys = ['hemisphere', 'ico_node_id', 'fsaverage_vertex_id']
    if len(nodes) != 4687 or not nodes[keys].equals(length_meta[keys]):
        raise ValueError('Reference ico4 vertex order mismatch')
    expanded = nodes.loc[nodes.index.repeat(21)].reset_index(drop=True)
    if not expanded[keys].equals(md_meta[keys]) or not np.array_equal(md_meta.depth_index, np.tile(np.arange(21), 4687)):
        raise ValueError('Reference MD flattening order mismatch')
    if not md_meta.depth_direction.eq('pial_to_white').all():
        raise ValueError('Reference depth direction mismatch')
    md = np.lib.format.open_memmap(f / 'matrices/X_o2fs_ico4_md.npy', mode='w+', dtype='float32', shape=(196, 98427))
    lengths = np.lib.format.open_memmap(f / 'matrices/X_o2fs_ico4_column_length.npy', mode='w+', dtype='float32', shape=(196, 4687))
    spots, audit, sources = [], [], {}
    for row, subject in enumerate(s.subject_id):
        md_parts, length_parts = [], []
        offset = 0
        for hemi in ('lh', 'rh'):
            ids = nodes.loc[nodes.hemisphere.eq(hemi), 'fsaverage_vertex_id'].to_numpy(int)
            source_dir = args.output / 'source' / subject
            provenance = json.loads((source_dir / f'{hemi}.sources.json').read_text())
            manifest_path = Path(provenance['md_directory']) / 'vol2surf_manifest.json'
            manifest = json.loads(manifest_path.read_text())
            outputs = manifest['outputs']
            if (manifest['subject_id'] != subject or manifest['hemisphere'] != hemi
                    or manifest['template'] != 'fsaverage' or manifest['interp'] != 'trilinear'
                    or manifest['depth_direction'] != 'pial_to_white' or len(outputs) != 21):
                raise ValueError(f'Invalid MD manifest: {manifest_path}')
            if not np.allclose([v['depth'] for v in outputs], np.linspace(0, 1, 21)) or not np.allclose([v['projfrac'] for v in outputs], np.linspace(1, 0, 21)):
                raise ValueError(f'Unverified depth order: {manifest_path}')
            paths = [Path(v['output']) for v in outputs]
            values = np.column_stack([scalar(path, allow_nan=True)[ids] for path in paths])
            mapped_path = source_dir / f'{hemi}.fsaverage_column_length.mgh'
            if digest(mapped_path) != provenance['mapped_length_sha256']:
                raise ValueError('Mapped length changed since generation')
            length = scalar(mapped_path)[ids]
            md_parts.append(values.reshape(-1))
            length_parts.append(length)
            sources[str(manifest_path)] = digest(manifest_path)
            sources[str(mapped_path)] = digest(mapped_path)
            sources[str(source_dir / f'{hemi}.sources.json')] = digest(source_dir / f'{hemi}.sources.json')
            # Reopen sources independently of the matrix-construction arrays.
            for node in (0, len(ids) // 2, len(ids) - 1):
                for depth in (0, 10, 20):
                    spots.append(dict(row_index=row, subject_id=subject, hemisphere=hemi,
                                      kind='md', feature_index=(offset + node) * 21 + depth,
                                      fsaverage_vertex_id=int(ids[node]), depth_index=depth,
                                      source_value=float(scalar(paths[depth], allow_nan=True)[ids[node]])))
                spots.append(dict(row_index=row, subject_id=subject, hemisphere=hemi,
                                  kind='length', feature_index=offset + node,
                                  fsaverage_vertex_id=int(ids[node]), depth_index=-1,
                                  source_value=float(scalar(mapped_path)[ids[node]])))
            audit.append(dict(subject_id=subject, hemisphere=hemi, n_vertices=len(ids),
                              md_nan_count=int(np.isnan(values).sum()),
                              md_finite_fraction=float(np.isfinite(values).mean()),
                              n_depths=21, first_projfrac=outputs[0]['projfrac'], last_projfrac=outputs[-1]['projfrac']))
            offset += len(ids)
        md[row] = np.concatenate(md_parts)
        lengths[row] = np.concatenate(length_parts)
        print(f'validated_subject={row + 1}/196', flush=True)
    md.flush(); lengths.flush()
    for spot in spots:
        matrix = md if spot['kind'] == 'md' else lengths
        spot['matrix_value'] = float(matrix[spot['row_index'], spot['feature_index']])
        spot['passed'] = bool(np.isclose(spot['source_value'], spot['matrix_value'], rtol=1e-6, atol=1e-10, equal_nan=True))
    if len(spots) < 72 or not all(v['passed'] for v in spots):
        raise ValueError('Source-value spot checks failed')
    if np.isinf(md).any() or not np.isfinite(lengths).all():
        raise ValueError('Infinite MD or nonfinite length feature values')
    pd.DataFrame(spots).to_csv(f / 'source_value_spot_checks.csv', index=False)
    pd.DataFrame(audit).to_csv(f / 'source_audit.csv', index=False)
    md_meta.to_csv(f / 'feature_metadata/features_o2fs_ico4_md.csv', index=False)
    length_meta.to_csv(f / 'feature_metadata/features_o2fs_ico4_column_length.csv', index=False)
    files = [*sorted((f / 'matrices').glob('*.npy')), f / 'subject_order.csv',
             *sorted((f / 'feature_metadata').glob('*.csv')),
             REFERENCE / 'features/subject_order.csv', REFERENCE / 'template/ico4_vertex_order.csv',
             REFERENCE / 'folds/outer_folds.json', REFERENCE / 'folds/inner_folds_by_outer_fold.json']
    report = dict(status='passed', n_subjects=196, n_vertices=4687,
                  md_shape=list(md.shape), length_shape=list(lengths.shape),
                  source_value_checks=len(spots), depth_direction='pial_to_white',
                  md_nan_count=int(np.isnan(md).sum()),
                  md_columns_with_nan=int(np.isnan(md).any(axis=0).sum()),
                  md_all_nan_columns=int(np.isnan(md).all(axis=0).sum()),
                  missing_value_policy='Preserve MD NaNs; median imputation fitted only on training folds by the shared Ridge pipeline',
                  file_sha256={str(p): digest(p) for p in files}, source_sha256=sources)
    (f / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')


def train(args):
    f = args.output / 'features'
    report = json.loads((f / 'validation.json').read_text())
    if report['status'] != 'passed' or report['source_value_checks'] < 72:
        raise ValueError('Feature validation gate failed')
    for path, expected in report['file_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'Validated input changed: {path}')
    if not pd.read_csv(f / 'subject_order.csv').equals(subjects()):
        raise ValueError('Training subject order differs from reference')
    run([sys.executable, HERE / 'run_ridge.py', '--representation', MODELS[args.model_index],
         '--feature-dir', f, '--fold-dir', REFERENCE / 'folds',
         '--output-root', args.output / 'models_lsqr', '--solver', 'lsqr', '--alpha-count', '21'])


def compare(args):
    out = args.output / 'comparison'
    out.mkdir(parents=True, exist_ok=True)
    s = subjects()
    predictions, summaries, alphas = {}, [], []
    for name in (*MODELS, 'ico4_md', 'ico4_md_length'):
        root = args.output if name in MODELS else REFERENCE
        model_dir = root / 'models_lsqr' / name
        metrics = json.loads((model_dir / 'metrics.json').read_text())
        if metrics['ridge_solver'] != 'lsqr' or not np.allclose(metrics['alpha_grid'], np.logspace(-4, 6, 21)):
            raise ValueError('Reference solver or alpha mismatch')
        p = pd.read_csv(model_dir / f'{name}_oof_predictions.csv')
        if not np.array_equal(p.row_index, s.row_index) or not np.allclose(p.true_age, s.age_at_scan) or not np.isfinite(p.predicted_age).all():
            raise ValueError('Incomplete or misordered OOF predictions')
        p.insert(1, 'subject_id', s.subject_id)
        predictions[name] = p
        p.to_csv(out / f'{name}_oof_predictions.csv', index=False)
        summaries.append(dict(model=name, n=196, mae=mean_absolute_error(p.true_age, p.predicted_age),
                              rmse=np.sqrt(mean_squared_error(p.true_age, p.predicted_age)),
                              pearson_r=pearsonr(p.true_age, p.predicted_age).statistic,
                              r2=r2_score(p.true_age, p.predicted_age)))
        a = p[['outer_fold', 'selected_alpha']].drop_duplicates()
        if len(a) != 5:
            raise ValueError('Expected five selected alphas')
        a.insert(0, 'model', name); alphas.append(a)
    rng = np.random.default_rng(20261102)
    boot_rows, pairs = [], []
    for a, b in ((MODELS[0], 'ico4_md'), (MODELS[1], 'ico4_md_length'), (MODELS[0], MODELS[1])):
        pa, pb = predictions[a], predictions[b]
        if not np.array_equal(pa.outer_fold, pb.outer_fold):
            raise ValueError('Paired outer fold assignments differ')
        ea = np.abs(pa.predicted_age.to_numpy() - s.age_at_scan.to_numpy())
        eb = np.abs(pb.predicted_age.to_numpy() - s.age_at_scan.to_numpy())
        delta = ea - eb
        draws = delta[rng.integers(0, 196, (10000, 196))].mean(axis=1)
        pd.DataFrame({'replicate': np.arange(10000), 'delta_mae_a_minus_b': draws}).to_csv(out / f'{a}_minus_{b}_bootstrap.csv', index=False)
        ci = np.percentile(draws, [2.5, 97.5])
        boot_rows.append(dict(model_a=a, model_b=b, delta_mae_a_minus_b=delta.mean(),
                              ci95_lower=ci[0], ci95_upper=ci[1], bootstrap_replicates=10000,
                              seed=20261102))
        pairs.append(pd.DataFrame(dict(subject_id=s.subject_id, row_index=s.row_index,
                                       model_a=a, model_b=b, absolute_error_a=ea,
                                       absolute_error_b=eb, difference_a_minus_b=delta)))
    pd.DataFrame(summaries).to_csv(out / 'model_metrics.csv', index=False)
    pd.concat(alphas).to_csv(out / 'selected_alphas.csv', index=False)
    pd.concat(pairs).to_csv(out / 'paired_absolute_errors.csv', index=False)
    pd.DataFrame(boot_rows).to_csv(out / 'paired_bootstrap_differences.csv', index=False)
    (out / 'complete.json').write_text(json.dumps(dict(status='complete', n_subjects=196,
         bootstrap_unit='paired subject OOF absolute errors; conditional on fixed fitted folds',
         difference_definition='MAE(model_a) - MAE(model_b); negative favors a'), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'build', 'train', 'compare'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--subject-index', type=int)
    parser.add_argument('--model-index', type=int)
    args = parser.parse_args()
    globals()[args.stage](args)
