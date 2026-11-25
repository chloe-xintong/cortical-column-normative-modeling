"""Audit cached Figure 4B null and bootstrap nearest-bin residual gain.

No null fields, profiles or subject fits are generated. Bootstrap resamples
subjects after within-subject median aggregation over 100 null realizations.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


def main():
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort-dir', type=Path, default=Path(
        '/mnt/newStor/paros/paros_WORK/xintong/result/cortical_trajectory_analysis/'
        'analysis_results/figure4/outputs/coverage95_pass_313/'
        'acquisition_null_fwhm0_seedfix/cohort'))
    parser.add_argument('--output-dir', type=Path, default=repo /
                        'analysis_results/figure4/audits/figure4B_nearest_gain_313')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(repo / 'manifests/figure4_coverage95_pass_313_subjects.csv')
    source = args.cohort_dir
    curves = pd.read_csv(source / 'figure4B_acquisition_subject_realization_bins.csv',
                         dtype={'replicate_seed': str})
    assert set(curves.subject_id) == set(manifest.subject_id)
    assert curves.psf_fwhm_mm.eq(0).all()
    assert curves.laminar_recolored.all()
    assert not curves.duplicated(['subject_id', 'replicate', 'distance_bin_index']).any()
    assert curves.groupby('subject_id').replicate.nunique().eq(100).all()
    assert curves.groupby(['subject_id', 'replicate']).distance_bin_index.nunique().eq(15).all()

    keys = ['subject_id', 'replicate']
    far = curves.loc[curves.distance_bin_index.between(10, 14)].groupby(keys)[
        ['real_dissimilarity', 'null_dissimilarity']].median()
    nearest = curves.loc[curves.distance_bin_index.eq(0)].copy()
    nearest = nearest.join(far, on=keys, rsuffix='_far')
    recomputed = ((nearest.real_dissimilarity_far - nearest.real_dissimilarity)
                  - (nearest.null_dissimilarity_far - nearest.null_dissimilarity))
    formula_error = float(np.max(np.abs(recomputed - nearest.real_minus_null_local_excess)))
    # Cached CSVs contain float32-scale rounded values.
    assert formula_error < 1e-7
    gain = 'real_minus_null_local_excess'
    subject = nearest.groupby('subject_id')[gain].median().rename('nearest_gain').reset_index()
    cached = pd.read_csv(source / 'figure4B_acquisition_subject_summary.csv')
    cached = cached.loc[cached.distance_bin_index.eq(0)].set_index('subject_id')
    assert np.allclose(subject.set_index('subject_id').nearest_gain.sort_index(),
                       cached[gain].sort_index(), rtol=0, atol=1e-12)
    subject.to_csv(args.output_dir / 'nearest_gain_by_subject.csv', index=False)
    values = subject.nearest_gain.to_numpy()
    rng = np.random.default_rng(20261119)
    bootstrap = np.median(values[rng.integers(0, len(values), (10000, len(values)))], axis=1)
    low, high = np.quantile(bootstrap, [.025, .975])

    convergence = []
    for number in [1, 10, 25, 50, 100]:
        current = nearest.loc[nearest.replicate < number].groupby('subject_id')[gain].median()
        convergence.append({'n_realizations': number, 'n_subjects': len(current),
                            'nearest_gain_group_median': current.median()})
    convergence = pd.DataFrame(convergence)
    convergence.to_csv(args.output_dir / 'nearest_gain_convergence.csv', index=False)
    # Disjoint cached batches check dependence on the specific realizations.
    batches = []
    for start in range(0, 100, 10):
        current = nearest.loc[nearest.replicate.between(start, start+9)].groupby('subject_id')[gain].median()
        batches.append({'first_replicate': start, 'last_replicate': start+9,
                        'n_subjects': len(current), 'nearest_gain_group_median': current.median()})
    batches = pd.DataFrame(batches)
    batches.to_csv(args.output_dir / 'nearest_gain_disjoint_seed_batches.csv', index=False)
    single = nearest.groupby('replicate')[gain].median()
    single.rename('nearest_gain_group_median').to_csv(args.output_dir / 'nearest_gain_by_realization.csv')
    seeds = nearest[['subject_id', 'replicate', 'replicate_seed']]
    assert not seeds.replicate_seed.duplicated().any()
    validation = pd.read_csv(source / 'figure4B_acquisition_validation_by_subject.csv')
    assert set(validation.subject_id) == set(manifest.subject_id)
    assert validation.mask_identical.all()
    assert validation.count_mismatch_rows.eq(0).all()
    report = {
        'source_cohort_dir': str(source), 'n_subjects': len(subject),
        'nearest_bin': '(0,2] mm', 'psf_fwhm_mm': 0,
        'extra_gaussian_smoothing': False,
        'plateau': 'median of five 2-mm dissimilarity bins spanning (20,30] mm, separately per subject and null realization',
        'estimand': 'across-subject median of within-subject median-over-realizations E_s,r(0,2]',
        'formula': 'E_s,r(d) = [P_real_s - D_real_s(d)] - [P_null_s,r - D_null_s,r(d)]',
        'max_formula_reconstruction_error': formula_error,
        'median': float(np.median(values)),
        'q25': float(np.quantile(values, .25)), 'q75': float(np.quantile(values, .75)),
        'bootstrap_ci95_lower': float(low), 'bootstrap_ci95_upper': float(high),
        'bootstrap_replicates': 10000, 'bootstrap_seed': 20261119,
        'bootstrap_unit': 'subject', 'bootstrap_conditional_on': 'cached 100-realization null per subject',
        'positive_subjects': int((values > 0).sum()),
        'realizations_per_subject': 100, 'unique_cached_replicate_seed_tokens': seeds.replicate_seed.nunique(),
        'single_realization_group_median_min': float(single.min()),
        'single_realization_group_median_max': float(single.max()),
        'disjoint_10_realization_batch_median_min': float(batches.nearest_gain_group_median.min()),
        'disjoint_10_realization_batch_median_max': float(batches.nearest_gain_group_median.max()),
        'retained_cohort_max_depth_mean_mae': float(validation.depth_mean_mae.max()),
        'retained_cohort_max_depth_sd_mae': float(validation.depth_sd_mae.max()),
        'retained_cohort_max_covariance_mae': float(validation.covariance_mae.max()),
        'conclusion': 'After plateau correction, real data retain local similarity gain beyond the current acquisition-and-sampling null. This does not isolate physiology.',
    }
    (args.output_dir / 'audit_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    print(convergence.to_string(index=False))


if __name__ == '__main__':
    main()
