"""Fixed-pair coverage and common-anchor sensitivity; real profiles only.

Reuses subject anchor-bin caches generated with formal pairs and each
subject's valid-profile mask. No resampling of pairs or null generation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd

from spatial_metrics import aggregate_subject_bins
from fit_correlation_lengths import fit_curve


KEY = ['hemisphere', 'anchor_vertex']
BIN = 'distance_bin_index'


def summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {'n': len(values), 'median': float(np.median(values)) if len(values) else None,
            'q25': float(np.quantile(values, .25)) if len(values) else None,
            'q75': float(np.quantile(values, .75)) if len(values) else None}


def main():
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(
        '/mnt/newStor/paros/paros_WORK/xintong/results/qc/full_cohort_spatial_scale/figure4_ico5_formal_k15'))
    parser.add_argument('--output-dir', type=Path, default=repo /
                        'analysis_results/figure4/audits/common_anchor_sensitivity_313')
    parser.add_argument('--smoke', action='store_true', help='Use first three participants; separate output directory required.')
    args = parser.parse_args()
    started = time.time()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    source = args.source
    manifest_path = repo / 'manifests/figure4_coverage95_pass_313_subjects.csv'
    roster = pd.read_csv(manifest_path)
    assert len(roster) == roster.subject_id.nunique() == 313
    if args.smoke:
        roster = roster.head(3)
    roster[['subject_id']].to_csv(out / 'participant_roster.csv', index=False)

    pair_path = source / 'pairs/figure4_geodesic_pairs_all.parquet'
    cols = KEY + ['anchor_region', BIN, 'distance_bin', 'neighbor_vertex', 'pair_type']
    pairs = pd.read_parquet(pair_path, columns=cols)
    assert pairs.pair_type.eq('within_region').all()
    assert not pairs.duplicated(KEY+[BIN, 'neighbor_vertex']).any()
    counts = pairs.groupby(KEY+['anchor_region', BIN, 'distance_bin'], observed=True).size().rename('n_pairs').reset_index()
    anchors = counts[KEY+['anchor_region']].drop_duplicates()
    assert len(anchors) == 18742 and not anchors.duplicated(KEY).any()
    assignment = pd.read_csv(source / 'pair_qc/formal_fixed_stratified_split_assignment.csv')
    assert set(map(tuple, assignment[KEY].to_numpy())) == set(map(tuple, anchors[KEY].to_numpy()))
    assert counts.n_pairs.between(1, 15).all()
    assert set(counts[BIN]) == set(range(15))
    bins = pd.DataFrame({BIN: range(15)})
    grid = anchors.merge(bins, how='cross').merge(counts.drop(columns='distance_bin'),
        on=KEY+['anchor_region', BIN], how='left', validate='one_to_one')
    grid.n_pairs = grid.n_pairs.fillna(0).astype(int)
    grid['distance_bin'] = grid[BIN].map(lambda b: f'({2*b},{2*b+2}]')
    grid.to_parquet(out / 'template_anchor_bin_pair_counts.parquet', index=False)
    rows = []
    for b, t in grid.groupby(BIN):
        k = t.n_pairs
        rows.append({BIN: b, 'distance_bin': f'({2*b},{2*b+2}]', 'n_anchors_total':18742,
                     'n_active_anchors':int(k.gt(0).sum()), 'active_anchor_fraction':float(k.gt(0).mean()),
                     'n_pairs':int(k.sum()), 'pair_count_mean':k.mean(), 'pair_count_median':k.median(),
                     'pair_count_q25':k.quantile(.25), 'pair_count_q75':k.quantile(.75),
                     'fraction_0':k.eq(0).mean(), 'fraction_1_2':k.between(1,2).mean(),
                     'fraction_3_14':k.between(3,14).mean(), 'fraction_15':k.eq(15).mean()})
    coverage = pd.DataFrame(rows)
    coverage.to_csv(out / 'template_coverage_by_bin.csv', index=False)
    active = grid.assign(active=grid.n_pairs.gt(0))
    dk = active.groupby([BIN, 'anchor_region']).active.sum().rename('n_active_anchors').reset_index()
    dk['fraction_of_active_anchors'] = dk.n_active_anchors / dk.groupby(BIN).n_active_anchors.transform('sum')
    dk.to_csv(out / 'template_dk_composition_by_bin.csv', index=False)
    hemi_dk = active.groupby([BIN, 'hemisphere', 'anchor_region']).active.sum().rename('n_active_anchors').reset_index()
    hemi_dk['fraction_of_all_active_anchors'] = hemi_dk.n_active_anchors / hemi_dk.groupby(BIN).n_active_anchors.transform('sum')
    hemi_dk.to_csv(out / 'template_hemisphere_dk_composition_by_bin.csv', index=False)

    common_keys = active.groupby(KEY).active.all()
    common_keys = common_keys[common_keys].reset_index()[KEY]
    common = anchors.merge(common_keys, on=KEY, validate='one_to_one')
    assert len(common) > 0
    common.to_csv(out / 'template_common_anchors.csv', index=False)
    dk_common = anchors.groupby('anchor_region').size().rename('n_all_anchors').to_frame()
    dk_common['n_common_anchors'] = common.groupby('anchor_region').size().reindex(dk_common.index, fill_value=0)
    dk_common['retention_fraction'] = dk_common.n_common_anchors / dk_common.n_all_anchors
    dk_common['all_anchor_share'] = dk_common.n_all_anchors / len(anchors)
    dk_common['common_anchor_share'] = dk_common.n_common_anchors / len(common)
    dk_common.to_csv(out / 'template_common_anchor_dk_composition.csv')
    tv = .5*float((dk_common.common_anchor_share-dk_common.all_anchor_share).abs().sum())
    common_index = pd.MultiIndex.from_frame(common[KEY])
    template_index = pd.MultiIndex.from_frame(counts[KEY+[BIN]])
    expected_counts = counts.set_index(KEY+[BIN]).n_pairs.sort_index()
    del pairs

    originals, commons, fit_rows, qc_rows = [], [], [], []
    max_curve_error = 0.
    max_fit_error = 0.
    for index, sid in enumerate(roster.subject_id):
        cache_path = source / 'region_real/subjects' / sid / 'figure4_region_real_anchor_bins.parquet'
        cache = pd.read_parquet(cache_path)
        assert cache.subject_id.eq(sid).all()
        assert len(cache) == len(counts)
        assert not cache.duplicated(KEY+[BIN]).any()
        actual_index = pd.MultiIndex.from_frame(cache[KEY+[BIN]])
        assert set(actual_index) == set(template_index)
        fixed = cache.set_index(KEY+[BIN]).fixed_sampled_k.sort_index()
        assert np.array_equal(fixed.to_numpy(), expected_counts.to_numpy())
        assert (cache.k_effective >= 0).all() and (cache.k_effective <= cache.fixed_sampled_k).all()
        valid = cache.loc[cache.has_valid_pair].copy()
        assert cache.has_valid_pair.eq(cache.k_effective.gt(0)).all()
        assert np.isfinite(valid.median_dissimilarity).all()
        valid = valid.rename(columns={'k_effective':'n_pairs'})
        valid['pair_type'] = 'within_region'
        reconstructed = aggregate_subject_bins(valid)
        original = pd.read_csv(source / 'real/subjects' / sid / 'figure4_ico5_k15_real_subject_bins.csv')
        check = reconstructed.merge(original, on=[BIN,'distance_bin','pair_type'], suffixes=('_cache','_original'), validate='one_to_one')
        assert len(check) == 15
        err = float(np.max(np.abs(check.median_dissimilarity_cache-check.median_dissimilarity_original)))
        max_curve_error = max(max_curve_error, err)
        assert err < 1e-7
        assert np.array_equal(check.n_anchors_cache, check.n_anchors_original)
        assert np.array_equal(check.n_pairs_cache, check.n_pairs_original)
        # Exact same pairs: filtering at anchor level leaves each anchor-bin median unchanged.
        selected = valid.loc[pd.MultiIndex.from_frame(valid[KEY]).isin(common_index)]
        restricted = aggregate_subject_bins(selected)
        restricted = bins.merge(restricted, on=BIN, how='left')
        restricted['subject_id'] = sid
        original['subject_id'] = sid
        originals.append(original)
        commons.append(restricted)
        old_fits = pd.read_csv(source / 'correlation_length/subjects' / sid / 'figure4C_correlation_length_fits.csv')
        old_fit = old_fits.loc[old_fits.fit_scope.eq('full')].iloc[0].to_dict()
        reproduced_fit = fit_curve(reconstructed, min_fit_r2=.5)
        assert bool(old_fit['fit_acceptable']) == reproduced_fit['fit_acceptable']
        if bool(old_fit['fit_success']) and reproduced_fit['fit_success']:
            fit_error = abs(float(old_fit['lambda_mm'])-reproduced_fit['lambda_mm'])
            max_fit_error = max(max_fit_error, fit_error)
            assert fit_error < 1e-3
        for name, fit in [('original',old_fit),('common',fit_curve(restricted, min_fit_r2=.5))]:
            fit_rows.append({'subject_id':sid,'analysis':name,**{k:v for k,v in fit.items() if k not in ['subject_id','fit_scope']}})
        q = original[['subject_id',BIN,'n_anchors','n_pairs']].copy()
        q['template_active_anchors'] = coverage.set_index(BIN).n_active_anchors.reindex(q[BIN]).to_numpy()
        q['template_pairs'] = coverage.set_index(BIN).n_pairs.reindex(q[BIN]).to_numpy()
        q['lost_anchors_after_profile_mask'] = q.template_active_anchors-q.n_anchors
        q['lost_pairs_after_profile_mask'] = q.template_pairs-q.n_pairs
        qc_rows.append(q)
        if index % 25 == 0 or index+1 == len(roster):
            print(f'{index+1}/{len(roster)} subjects complete', flush=True)

    original = pd.concat(originals, ignore_index=True)
    restricted = pd.concat(commons, ignore_index=True)
    fits = pd.DataFrame(fit_rows)
    individual = pd.concat(qc_rows, ignore_index=True)
    individual.to_csv(out / 'individual_validity_by_subject_bin.csv', index=False)
    qc_group = individual.groupby(BIN).agg(**{
        f'{col}_{stat}':(col,func) for col in ['n_anchors','n_pairs','lost_anchors_after_profile_mask','lost_pairs_after_profile_mask']
        for stat,func in [('median','median'),('q25',lambda v:v.quantile(.25)),('q75',lambda v:v.quantile(.75))]})
    qc_group.to_csv(out / 'individual_validity_group_summary.csv')
    original.to_csv(out / 'original_subject_curves.csv', index=False)
    restricted.to_csv(out / 'common_anchor_subject_curves.csv', index=False)
    fits.to_csv(out / 'lambda_fits.csv', index=False)
    paired = original[['subject_id',BIN,'median_dissimilarity']].merge(
        restricted[['subject_id',BIN,'median_dissimilarity']], on=['subject_id',BIN], suffixes=('_original','_common'), validate='one_to_one')
    paired['delta_D'] = paired.median_dissimilarity_common-paired.median_dissimilarity_original
    paired.to_csv(out / 'paired_delta_D_by_subject_bin.csv', index=False)
    delta_group = paired.groupby(BIN).delta_D.agg(n='count',median='median',q25=lambda v:v.quantile(.25),q75=lambda v:v.quantile(.75))
    delta_group.to_csv(out / 'paired_delta_D_group_summary.csv')
    group_parts=[]
    for name,frame in [('original',original),('common',restricted)]:
        g=frame.groupby(BIN).median_dissimilarity.agg(n='count',median='median',q25=lambda v:v.quantile(.25),q75=lambda v:v.quantile(.75)).reset_index()
        g['analysis']=name
        group_parts.append(g)
    group=pd.concat(group_parts,ignore_index=True)
    group.to_csv(out / 'group_curves.csv',index=False)
    accepted = fits.loc[fits.fit_acceptable]
    fit_summary=[]
    for name in ['original','common']:
        f=accepted.loc[accepted.analysis.eq(name)]
        fit_summary.append({'analysis':name,'n_total':len(roster),'n_fit_success':int(fits.loc[fits.analysis.eq(name),'fit_success'].sum()),**summary(f.lambda_mm)})
    pd.DataFrame(fit_summary).to_csv(out / 'lambda_group_summary.csv',index=False)
    paired_fits=accepted.pivot(index='subject_id',columns='analysis',values='lambda_mm').dropna(subset=['original','common'])
    paired_fits['delta_lambda_mm']=paired_fits['common']-paired_fits['original']
    paired_fits.to_csv(out / 'paired_lambda_both_pass.csv')
    values=paired_fits.delta_lambda_mm.to_numpy()
    if not len(values):
        raise RuntimeError('No subjects pass both fits')
    rng=np.random.default_rng(20261119)
    boot=np.median(values[rng.integers(0,len(values),(10000,len(values)))],axis=1)
    ci=np.quantile(boot,[.025,.975])
    report={'status':'complete','smoke':args.smoke,'n_subjects':len(roster),'n_template_anchors':18742,
            'n_common_anchors':len(common),'common_anchor_retention_fraction':len(common)/18742,
            'n_dk_regions':len(dk_common),'n_common_dk_regions':int(dk_common.n_common_anchors.gt(0).sum()),
            'dk_composition_total_variation_distance':tv,
            'top5_dk_share_all':float(dk_common.all_anchor_share.nlargest(5).sum()),
            'top5_dk_share_common':float(dk_common.common_anchor_share.nlargest(5).sum()),
            'maximum_reconstructed_original_curve_error':max_curve_error,
            'maximum_reconstructed_original_lambda_error_mm':max_fit_error,
            'lambda_summaries':fit_summary,'paired_delta_lambda_mm':summary(values),
            'delta_lambda_bootstrap_ci95':[float(ci[0]),float(ci[1])],
            'bootstrap_replicates':10000,'bootstrap_seed':20261119,
            'pair_file':str(pair_path),'pair_sha256':hashlib.sha256(pair_path.read_bytes()).hexdigest(),
            'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'cache_validity':'subject-specific masks already applied by formal pair_metrics; exact subject pair/anchor counts verified against primary curves',
            'aggregation':'pair -> anchor-bin median -> bilateral subject median -> group median/IQR',
            'common_definition':'all 15 bins have at least one fixed template pair; individual masks retained; no imputation or replacement',
            'plot_distance_mm':'2*bin_index+1; formal bin centers',
            'fit_distance_mm':'subject-specific median_geodesic_distance_mm; unchanged formal fit_curve',
            'fit_qc':'at least 6 valid bins; R2>=0.5; no parameter boundary; lambda bounds [0.1,100] mm',
            'elapsed_seconds':time.time()-started}
    (out / 'run_report.json').write_text(json.dumps(report,indent=2)+'\n')
    plot_results(out,group,delta_group,coverage,dk_common)
    print(json.dumps(report,indent=2),flush=True)


def plot_results(out,group,delta,coverage,dk):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for name,label,color in [('original','Original','#4B74B2'),('common','Common anchors','#FC8C5A')]:
        g=group.loc[group.analysis.eq(name)].sort_values(BIN)
        x=g[BIN]*2+1
        axes[0].plot(x,g['median'],color=color,label=label)
        axes[0].fill_between(x,g.q25,g.q75,color=color,alpha=.15)
    axes[0].set(xlabel='White-surface geodesic distance (mm)',ylabel='Profile dissimilarity (1 − r)')
    axes[0].legend(frameon=False,loc='lower right')
    x=delta.index.to_numpy()*2+1
    axes[1].plot(x,delta['median'],color='#4B74B2')
    axes[1].fill_between(x,delta.q25,delta.q75,color='#4B74B2',alpha=.2)
    axes[1].axhline(0,color='.5',ls='--',lw=1)
    axes[1].set(xlabel='White-surface geodesic distance (mm)',ylabel='Paired ΔD (common − original)')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    for ext in ['png','pdf']:fig.savefig(out/f'common_anchor_curve_comparison.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    axes[0].plot(coverage[BIN]*2+1,100*coverage.active_anchor_fraction,'o-',color='#4B74B2')
    axes[0].set(xlabel='White-surface geodesic distance (mm)',ylabel='Template anchors with ≥1 pair (%)',ylim=(0,101))
    dk=dk.sort_values('all_anchor_share',ascending=False)
    x=np.arange(len(dk))
    axes[1].bar(x-.2,100*dk.all_anchor_share,width=.4,color='#4B74B2',label='All anchors')
    axes[1].bar(x+.2,100*dk.common_anchor_share,width=.4,color='#FC8C5A',label='Common anchors')
    axes[1].set(xticks=x,xticklabels=dk.index,ylabel='DK composition (%)')
    axes[1].tick_params(axis='x',rotation=90,labelsize=6)
    axes[1].legend(frameon=False)
    for ext in ['png','pdf']:fig.savefig(out/f'template_coverage_and_dk_composition.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main()
