"""Zero-inclusive pair-count summaries for template and individually valid pairs."""
from pathlib import Path
import numpy as np
import pandas as pd


def quantile(hist, probability):
    rank = (hist.sum()-1)*probability
    cumulative = np.cumsum(hist)
    lo = np.searchsorted(cumulative, int(np.floor(rank))+1)
    hi = np.searchsorted(cumulative, int(np.ceil(rank))+1)
    return float(lo+(hi-lo)*(rank-np.floor(rank)))


def describe(hist):
    n = int(hist.sum())
    return {'n_anchor_bins':n, 'mean':float(hist @ np.arange(16)/n),
            'median':quantile(hist,.5), 'q25':quantile(hist,.25),
            'q75':quantile(hist,.75), 'fraction_15':float(hist[15]/n),
            'fraction_0':float(hist[0]/n), 'fraction_1_2':float(hist[1:3].sum()/n)}


def main():
    repo = Path(__file__).resolve().parents[3]
    out = repo/'analysis_results/figure4/audits/common_anchor_sensitivity_313'
    root = Path('/mnt/newStor/paros/paros_WORK/xintong/results/qc/full_cohort_spatial_scale/figure4_ico5_formal_k15')
    grid = pd.read_parquet(out/'template_anchor_bin_pair_counts.parquet')
    template = []
    for b,g in grid.groupby('distance_bin_index'):
        hist = np.bincount(g.n_pairs, minlength=16)
        assert hist.sum() == 18742
        template.append({'distance_bin_index':b,'distance_bin':f'({2*b},{2*b+2}]',**describe(hist)})
    t = pd.DataFrame(template)
    t.to_csv(out/'actual_pair_counts_template.csv',index=False)
    # Also add the missing mean to the original zero-inclusive coverage export.
    coverage = pd.read_csv(out/'template_coverage_by_bin.csv')
    coverage['pair_count_mean'] = coverage.distance_bin_index.map(t.set_index('distance_bin_index')['mean'])
    coverage.to_csv(out/'template_coverage_by_bin.csv',index=False)
    pooled = np.zeros((15,16),dtype=np.int64)
    subject_rows = []
    subjects = pd.read_csv(out/'participant_roster.csv').subject_id
    assert len(subjects) == 313
    for i,sid in enumerate(subjects):
        cache = pd.read_parquet(root/'region_real/subjects'/sid/'figure4_region_real_anchor_bins.parquet',columns=['distance_bin_index','k_effective'])
        for b,g in cache.groupby('distance_bin_index'):
            hist = np.bincount(g.k_effective.to_numpy(dtype=int),minlength=16)
            hist[0] += 18742-len(g)  # Template anchor-bins with no fixed pair.
            assert hist.sum() == 18742
            pooled[b] += hist
            subject_rows.append({'subject_id':sid,'distance_bin_index':b,**describe(hist)})
        if i%50==0:print(f'{i+1}/313',flush=True)
    pd.DataFrame(subject_rows).to_csv(out/'actual_pair_counts_by_subject_bin.csv',index=False)
    p = pd.DataFrame([{'distance_bin_index':b,'distance_bin':f'({2*b},{2*b+2}]',**describe(h)} for b,h in enumerate(pooled)])
    assert p.n_anchor_bins.eq(313*18742).all()
    p.to_csv(out/'actual_pair_counts_individual_pooled.csv',index=False)
    print('TEMPLATE');print(t.to_string(index=False))
    print('INDIVIDUAL VALID PAIRS (pooled subject × anchor observations)');print(p.to_string(index=False))


if __name__=='__main__':
    main()
