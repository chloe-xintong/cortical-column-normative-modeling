import json
from pathlib import Path

path = Path('validation/profile_stability/depth_stability_analysis.ipynb')
notebook = json.loads(path.read_text())
markers = ['Overall CV-depth curve', "overall = depth.groupby('depth')", 'Mean, SD, and CV across depth', 'overall_stats =', 'Pial-depth exclusion sensitivity', 'pial_sensitivity', 'Depth 0–17 CV change', 'cv_delta_0_17']
notebook['cells'] = [cell for cell in notebook['cells'] if not any(marker in ''.join(cell.get('source', [])) for marker in markers)]
notebook['cells'].extend([
    {
        'cell_type': 'markdown', 'metadata': {},
        'source': ['## Overall CV-depth curve\n', '\n', 'At each depth, the line is the median CV across all subject × hemisphere × region observations. The shaded interval is the interquartile range. Depth 0 is the white surface and depth 20 is the pial surface.\n'],
    },
    {
        'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
        'source': [
            "overall = depth.groupby('depth')['cv_pct'].agg(q25=lambda x: x.quantile(.25), median='median', q75=lambda x: x.quantile(.75)).reset_index()\n",
            "fig, ax = plt.subplots(figsize=(9, 5))\n",
            "ax.plot(overall['depth'], overall['median'], marker='o', label='Median CV')\n",
            "ax.fill_between(overall['depth'], overall['q25'], overall['q75'], alpha=.25, label='IQR')\n",
            "ax.set(title='Overall CV-depth curve', xlabel='Depth (0 = white surface, 20 = pial surface)', ylabel='CV (%)')\n",
            "ax.set_xticks(range(21)); ax.grid(alpha=.25); ax.legend()\n",
            "fig.tight_layout(); fig.savefig(ROOT/'overall_cv_depth_curve.png', dpi=180); plt.show()\n",
        ],
    },
    {
        'cell_type': 'markdown', 'metadata': {},
        'source': ['## Mean, SD, and CV across depth\n', '\n', 'Each curve is the median across subject × hemisphere × region observations. The three panels retain their own y-axis because MD mean, SD, and CV have different units.\n'],
    },
    {
        'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
        'source': [
            "overall_stats = depth.groupby('depth', as_index=False).agg(mean_md=('mean_md','median'), sd_md=('sd_md','median'), cv_pct=('cv_pct','median'))\n",
            "fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)\n",
            "for ax, column, title, ylabel in zip(axes, ['mean_md','sd_md','cv_pct'], ['Mean MD by depth','MD SD by depth','MD CV by depth'], ['Median mean MD','Median SD','Median CV (%)']):\n",
            "    ax.plot(overall_stats['depth'], overall_stats[column], marker='o')\n",
            "    ax.set(title=title, xlabel='Depth (0 = white, 20 = pial)', ylabel=ylabel); ax.set_xticks(range(0,21,2)); ax.grid(alpha=.25)\n",
            "fig.tight_layout(); fig.savefig(ROOT/'overall_mean_sd_cv_depth_curves.png', dpi=180); plt.show()\n",
        ],
    },
    {
        'cell_type': 'markdown', 'metadata': {},
        'source': ['## Pial-depth exclusion sensitivity\n', '\n', 'This tests whether the CV-depth trend is driven mainly by the pial endpoint. For each subject × hemisphere × region, Spearman correlation is recalculated after removing 0, 1, 2, or 3 pial-side depth points.\n'],
    },
    {
        'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
        'source': [
            "rows = []\n",
            "for removed in range(4):\n",
            "    subset = depth[depth['depth'] <= 20 - removed]\n",
            "    rho = subset.groupby(['subject','hemisphere','region']).apply(lambda g: g['depth'].corr(g['cv_pct'], method='spearman'), include_groups=False).rename('rho').reset_index()\n",
            "    rows.append({'pial_depths_removed': removed, 'median_rho': rho['rho'].median(), 'q25_rho': rho['rho'].quantile(.25), 'q75_rho': rho['rho'].quantile(.75), 'positive_rho_pct': 100*(rho['rho'] > 0).mean(), 'n_subject_region': len(rho)})\n",
            "pial_sensitivity = pd.DataFrame(rows)\n",
            "pial_sensitivity.to_csv(ROOT/'pial_depth_exclusion_sensitivity.csv', index=False)\n",
            "display(pial_sensitivity)\n",
            "fig, ax = plt.subplots(figsize=(8, 5))\n",
            "ax.plot(pial_sensitivity['pial_depths_removed'], pial_sensitivity['median_rho'], marker='o', label='Median Spearman rho')\n",
            "ax.fill_between(pial_sensitivity['pial_depths_removed'], pial_sensitivity['q25_rho'], pial_sensitivity['q75_rho'], alpha=.25, label='IQR')\n",
            "ax.axhline(0, color='black', lw=1); ax.set(title='CV-depth trend after pial-depth exclusion', xlabel='Number of pial-side depths removed', ylabel='CV-depth Spearman rho', xticks=range(4))\n",
            "ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(ROOT/'pial_depth_exclusion_sensitivity.png', dpi=180); plt.show()\n",
        ],
    },
    {
        'cell_type': 'markdown', 'metadata': {},
        'source': ['## Depth 0–17 CV change\n', '\n', 'For each subject × hemisphere × region, this compares mean CV at depths 0–2 with mean CV at depths 15–17. Depths 18–20 are excluded.\n'],
    },
    {
        'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
        'source': [
            "low = depth[depth['depth'].between(0, 2)].groupby(['subject','hemisphere','region'], as_index=False)['cv_pct'].mean().rename(columns={'cv_pct':'mean_cv_depth_0_2'})\n",
            "high = depth[depth['depth'].between(15, 17)].groupby(['subject','hemisphere','region'], as_index=False)['cv_pct'].mean().rename(columns={'cv_pct':'mean_cv_depth_15_17'})\n",
            "cv_delta_0_17 = low.merge(high, on=['subject','hemisphere','region'])\n",
            "cv_delta_0_17['delta_cv_15_17_minus_0_2'] = cv_delta_0_17['mean_cv_depth_15_17'] - cv_delta_0_17['mean_cv_depth_0_2']\n",
            "cv_delta_0_17.to_csv(ROOT/'cv_delta_depth_0_17_by_subject_region.csv', index=False)\n",
            "delta_summary = pd.DataFrame([{'n_subject_region':len(cv_delta_0_17), 'median_delta_cv_pct':cv_delta_0_17['delta_cv_15_17_minus_0_2'].median(), 'q25_delta_cv_pct':cv_delta_0_17['delta_cv_15_17_minus_0_2'].quantile(.25), 'q75_delta_cv_pct':cv_delta_0_17['delta_cv_15_17_minus_0_2'].quantile(.75), 'positive_delta_pct':100*(cv_delta_0_17['delta_cv_15_17_minus_0_2']>0).mean()}])\n",
            "display(delta_summary)\n",
            "fig, ax = plt.subplots(figsize=(8, 5))\n",
            "ax.hist(cv_delta_0_17['delta_cv_15_17_minus_0_2'], bins=50, edgecolor='white')\n",
            "ax.axvline(0, color='black', lw=1); ax.set(title='CV change: depths 15–17 minus depths 0–2', xlabel='Delta CV (%)', ylabel='Subject × hemisphere × region count')\n",
            "fig.tight_layout(); fig.savefig(ROOT/'cv_delta_depth_0_17_distribution.png', dpi=180); plt.show()\n",
        ],
    },
])
notebook['metadata']['kernelspec'] = {'display_name':'Python (cortical_trajectory_analysis)','language':'python','name':'cortical_trajectory_analysis'}
path.write_text(json.dumps(notebook, indent=1))
print('updated', path)
