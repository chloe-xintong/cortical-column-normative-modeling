import json
from pathlib import Path

p = Path('validation/column_badpoint_pattern_analysis.ipynb')
nb = json.loads(p.read_text())

nb['cells'].extend([
    {
        'cell_type': 'markdown', 'metadata': {}, 'source': [
            '## 6. Column 保留率与插值验证\n\n',
            '本部分直接读取 `column_retention_interpolation.py` 生成的结果，不重新读取原始 profile。\n\n',
            '保留率定义为：允许每条 column 中最多有 0、1、2 或 3 个坏点时，仍然可以保留的 column 比例。插值验证使用原本没有坏点的 column，人为删除 1 个或 2 个内部 depth，再与真实值比较。'
        ]
    },
    {
        'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
            "retention_path = QC_ROOT / 'subject_column_retention_curve.csv'\n",
            "population_path = QC_ROOT / 'population_column_retention_summary.csv'\n",
            "interp_path = QC_ROOT / 'interpolation_validation_summary.csv'\n",
            "for required in (retention_path, population_path, interp_path):\n",
            "    if not required.is_file():\n",
            "        raise FileNotFoundError(f'请先运行 column_retention_interpolation.py: {required}')\n",
            "retention = pd.read_csv(retention_path)\n",
            "population_retention = pd.read_csv(population_path)\n",
            "interpolation = pd.read_csv(interp_path)\n",
            "print('已读取 retention 和 interpolation 结果。')\n",
            "display(population_retention)\n",
            "display(interpolation)\n"
        ]
    },
    {
        'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [
            "import matplotlib.pyplot as plt\n",
            "plt.close('all')\n",
            "allowed = [0, 1, 2, 3]\n",
            "median = [population_retention.loc[population_retention.allowed_bad_points.eq(k), 'median_retention_pct'].iloc[0] for k in allowed]\n",
            "q1 = [population_retention.loc[population_retention.allowed_bad_points.eq(k), 'q1_retention_pct'].iloc[0] for k in allowed]\n",
            "q3 = [population_retention.loc[population_retention.allowed_bad_points.eq(k), 'q3_retention_pct'].iloc[0] for k in allowed]\n",
            "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
            "axes[0,0].plot(allowed, median, marker='o', linewidth=2, label='Median')\n",
            "axes[0,0].fill_between(allowed, q1, q3, alpha=.25, label='IQR')\n",
            "axes[0,0].set(title='Population column retention curve', xlabel='允许的最多坏点数', ylabel='Retention (%)', xticks=allowed)\n",
            "axes[0,0].set_ylim(0, 100); axes[0,0].grid(alpha=.25); axes[0,0].legend()\n",
            "for k in allowed:\n",
            "    axes[0,1].boxplot(retention[f'retention_allowed_{k}'].dropna(), positions=[k], widths=.55)\n",
            "axes[0,1].set(title='Retention distribution across subjects', xlabel='允许的最多坏点数', ylabel='Retention (%)', xticks=allowed); axes[0,1].set_ylim(0,100); axes[0,1].grid(alpha=.25)\n",
            "outliers = retention.nlargest(10, 'bad_point_pct')\n",
            "for _, row in outliers.iterrows():\n",
            "    axes[1,0].plot(allowed, [row[f'retention_allowed_{k}'] for k in allowed], marker='o', label=row.subject)\n",
            "axes[1,0].set(title='Highest bad-point subjects', xlabel='允许的最多坏点数', ylabel='Retention (%)', xticks=allowed); axes[1,0].set_ylim(0,100); axes[1,0].grid(alpha=.25); axes[1,0].legend(fontsize=8, ncol=2)\n",
            "x = interpolation.artificial_bad_points.astype(str)\n",
            "axes[1,1].bar(x, interpolation.mae_mean, color=['#4c78a8','#f58518'], alpha=.85)\n",
            "axes[1,1].set(title='Linear interpolation validation', xlabel='人为删除的坏点数', ylabel='Mean absolute error (MAE)'); axes[1,1].grid(axis='y', alpha=.25)\n",
            "fig.tight_layout()\n",
            "out = QC_ROOT / 'retention_interpolation_overview.png'\n",
            "fig.savefig(out, dpi=180, bbox_inches='tight')\n",
            "display(fig)\n",
            "print('图像已保存:', out)\n"
        ]
    }
])
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print('updated', p)
