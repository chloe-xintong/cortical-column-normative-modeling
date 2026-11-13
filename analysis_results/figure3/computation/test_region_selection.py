"""Thickness-only quantile selection and subject-first bilateral means.

Description
-----------
Thickness-only quantile selection and subject-first bilateral means.

Methods
-------
1. Set up the test inputs.
2. Execute the target behavior.
3. Assert the expected results.

Inputs
------
- No required command-line inputs; optional inputs are defined by the script.

Outputs
-------
- Test pass/fail status; no analysis result files are generated.

Usage
-----
pytest analysis_results/figure3/computation/test_region_selection.py
"""
import unittest
import numpy as np
import pandas as pd
from select_representative_regions import (rank_and_select, bilateral_subject_profiles,
                                           select_extreme_regions, COLORS, EXTREME_COLORS)


class SelectionTests(unittest.TestCase):
    def test_mean_ci95_uses_subject_level_t_interval(self):
        from figure3_panel_a_plot import mean_ci95_halfwidth
        from scipy.stats import t
        sd=np.array([2.,3.])
        n=np.array([10,25])
        expected=t.ppf(.975,n-1)*sd/np.sqrt(n)
        np.testing.assert_allclose(mean_ci95_halfwidth(sd,n),expected)

    def test_linear_quantiles_and_stable_median_tie(self):
        frame=pd.DataFrame({'region_name':[f'r{i:02d}' for i in range(34)],
                            'group_mean_thickness':np.arange(34,dtype=float)})
        ranked=rank_and_select(frame.sample(frac=1,random_state=4))
        selected=ranked.loc[ranked.selected_for_figure3].sort_values('target_percentile')
        self.assertEqual(selected.thickness_rank.tolist(),[4,17,31])
        self.assertEqual(selected.selected_level.tolist(),['thin','intermediate','thick'])
        self.assertFalse(ranked.iloc[[0,-1]].selected_for_figure3.any())
        np.testing.assert_allclose(selected.thickness_percentile,[100*3/33,100*16/33,100*30/33])

    def test_bilateral_equal_hemisphere_weights_missing_not_filled(self):
        table=pd.DataFrame({'subject_id':['s1','s1','s2','s2'], 'region':['r']*4,
            'depth_index':[0]*4,'normalized_depth':[0.]*4,'hemisphere':['lh','rh','lh','rh'],
            'md':[1.,3.,5.,np.nan], 'n_valid_vertices':[10000,1,1,10000]})
        result=bilateral_subject_profiles(table).set_index('subject_id')
        self.assertEqual(result.loc['s1','md'],2.)
        self.assertTrue(np.isnan(result.loc['s2','md']))

    def test_extreme_selection_uses_true_endpoints_and_lower_median_tie(self):
        frame=pd.DataFrame({'region_name':[f'r{i:02d}' for i in range(34)],
                            'group_mean_thickness':np.arange(34,dtype=float)})
        ranking=rank_and_select(frame.sample(frac=1,random_state=8))
        selected=select_extreme_regions(ranking)
        self.assertEqual(selected.region_name.tolist(),['r00','r16','r33'])
        self.assertEqual(selected.extreme_level.tolist(),['minimum','median','maximum'])
        self.assertEqual(selected.extreme_color.tolist(),list(EXTREME_COLORS))
        self.assertEqual(selected.thickness_rank.tolist(),[1,17,34])
        self.assertEqual(tuple(COLORS),tuple(EXTREME_COLORS))

    def test_shared_axis_profile_titles_have_finite_renderer_bounds(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from figure3_profile_plot import draw_raw_adjusted_profiles
        ranking=rank_and_select(pd.DataFrame({'region_name':[f'r{i:02d}' for i in range(34)],
                                               'group_mean_thickness':np.linspace(1.5,3.5,34)}))
        selected=ranking.loc[ranking.selected_for_figure3].sort_values('target_percentile')
        rows=[dict(region=name,depth_index=d,normalized_depth=d/20,mean_md=.001-.0002*d/20,
                    sd_md=.00005,n_subjects=320) for name in selected.region_name for d in range(21)]
        data=pd.DataFrame(rows)
        fig,axes=draw_raw_adjusted_profiles(data,data,320,selected,ranking)
        fig.canvas.draw()
        for ax in axes:
            self.assertTrue(np.isfinite(ax.title.get_position()).all())
            self.assertTrue(np.isfinite(ax.title.get_window_extent(fig.canvas.get_renderer()).bounds).all())
        plt.close(fig)


if __name__=='__main__':unittest.main()
