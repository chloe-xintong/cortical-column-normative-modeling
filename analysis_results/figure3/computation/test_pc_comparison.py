"""Check paired estimands and resampling with analytically known examples.

Description
-----------
Check paired estimands and resampling with analytically known examples.

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
pytest analysis_results/figure3/computation/test_pc_comparison.py
"""
import unittest
import numpy as np
import pandas as pd
from compare_pc1_pc2_thickness import paired_summary


class PairedSummaryTests(unittest.TestCase):
    def test_median_of_pairs_not_difference_of_medians(self):
        a=np.array([0.,.7,.8]);b=np.array([.2,.1,.9])
        frame=pd.DataFrame(dict(subject_id=['a','b','c'],rho_thickness_pc1=a,
                                rho_thickness_pc2=b,delta_abs_rho=np.abs(b)-np.abs(a)))
        summary,comparison,_=paired_summary(frame)
        self.assertAlmostEqual(comparison['median_paired_difference'],.1)
        self.assertNotAlmostEqual(comparison['median_paired_difference'],np.median(b)-np.median(a))
        self.assertAlmostEqual(comparison['ci95_lower'],-.6)
        self.assertAlmostEqual(comparison['ci95_upper'],.2)
        self.assertAlmostEqual(summary.iloc[0].median_rho,.7)
        self.assertAlmostEqual(summary.iloc[0].rho_iqr,.4)
        self.assertAlmostEqual(comparison['wilcoxon_pvalue'],1.)

    def test_absolute_comparison_invariant_to_component_sign(self):
        a=np.array([.1,.2,.3,.4]);b=np.array([.5,.4,.6,.8])
        def calculate(sign):
            f=pd.DataFrame(dict(subject_id=list('abcd'),rho_thickness_pc1=a,
                                rho_thickness_pc2=sign*b,delta_abs_rho=b-a))
            return paired_summary(f,repetitions=1000)[1]
        x,y=calculate(1),calculate(-1)
        for k in ('median_paired_difference','ci95_lower','ci95_upper','wilcoxon_pvalue'):
            self.assertEqual(x[k],y[k])


if __name__=='__main__':unittest.main()
