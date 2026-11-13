"""Test Curvature Supplement

Description
-----------
Test Curvature Supplement

Methods
-------
1. Set up the test inputs.
2. Execute the target behavior.
3. Assert the expected results.

Inputs
------
- Data supplied through the module's public function arguments.

Outputs
-------
- Test pass/fail status; no analysis result files are generated.

Usage
-----
pytest analysis_results/figure3/computation/test_curvature_supplement.py
"""

import numpy as np

from compute_curvature_supplement import multivariable_effect, pearson


def test_multivariable_effect_matches_explicit_ols():
    rng=np.random.default_rng(4);n=2000
    thickness=rng.normal(size=n)
    curvature=.35*thickness+rng.normal(size=n)
    score=.5*thickness-.3*curvature+rng.normal(scale=.7,size=n)
    result=multivariable_effect(score,thickness,curvature)
    design=np.column_stack([np.ones(n),thickness,curvature])
    beta=np.linalg.lstsq(design,score,rcond=None)[0]
    assert np.allclose([result['raw_beta_thickness'],result['raw_beta_curvature']],
                       beta[1:],rtol=1e-12,atol=1e-12)


def test_partial_r2_matches_reduced_and_full_ols():
    rng=np.random.default_rng(8);n=700
    thickness=rng.normal(size=n);curvature=rng.normal(size=n)
    score=.2*thickness+.4*curvature+rng.normal(size=n)
    result=multivariable_effect(score,thickness,curvature)
    reduced=np.column_stack([np.ones(n),thickness])
    full=np.column_stack([reduced,curvature])
    red_res=score-reduced@np.linalg.lstsq(reduced,score,rcond=None)[0]
    full_res=score-full@np.linalg.lstsq(full,score,rcond=None)[0]
    expected=(red_res@red_res-full_res@full_res)/(red_res@red_res)
    assert np.isclose(result['partial_r2_curvature'],expected,rtol=1e-12,atol=1e-12)


def test_pearson_is_centered_and_offset_invariant():
    rng=np.random.default_rng(9);a=rng.normal(size=300);b=.4*a+rng.normal(size=300)
    assert np.isclose(pearson(a,b),np.corrcoef(a,b)[0,1])
    assert np.isclose(pearson(a+100,b-70),pearson(a,b))
