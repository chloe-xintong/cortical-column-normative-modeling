"""Reference OLS and weighting checks for scientific analysis kernels.

Description
-----------
Reference OLS and weighting checks for scientific analysis kernels.

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
pytest analysis_results/figure3/computation/test_figure3_statistics.py
"""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from figure3_statistics import (center_profiles, weighted_pca, thickness_effect, regional_partial_r2,
                                regional_sufficient_statistics, regional_effects_from_statistics)


class StatisticsTests(unittest.TestCase):
    def test_unbalanced_raw_adjusted_and_bootstrap_equal_explicit_ols(self):
        rng=np.random.default_rng(5)
        y=rng.normal(size=(7,5,21));t=rng.normal(size=(7,5))
        y[0,2]=np.nan;t[0,2]=np.nan;y[3,0]=np.nan;t[3,0]=np.nan
        stats=regional_sufficient_statistics(y,t)
        weights=np.array([[1,1,1,1,1,1,1],[2,0,1,1,0,2,1]])
        beta=rng.normal(size=(2,21))
        raw,adj=regional_effects_from_statistics(stats,weights,beta)
        for b,w in enumerate(weights):
            indices=np.repeat(np.arange(7),w)
            for adjusted,observed in [(False,raw[b]),(True,adj[b])]:
                yy=y[indices].copy();tt=t[indices]
                if adjusted:yy-=beta[b]*(tt-np.nanmean(tt,axis=1)[:,None])[:,:,None]
                valid=np.isfinite(yy).all(axis=2)
                n,r,d=yy.shape
                subject=np.repeat(np.eye(n),r,axis=0)[valid.reshape(-1)]
                region=np.tile(np.eye(r),(n,1))[valid.reshape(-1)]
                flat=yy.reshape(-1,d)[valid.reshape(-1)]
                residual0=flat-subject@np.linalg.lstsq(subject,flat,rcond=None)[0]
                full=np.column_stack([subject,region]);residual1=flat-full@np.linalg.lstsq(full,flat,rcond=None)[0]
                expected=1-(residual1**2).sum(axis=0)/(residual0**2).sum(axis=0)
                assert_allclose(observed,expected,atol=1e-12)
        assert_allclose(regional_partial_r2(y,weights),raw,atol=1e-12)

    def test_profile_centering_removes_offsets_not_amplitude(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(30,21))
        y=center_profiles(x)
        assert_allclose(y.mean(axis=1),0,atol=1e-15)
        assert_allclose(center_profiles(x+np.arange(30)[:,None]),y,atol=1e-14)
        assert_allclose(center_profiles(2*x),2*y)

    def test_equal_subject_covariance_and_sign(self):
        rng=np.random.default_rng(2)
        samples=[center_profiles(rng.normal(size=(n,21))*s) for n,s in [(10,1),(100,3)]]
        means=np.array([x.mean(axis=0) for x in samples]);moments=np.array([x.T@x/len(x) for x in samples])
        mean,e,w,_,correlation=weighted_pca(means,moments)
        expected=sum(x.T@x/len(x) for x in samples)/2-np.outer(mean,mean)
        assert_allclose(expected,w@np.diag(e)@w.T,atol=1e-12)
        self.assertGreaterEqual(correlation,0)
        assert_allclose(w[:,0].sum(),0,atol=1e-12)

    def test_thickness_effect_equals_subject_fixed_effect_ols(self):
        rng=np.random.default_rng(3);tt=[];ty=[];yy=[];design=[];outcomes=[]
        for s,n in enumerate([10,17,30,21]):
            t=rng.normal(size=n);y=rng.normal(size=(n,21))+t[:,None]*np.linspace(-1,1,21)+s
            tc=t-t.mean();yc=y-y.mean(axis=0)
            tt.append(tc@tc);ty.append(tc@yc);yy.append((yc*yc).sum(axis=0))
            dummy=np.zeros((n,4));dummy[:,s]=1;design.append(np.column_stack([dummy,t]));outcomes.append(y)
        beta,_,r2=thickness_effect(np.array(tt),np.array(ty),np.array(yy))
        expected=np.linalg.lstsq(np.vstack(design),np.vstack(outcomes),rcond=None)[0][-1]
        assert_allclose(beta[0],expected,atol=1e-12)
        self.assertTrue(np.all((r2>=0)&(r2<=1)))

    def test_regional_r2_equals_dummy_variable_ols_and_bootstrap(self):
        rng=np.random.default_rng(4);y=rng.normal(size=(7,5,21))+np.arange(5)[None,:,None]*.3
        for weights in [np.ones(7,int),np.array([2,0,1,1,0,2,1])]:
            expanded=y[np.repeat(np.arange(7),weights)];s,r,d=expanded.shape
            subject=np.repeat(np.eye(s),r,axis=0);region=np.tile(np.eye(r),(s,1))
            flat=expanded.reshape(-1,d)
            reduced=flat-subject@np.linalg.lstsq(subject,flat,rcond=None)[0]
            full=np.column_stack([subject,region]);residual=flat-full@np.linalg.lstsq(full,flat,rcond=None)[0]
            expected=1-(residual**2).sum(axis=0)/(reduced**2).sum(axis=0)
            assert_allclose(regional_partial_r2(y,weights)[0],expected,atol=1e-12)


if __name__=='__main__':unittest.main()
