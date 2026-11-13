#!/usr/bin/env python3
"""Complete Figure 3: cortical-depth MD shape and thickness confounding.

Description
-----------
All paths are CLI inputs. Preparation streams one subject at a time; the
only full-resolution cohort arrays are disk-backed caches. Statistical
inference resamples subjects, never individual vertices.

Methods
-------
1. Parse and validate the command-line inputs.
2. Execute the processing implemented by this script.
3. Validate and save the configured results.

Inputs
------
- ``--qc-exclusions`` command-line input
- Additional optional, conditional, or mutually exclusive inputs are listed by ``--help``.

Outputs
-------
- Result files written to the output paths configured by the script.

Usage
-----
python analysis_results/figure3/computation/run_all_figure3.py --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from figure3_statistics import (center_profiles, weighted_pca, thickness_effect, regional_partial_r2,
                                regional_sufficient_statistics, regional_effects_from_statistics)

HEMIS = ('lh', 'rh')
DEPTHS = np.linspace(0, 1, 21)
EXCLUDED = {'unknown', 'corpuscallosum', 'medial_wall', 'medialwall', 'background+freesurfer_defined_medial_wall'}


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def paths(args, subject, hemi):
    return (args.profiles_dir / subject / 'profiles' / f'{subject}_{hemi}_md_profiles.npz',
            args.mapping_dir / subject / 'columns' / f'{subject}_{hemi}_columns.npz',
            args.subjects_dir / subject / 'surf' / f'{hemi}.thickness')


def profile_coverage(args, subject):
    """Figure-5-compatible bilateral coverage from cleaned profile metadata."""
    n_mask = 0
    n_valid = 0
    for hemi in HEMIS:
        profile_path = paths(args, subject, hemi)[0]
        with np.load(profile_path, allow_pickle=False) as profile:
            mask = profile['in_analysis_mask'].astype(bool)
            valid = profile['column_valid'].astype(bool)
            if mask.shape != valid.shape:
                raise ValueError(f'{subject} {hemi}: coverage arrays differ in shape')
            n_mask += int(mask.sum())
            n_valid += int((mask & valid).sum())
    if n_mask == 0:
        raise ValueError(f'{subject}: empty bilateral analysis mask')
    return n_mask, n_valid, n_valid / n_mask


def atlas(args):
    rows, offset, region_id = [], 0, 0
    for hemi in HEMIS:
        labels, _, names = fsio.read_annot(args.fsaverage_dir / 'label' / f'{hemi}.aparc.annot')
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in names]
        if len(names)!=len(set(names)):
            raise ValueError(f'Ambiguous exact annotation names: {hemi}')
        remap = {}
        for label_id, name in enumerate(names):
            if name.lower().replace('-', '_').replace(' ', '_') not in EXCLUDED:
                remap[label_id] = region_id
                region_id += 1
        for vertex, label in enumerate(labels):
            name = names[label] if label >= 0 else 'unknown'
            rows.append((offset + vertex, hemi, vertex, int(label), name,
                         label in remap, remap.get(label, -1)))
        offset += len(labels)
    return pd.DataFrame(rows, columns=['global_vertex_id', 'hemisphere', 'fsaverage_vertex_id',
                                      'dk_label_id', 'dk_label_name', 'atlas_mask', 'region_index'])


def load_subject(args, subject, nodes):
    arrays, thick, length, qc = [], [], [], []
    for hemi in HEMIS:
        p, m, t = paths(args, subject, hemi)
        ref = nodes.loc[nodes.hemisphere.eq(hemi)]
        mask = ref.atlas_mask.to_numpy(bool)
        ids = ref.fsaverage_vertex_id.to_numpy(int)
        with np.load(p, allow_pickle=False) as profile, np.load(m, allow_pickle=False) as mapping:
            for key in ('fsaverage_vertex_id', 'target_vertex_id'):
                if not np.array_equal(profile[key], ids) or not np.array_equal(mapping[key], ids):
                    raise ValueError(f'{subject} {hemi}: vertex order mismatch')
            for data in (profile, mapping):
                if str(data['subject_id'].item()) != subject or str(data['hemisphere'].item()) != hemi:
                    raise ValueError('Subject/hemisphere metadata mismatch')
                if str(data['template'].item()) != 'fsaverage':
                    raise ValueError('Figure 3 requires fsaverage profiles')
                if str(data['depth_direction'].item()) != 'pial_to_white' or not np.allclose(data['depth_fractions'], DEPTHS):
                    raise ValueError('Noncanonical depth direction')
                if not np.array_equal(data['in_analysis_mask'].astype(bool), mask):
                    raise ValueError('Subject/template analysis mask mismatch')
            if not np.array_equal(profile['template_label_id'], ref.dk_label_id.to_numpy(int)):
                raise ValueError('DK annotation IDs differ')
            profile_names = profile['template_label_name'].astype(str)
            if not np.array_equal(profile_names[mask], ref.dk_label_name.to_numpy(str)[mask]):
                raise ValueError('Exact DK annotation names differ')
            tri = mapping['native_triangle_vertex_ids'].astype(int)
            weights = mapping['barycentric_weights'].astype(float)
            if not np.array_equal(profile['native_triangle_vertex_ids'], tri) or not np.allclose(profile['barycentric_weights'], weights, equal_nan=True):
                raise ValueError('Profile/mapping triangle correspondence differs')
            native = fsio.read_morph_data(t).astype(float)
            good = (mapping['mapping_valid'].astype(bool) & np.all(tri >= 0, axis=1)
                    & np.all(tri < len(native), axis=1) & np.isfinite(weights).all(axis=1)
                    & np.isclose(weights.sum(axis=1), 1, atol=1e-6, rtol=0)
                    & np.all(weights >= -1e-6, axis=1) & np.all(weights <= 1+1e-6, axis=1))
            thickness = np.full(len(ids), np.nan)
            thickness[good] = np.sum(native[tri[good]] * weights[good], axis=1)
            stored = profile['cortical_thickness'].astype(float)
            if not np.allclose(stored[good], thickness[good], rtol=1e-5, atol=1e-6, equal_nan=True):
                raise ValueError('Recomputed native thickness disagrees with stored thickness')
            points = mapping['column_points']
            endpoint_ok = (np.allclose(points[good, 0], mapping['pial_endpoint'][good], atol=1e-6)
                           and np.allclose(points[good, -1], mapping['white_endpoint'][good], atol=1e-6))
            if not endpoint_ok:
                raise ValueError('Actual column endpoints do not follow pial-to-white direction')
            values = profile['profiles'].astype(float)
            if values.shape != (len(ids), 21):
                raise ValueError('Unexpected MD dimensions')
            md_valid = (profile['column_valid'].astype(bool) & mask & good
                        & np.isfinite(values).all(axis=1) & (values > 0).all(axis=1))
            plausible = np.isfinite(thickness) & (thickness > args.thickness_min) & (thickness <= args.thickness_max)
            valid = md_valid & plausible
            values[~valid] = np.nan
            thickness[~valid] = np.nan
            column_length = profile['column_length'].astype(float)
            column_length[~valid] = np.nan
            arrays.append(values.astype(np.float32)); thick.append(thickness.astype(np.float32))
            length.append(column_length.astype(np.float32))
            qc.append(dict(subject_id=subject, hemisphere=hemi, n_mask_vertices=int(mask.sum()),
                           n_valid_md=int(md_valid.sum()), n_valid=int(valid.sum()),
                           n_bad_thickness=int((md_valid & ~plausible).sum()),
                           n_bad_mapping_in_mask=int((mask & ~good).sum()),
                           thickness_min=float(np.min(thickness[valid])), thickness_max=float(np.max(thickness[valid])),
                           max_weight_sum_error=float(np.max(np.abs(weights[good].sum(axis=1)-1))),
                           actual_pial_white_endpoint_check=bool(endpoint_ok),
                           native_thickness_file=str(t), native_thickness_sha256=sha256(t),
                           profile_file=str(p), mapping_file=str(m)))
    return np.concatenate(arrays), np.concatenate(thick), np.concatenate(length), qc


def prepare(args):
    root = args.output_dir
    for name in ('tables', 'surface_maps', 'adjusted_profiles', 'qc', 'figures', 'cache', 'figure3B_subject_pc1_scores'):
        (root / name).mkdir(parents=True, exist_ok=True)
    write_json(root/'qc/run_configuration.json',
               {key: str(value) if isinstance(value,Path) else value for key,value in vars(args).items()})
    nodes = atlas(args)
    nodes.to_csv(root / 'qc/vertex_order_and_labels.csv', index=False)
    exclusions = set()
    if args.qc_exclusions:
        table = pd.read_csv(args.qc_exclusions, dtype={'subject_id': str})
        exclusions = set(table.subject_id)
    if args.subject_list:
        if args.subject_list.suffix == '.csv':
            candidates = pd.read_csv(args.subject_list, dtype={'subject_id': str}).subject_id.tolist()
        else:
            candidates = [x.strip() for x in args.subject_list.read_text().splitlines() if x.strip() and not x.startswith('#')]
    else:
        candidates = sorted(p.name for p in args.profiles_dir.iterdir() if p.is_dir())
    if len(set(candidates)) != len(candidates):
        raise ValueError('Duplicate candidate subjects')
    included, flow, coverage_screen = [], [], []
    for subject in candidates:
        missing = [str(p) for hemi in HEMIS for p in paths(args, subject, hemi) if not p.is_file()]
        n_mask = n_valid = 0
        coverage_fraction = np.nan
        passes_coverage = False
        if subject in exclusions:
            reason = 'registration_qc_exclusion'
        elif missing:
            reason = 'missing_inputs'
        else:
            n_mask, n_valid, coverage_fraction = profile_coverage(args, subject)
            passes_coverage = coverage_fraction >= args.minimum_subject_coverage
            reason = 'candidate' if passes_coverage else 'profile_coverage_below_threshold'
            coverage_screen.append(dict(
                subject_id=subject, n_analysis_vertices=n_mask, n_valid_columns=n_valid,
                coverage_fraction=coverage_fraction,
                minimum_subject_coverage=args.minimum_subject_coverage,
                passes_minimum_coverage=passes_coverage))
        flow.append(dict(subject_id=subject, initial_status=reason,
                         missing_inputs=';'.join(missing),
                         profile_coverage_fraction=coverage_fraction,
                         passes_minimum_coverage=passes_coverage))
        if reason == 'candidate':
            included.append(subject)
    if args.max_subjects: included = included[:args.max_subjects]
    if len(included) < 4: raise ValueError('At least four complete QC-passed subjects required')
    n, v = len(included), len(nodes)
    raw = np.lib.format.open_memmap(root/'cache/raw_md.npy', mode='w+', dtype='float32', shape=(n,v,21))
    thick = np.lib.format.open_memmap(root/'cache/thickness.npy', mode='w+', dtype='float32', shape=(n,v))
    lengths = np.lib.format.open_memmap(root/'cache/column_length.npy', mode='w+', dtype='float32', shape=(n,v))
    coverage = np.zeros(v, dtype=int); qcs = []
    for row, subject in enumerate(included):
        values, t, l, qc = load_subject(args, subject, nodes)
        valid = np.isfinite(t)
        if not valid.any():
            raise ValueError(f'{subject}: no valid vertices after full input validation')
        raw[row], thick[row], lengths[row] = values, t, l
        coverage += valid; qcs.extend(qc)
        print(f'prepare_subject={row+1}/{n} subject={subject} valid={valid.sum()}', flush=True)
    raw.flush(); thick.flush(); lengths.flush()
    mask = nodes.atlas_mask.to_numpy(bool) & (coverage/n >= args.minimum_vertex_coverage)
    if not mask.any(): raise ValueError('Empty common analysis mask')
    np.save(root/'tables/figure3_analysis_mask.npy', mask.reshape(2,-1))
    cov_table = nodes[['hemisphere','fsaverage_vertex_id','dk_label_name']].copy()
    cov_table['n_valid_subjects'] = coverage
    cov_table['coverage_fraction'] = coverage/n
    cov_table['in_analysis_mask'] = mask
    cov_table.to_csv(root/'tables/figure3_coverage_by_vertex.csv', index=False)
    subject_table = pd.DataFrame({'subject_id':included, 'row_index':np.arange(n)})
    subject_table['valid_vertices'] = [int(np.isfinite(thick[i,mask]).sum()) for i in range(n)]
    subject_table['atlas_coverage_fraction'] = [float(np.isfinite(thick[i]).sum()/nodes.atlas_mask.sum()) for i in range(n)]
    subject_table.to_csv(root/'tables/figure3_subjects.csv', index=False)
    pd.DataFrame(qcs).to_csv(root/'qc/subject_input_qc.csv', index=False)
    pd.DataFrame(flow).to_csv(root/'qc/candidate_subject_flow.csv', index=False)
    pd.DataFrame(coverage_screen).to_csv(root/'qc/candidate_profile_coverage_screen.csv', index=False)
    write_json(root/'qc/preparation.json', dict(n_subjects=n, n_full_vertices=v, n_analysis_vertices=int(mask.sum()),
        candidates=len(candidates), known_qc_exclusions=sorted(exclusions),
        minimum_subject_coverage=args.minimum_subject_coverage, minimum_vertex_coverage=args.minimum_vertex_coverage,
        thickness_range_mm=[args.thickness_min,args.thickness_max],
        mask_layout='(hemisphere lh/rh, fsaverage vertex); retained vertices require cohort coverage threshold',
        subject_coverage_definition='Bilateral sum(in_analysis_mask & column_valid) / sum(in_analysis_mask), matching Figure 5',
        missing_depth_policy='Whole profile invalid if any depth nonfinite/nonpositive; no interpolation',
        thickness_definition='Native FreeSurfer thickness mapped with the exact Option 3 native triangle and barycentric weights',
        depth_direction='pial_to_white', smoke_test=bool(args.max_subjects)))


def regional_table(subject_ids, regions, profiles):
    rows = []
    for i, subject in enumerate(subject_ids):
        for r, region in regions.iterrows():
            for d, depth in enumerate(DEPTHS):
                rows.append((subject,region.hemisphere,region.dk_label_name,r,d,depth,float(profiles[i,r,d])))
    return pd.DataFrame(rows, columns=['subject_id','hemisphere','region','region_index','depth_index','normalized_depth','md'])


def group_regional(table):
    return table.groupby(['hemisphere','region','region_index','depth_index','normalized_depth'], sort=False).md.agg(
        mean_md='mean',sd_md='std',n_subjects='count').reset_index()


def surface(path, values):
    nib.save(nib.MGHImage(np.asarray(values,dtype=np.float32).reshape(-1,1,1),np.eye(4)),path)


def analyze(args):
    root, tables = args.output_dir, args.output_dir/'tables'
    subjects = pd.read_csv(tables/'figure3_subjects.csv').subject_id.tolist()
    nodes = pd.read_csv(root/'qc/vertex_order_and_labels.csv')
    mask = np.load(tables/'figure3_analysis_mask.npy').reshape(-1)
    raw, thickness, lengths = [np.load(root/f'cache/{name}.npy',mmap_mode='r') for name in ('raw_md','thickness','column_length')]
    n, v, _ = raw.shape
    regions = nodes.loc[mask,['hemisphere','dk_label_name','region_index']].drop_duplicates().sort_values('region_index').reset_index(drop=True)
    if not np.array_equal(regions.region_index,np.arange(68)):
        raise ValueError('Expected all 68 exact hemisphere-DK labels')
    region_ids = nodes.region_index.to_numpy(int)
    reg = np.full((n,68,21),np.nan); rt = np.full((n,68),np.nan); rcount = np.zeros((n,68),int)
    means = np.empty((n,21)); moments = np.empty((n,21,21)); tt = np.empty(n)
    ty = np.empty((n,21)); yy = np.empty((n,21)); lss = np.empty(n); ly = np.empty((n,21))
    center_max = 0.; valid_counts = []; t_means = []; distributions=[]
    for i, subject in enumerate(subjects):
        valid = mask & np.isfinite(thickness[i])
        x = np.asarray(raw[i,valid],dtype=float); t = np.asarray(thickness[i,valid],dtype=float)
        shaped = center_profiles(x); center_max=max(center_max,float(np.max(np.abs(shaped.mean(axis=1)))))
        means[i],moments[i]=shaped.mean(axis=0),shaped.T@shaped/len(shaped)
        tc=t-t.mean(); xc=x-x.mean(axis=0); tt[i]=tc@tc; ty[i]=tc@xc; yy[i]=(xc*xc).sum(axis=0)
        l=np.asarray(lengths[i,valid],dtype=float); lc=l-l.mean(); lss[i]=lc@lc; ly[i]=lc@xc
        valid_counts.append(len(x)); t_means.append(t.mean())
        distributions.append(dict(subject_id=subject,n_valid=len(x),mean=t.mean(),sd=t.std(ddof=1),
                                  minimum=t.min(),q01=np.quantile(t,.01),median=np.median(t),q99=np.quantile(t,.99),maximum=t.max()))
        for r in range(68):
            keep=valid & (region_ids==r); rcount[i,r]=keep.sum()
            if not keep.any():continue
            reg[i,r]=np.asarray(raw[i,keep],float).mean(axis=0);rt[i,r]=np.mean(thickness[i,keep])
        print(f'covariance_subject={i+1}/{n}',flush=True)
    np.savez(root/'cache/subject_sufficient_statistics.npz',means=means,moments=moments,tt=tt,ty=ty,yy=yy,
             raw_regional=reg,regional_thickness=rt,regional_counts=rcount,t_means=t_means,lss=lss,ly=ly)
    pd.DataFrame(distributions).to_csv(root/'qc/thickness_distribution_by_subject.csv',index=False)
    rt_table=regions.copy();rt_table['mean_thickness']=np.nanmean(rt,axis=0);rt_table['SD_thickness']=np.nanstd(rt,axis=0,ddof=1);rt_table['n_subjects']=np.isfinite(rt).sum(axis=0)
    rt_table.to_csv(tables/'figure3A_region_thickness.csv',index=False)
    # Equal left/right regional means, then equal subject weights. Representative
    # regions are selected later from thickness only, never by MD appearance.
    combined=[]
    for name in regions.dk_label_name.unique():
        which=np.flatnonzero(regions.dk_label_name.eq(name))
        ct=rt[:,which].mean(axis=1)
        combined.append(dict(region=name,mean_thickness=np.nanmean(ct),SD_thickness=np.nanstd(ct,ddof=1),n_subjects=int(np.isfinite(ct).sum())))
    pd.DataFrame(combined).to_csv(tables/'figure3A_bilateral_region_thickness.csv',index=False)
    raw_table=regional_table(subjects,regions,reg)
    raw_table.to_csv(tables/'figure3A_subject_regional_profiles.csv',index=False)
    group_regional(raw_table).to_csv(tables/'figure3A_group_regional_profiles.csv',index=False)
    coverage_rows=[dict(subject_id=subjects[i],hemisphere=regions.iloc[r].hemisphere,region=regions.iloc[r].dk_label_name,n_valid_vertices=int(rcount[i,r])) for i in range(n) for r in range(68)]
    pd.DataFrame(coverage_rows).to_csv(root/'qc/region_coverage.csv',index=False)
    pd.DataFrame(coverage_rows).query('n_valid_vertices == 0').to_csv(root/'qc/missing_subject_region_cells.csv',index=False)
    global_mean,eigenvalues,vectors,flipped,signcorr=weighted_pca(means,moments)
    loadings=pd.DataFrame({'depth_index':np.arange(21),'normalized_depth':DEPTHS,'pca_feature_mean':global_mean})
    for k in range(5):loadings[f'PC{k+1}']=vectors[:,k]
    loadings.to_csv(tables/'figure3B_pca_loadings.csv',index=False)
    ev=eigenvalues[:5];ratios=ev/eigenvalues.sum()
    pd.DataFrame(dict(component=np.arange(1,6),explained_variance=ev,explained_variance_ratio=ratios,cumulative_explained_variance=np.cumsum(ratios))).to_csv(tables/'figure3B_pca_explained_variance.csv',index=False)
    np.save(root/'cache/pc1_loading.npy',vectors[:,0])
    beta,std_beta,pr2=thickness_effect(tt,ty,yy)
    beta,std_beta,pr2=beta[0],std_beta[0],pr2[0]
    rng=np.random.default_rng(args.seed)
    resamples=rng.integers(0,n,(args.bootstrap_repetitions,n))
    boot_weights=np.zeros((args.bootstrap_repetitions,n))
    for b,indices in enumerate(resamples):boot_weights[b]=np.bincount(indices,minlength=n)
    bb,bs,br=thickness_effect(tt,ty,yy,boot_weights)
    effect=pd.DataFrame(dict(depth_index=np.arange(21),normalized_depth=DEPTHS,beta_thickness=beta,
        standardized_beta=std_beta,partial_r2=pr2,beta_ci95_lower=np.percentile(bb,2.5,axis=0),beta_ci95_upper=np.percentile(bb,97.5,axis=0),
        standardized_beta_ci95_lower=np.percentile(bs,2.5,axis=0),standardized_beta_ci95_upper=np.percentile(bs,97.5,axis=0),
        partial_r2_ci95_lower=np.percentile(br,2.5,axis=0),partial_r2_ci95_upper=np.percentile(br,97.5,axis=0)))
    effect.to_csv(tables/'figure3C_depth_specific_thickness_effect.csv',index=False)
    adjusted_reg=reg-beta[None,None,:]*(rt-np.asarray(t_means)[:,None])[:,:,None]
    adj_table=regional_table(subjects,regions,adjusted_reg)
    adj_table.to_csv(tables/'figure3D_subject_adjusted_regional_profiles.csv',index=False)
    group_regional(adj_table).to_csv(tables/'figure3D_group_adjusted_regional_profiles.csv',index=False)
    # Bootstrap refits beta in each resample before adjusted regional R².
    # SSE and region SS are quadratic in beta, so this uses exact sufficient sums.
    region_stats=regional_sufficient_statistics(reg,rt)
    raw_result,adjusted_result=regional_effects_from_statistics(region_stats,beta=beta)
    raw_r2,adjusted_r2=raw_result[0],adjusted_result[0]
    raw_boot=np.empty((len(resamples),21));adjusted_boot=np.empty_like(raw_boot)
    for start in range(0,len(resamples),128):
        stop=min(start+128,len(resamples));w=boot_weights[start:stop];b=bb[start:stop]
        raw_boot[start:stop],adjusted_boot[start:stop]=regional_effects_from_statistics(region_stats,w,b)
    pd.DataFrame(dict(depth_index=np.arange(21),normalized_depth=DEPTHS,raw_partial_r2=raw_r2,
        raw_ci95_lower=np.percentile(raw_boot,2.5,axis=0),raw_ci95_upper=np.percentile(raw_boot,97.5,axis=0),
        adjusted_partial_r2=adjusted_r2,adjusted_ci95_lower=np.percentile(adjusted_boot,2.5,axis=0),adjusted_ci95_upper=np.percentile(adjusted_boot,97.5,axis=0),
        n_subjects=n,n_observed_subject_region_cells=int((rcount>0).sum())
        )).to_csv(tables/'figure3D_region_effect_by_depth.csv',index=False)
    sums=np.zeros(v);sums2=np.zeros(v);counts=np.zeros(v,int);tsum=np.zeros(v);mdsum=np.zeros(v)
    assoc=[];sensitivity=[]
    half=v//2
    for i,subject in enumerate(subjects):
        valid=mask & np.isfinite(thickness[i]);x=np.asarray(raw[i,valid],float);t=np.asarray(thickness[i,valid],float)
        shape=center_profiles(x);score=shape@vectors[:,0]
        sums[valid]+=score;sums2[valid]+=score**2;counts[valid]+=1;tsum[valid]+=t;mdsum[valid]+=x.mean(axis=1)
        rho=float(spearmanr(t,score).statistic);r=float(pearsonr(t,score).statistic)
        assoc.append(dict(subject_id=subject,n_valid_vertices=int(valid.sum()),spearman_rho=rho,pearson_r=r,
                          standardized_slope=r))
        for feature,values in [('mean_md',x.mean(axis=1)),('column_length',np.asarray(lengths[i,valid],float))]:
            sensitivity.append(dict(subject_id=subject,feature=feature,
                spearman_with_pc1=float(spearmanr(values,score).statistic),
                spearman_with_thickness=float(spearmanr(t,values).statistic)))
        score_full=np.full(v,np.nan,dtype=np.float32);score_full[valid]=score
        np.savez_compressed(root/'figure3B_subject_pc1_scores'/f'{subject}_pc1_scores.npz',
            subject_id=subject,scores=score_full.reshape(2,-1),valid_mask=valid.reshape(2,-1),depth_fractions=DEPTHS)
        for hi,hemi in enumerate(HEMIS):
            sl=slice(hi*half,(hi+1)*half);ok=valid[sl]
            adj=np.full((half,21),np.nan,dtype=np.float32)
            adj[ok]=np.asarray(raw[i,sl][ok],float)-beta*(np.asarray(thickness[i,sl][ok],float)-t_means[i])[:,None]
            np.savez_compressed(root/'adjusted_profiles'/f'{subject}_{hemi}_adjusted_md.npz',
                subject_id=subject,hemisphere=hemi,fsaverage_vertex_id=np.arange(half),profiles=adj,
                valid_mask=ok,depth_fractions=DEPTHS,depth_direction='pial_to_white',
                adjustment='MD - beta_depth * (native_mapped_thickness - subject_mean_thickness)')
        print(f'score_and_adjustment_subject={i+1}/{n}',flush=True)
    mean=np.divide(sums,counts,out=np.full(v,np.nan),where=counts>0)
    variance=np.divide(sums2-sums*mean,counts-1,out=np.full(v,np.nan),where=counts>1)
    sd=np.sqrt(np.maximum(variance,0));se=sd/np.sqrt(np.maximum(counts,1))
    d=np.divide(mean,sd,out=np.full(v,np.nan),where=sd>0)
    tmean=np.divide(tsum,counts,out=np.full(v,np.nan),where=counts>0)
    for hi,hemi in enumerate(HEMIS):
        sl=slice(hi*half,(hi+1)*half)
        for name,values in [('mean',mean),('sd',sd),('se',se),('cohens_d',d),('n_valid',counts)]:
            surface(root/'surface_maps'/f'{hemi}.figure3B_pc1_{name}.mgh',values[sl])
        surface(root/'surface_maps'/f'{hemi}.figure3_mean_thickness.mgh',tmean[sl])
    group_map=nodes[['hemisphere','fsaverage_vertex_id']].copy()
    group_map['mean_pc1_score']=mean;group_map['sd_pc1_score']=sd;group_map['se_pc1_score']=se
    group_map['cohens_d']=d;group_map['n_valid_subjects']=counts;group_map['mean_thickness']=tmean
    group_map['mean_md']=np.divide(mdsum,counts,out=np.full(v,np.nan),where=counts>0)
    group_map.to_csv(tables/'figure3B_group_pc1_surface_statistics.csv',index=False)
    associations=pd.DataFrame(assoc);associations.to_csv(tables/'figure3C_subject_spatial_correlations.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(tables/'figure3S_subject_sensitivity_associations.csv',index=False)
    vals=associations.spearman_rho.to_numpy();slopes=associations.standardized_slope.to_numpy()
    stats={'median_spearman_rho':(np.median(vals),np.median(vals[resamples],axis=1)),
           'mean_standardized_slope':(np.mean(slopes),np.mean(slopes[resamples],axis=1)),
           'proportion_positive_rho':(np.mean(vals>0),np.mean(vals[resamples]>0,axis=1))}
    summary=[]
    for name,(estimate,boots) in stats.items():
        lo,hi=np.percentile(boots,[2.5,97.5]);summary.append(dict(statistic=name,estimate=estimate,ci95_lower=lo,ci95_upper=hi,
            n_subjects=n,spearman_q25=np.percentile(vals,25),spearman_q75=np.percentile(vals,75),bootstrap_repetitions=len(resamples),seed=args.seed))
    pd.DataFrame(summary).to_csv(tables/'figure3C_group_association_summary.csv',index=False)
    good=mask & np.isfinite(mean) & np.isfinite(tmean)
    sorted_ids=np.flatnonzero(good)[np.argsort(tmean[good],kind='stable')]
    bins=[]
    for bi,ids in enumerate(np.array_split(sorted_ids,40)):
        bins.append(dict(bin=bi,n_vertices=len(ids),mean_thickness=float(tmean[ids].mean()),
            mean_pc1_score=float(mean[ids].mean()),vertex_se=float(mean[ids].std(ddof=1)/np.sqrt(len(ids)))))
    pd.DataFrame(bins).to_csv(tables/'figure3C_binned_group_scatter.csv',index=False)
    lb,ls,lr=thickness_effect(lss,ly,yy);lbboot,_,_=thickness_effect(lss,ly,yy,boot_weights)
    length_adjusted=reg.copy()
    # Regional length means use the same valid vertices, never replace thickness.
    regional_lengths=np.full((n,68),np.nan)
    for i in range(n):
        valid=mask & np.isfinite(thickness[i])
        for r in range(68):
            keep=valid & (region_ids==r)
            if keep.any():regional_lengths[i,r]=np.mean(lengths[i,keep])
        length_adjusted[i]-=lb[0]*(regional_lengths[i]-np.mean(lengths[i,valid]))[:,None]
    pd.DataFrame(dict(normalized_depth=DEPTHS,beta_length=lb[0],partial_r2=lr[0],
        beta_ci95_lower=np.percentile(lbboot,2.5,axis=0),beta_ci95_upper=np.percentile(lbboot,97.5,axis=0),
        length_adjusted_region_partial_r2=regional_partial_r2(length_adjusted)[0])).to_csv(tables/'figure3S_column_length_sensitivity.csv',index=False)
    write_json(root/'qc/analysis.json',dict(status='complete',n_subjects=n,n_valid_profiles_used_for_pca=int(np.sum(valid_counts)),
        pca_subject_weight='equal total weight per subject; covariance uses subject-normalized first and second moments',
        mean_centered_row_mean_max_abs=center_max,mean_centering_tolerance=1e-12,pc1_loading_sum=float(vectors[:,0].sum()),
        pc1_explained_variance_ratio=float(ratios[0]),pc1_sign_flipped=flipped,pc1_inner_outer_correlation=signcorr,
        pc1_score_definition='profile-wise centered MD dot loading, as specified; PCA fitting additionally feature-centers the covariance',
        positive_pc1='relatively higher white-side MD than pial-side MD',
        thickness_source='FreeSurfer native thickness via Option 3 barycentric mapping; not column length',
        bootstrap_repetitions=args.bootstrap_repetitions,seed=args.seed,
        region_model='68 hemisphere-DK region fixed effects plus subject fixed effects; observed cells only, exact unbalanced-design OLS; vertex means computed within subjects',
        n_missing_subject_region_cells=int((rcount==0).sum()),regional_missing_value_policy='Unobserved cells remain NaN and are omitted from regional OLS; no imputation or whole-subject removal',
        adjusted_bootstrap='refit depth-specific thickness beta in each subject resample before regional partial R2',
        no_depth_interpolation=True))


def stability(args):
    root=args.output_dir
    with np.load(root/'cache/subject_sufficient_statistics.npz') as f:
        means,moments=f['means'],f['moments']
    raw=np.load(root/'cache/raw_md.npy',mmap_mode='r')
    thickness=np.load(root/'cache/thickness.npy',mmap_mode='r')
    mask=np.load(root/'tables/figure3_analysis_mask.npy').reshape(-1)
    n,v,_=raw.shape; repeats=args.split_half_repetitions
    rng=np.random.default_rng(args.seed+1);membership=np.zeros((repeats,n),dtype=np.float32)
    w1,w2=np.empty((repeats,21)),np.empty((repeats,21));rows=[]
    for repeat in range(repeats):
        indices=rng.permutation(n);a,b=indices[:n//2],indices[n//2:];membership[repeat,a]=1
        _,e1,u1,_,_=weighted_pca(means[a],moments[a]);_,e2,u2,_,_=weighted_pca(means[b],moments[b])
        w1[repeat],w2[repeat]=u1[:,0],u2[:,0]
        rows.append(dict(repeat=repeat,n_half1=len(a),n_half2=len(b),loading_correlation=float(pearsonr(u1[:,0],u2[:,0]).statistic),
            explained_variance_ratio_half1=float(e1[0]/e1.sum()),explained_variance_ratio_half2=float(e2[0]/e2.sum())))
    # Read a vertex block across subjects, then form group-mean shapes in all
    # split-halves by BLAS. Never materialize the entire subject×vertex×depth array.
    sums=np.zeros((repeats,5),dtype=float);valid_counts=np.zeros(repeats,int)
    for start in range(0,v,args.vertex_block_size):
        stop=min(start+args.vertex_block_size,v);size=stop-start
        if not mask[start:stop].any():continue
        block=np.asarray(raw[:,start:stop],dtype=float)
        valid=np.isfinite(thickness[:,start:stop]) & mask[None,start:stop]
        block=np.nan_to_num(block,nan=0.0);block-=block.mean(axis=2,keepdims=True)
        block[~valid]=0
        c1=membership@valid.astype(np.float32);c2=(1-membership)@valid.astype(np.float32)
        shape1=(membership@block.reshape(n,-1)).reshape(repeats,size,21)
        shape2=((1-membership)@block.reshape(n,-1)).reshape(repeats,size,21)
        a=np.einsum('rvd,rd->rv',shape1,w1)/np.maximum(c1,1)
        b=np.einsum('rvd,rd->rv',shape2,w2)/np.maximum(c2,1)
        good=(c1>0)&(c2>0);a[~good]=0;b[~good]=0
        valid_counts+=good.sum(axis=1)
        sums+=np.column_stack([a.sum(axis=1),b.sum(axis=1),(a*a).sum(axis=1),(b*b).sum(axis=1),(a*b).sum(axis=1)])
        print(f'split_half_vertices={stop}/{v}',flush=True)
    for i,row in enumerate(rows):
        sa,sb,saa,sbb,sab=sums[i];count=valid_counts[i]
        row['group_surface_map_correlation']=float((sab-sa*sb/count)/np.sqrt((saa-sa*sa/count)*(sbb-sb*sb/count)))
        row['n_common_group_map_vertices']=int(count);row['seed']=args.seed+1
    pd.DataFrame(rows).to_csv(root/'tables/figure3S_pc1_split_half_stability.csv',index=False)
    np.savez(root/'qc/split_half_membership.npz',membership=membership,pc1_half1=w1,pc1_half2=w2)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('profiles-dir','subjects-dir','mapping-dir','fsaverage-dir','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--subject-list',type=Path)
    p.add_argument('--qc-exclusions',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','analyze','stability','plot','resume','all'],default='all')
    p.add_argument('--minimum-subject-coverage',type=float,default=.95,
                   help='Required bilateral cleaned-profile coverage; default 0.95 matches Figure 5')
    p.add_argument('--minimum-vertex-coverage',type=float,default=0.,help='Optional explicit cohort coverage filter; default retains the original atlas mask')
    p.add_argument('--thickness-min',type=float,default=0.)
    p.add_argument('--thickness-max',type=float,default=8.)
    p.add_argument('--bootstrap-repetitions',type=int,default=10000)
    p.add_argument('--split-half-repetitions',type=int,default=100)
    p.add_argument('--seed',type=int,default=20261102)
    p.add_argument('--vertex-block-size',type=int,default=2048)
    p.add_argument('--max-subjects',type=int,help='Smoke test only; recorded in metadata')
    args=p.parse_args()
    if not 0 <= args.minimum_subject_coverage <= 1:
        p.error('--minimum-subject-coverage must be in [0, 1]')
    if args.bootstrap_repetitions<1 or args.split_half_repetitions<1: p.error('Repetition counts must be positive')
    return args


if __name__=='__main__':
    args=parse_args();start=time.time()
    for stage in ('prepare','analyze','stability','plot'):
        if args.stage in ('all',stage) or (args.stage=='resume' and stage!='prepare'):
            if stage=='plot':
                from plot_figure3 import plot_all
                plot_all(args.output_dir,args.fsaverage_dir)
            else:globals()[stage](args)
    completion=dict(stage=args.stage,status='complete',elapsed_seconds=time.time()-start)
    if args.stage in ('all','resume'):
        write_json(args.output_dir/'qc/complete.json',completion)
    print(json.dumps(completion),flush=True)
