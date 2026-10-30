#!/usr/bin/env python3
"""Compute revised Option 3 QC tables and figures outside the presentation notebook."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')


DEFAULT_OUTPUT_DIR = Path(
    "/mnt/newStor/paros/paros_WORK/xintong/results/qc/"
    "full_cohort_spatial_scale/option3_alignment_qc_v2"
)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[3]
    parser.add_argument('--manifest', type=Path, default=repo/'manifests/full_300_subject_manifest.csv')
    parser.add_argument('--subjects-dir', type=Path, default=Path('/mnt/newStor/paros/paros_WORK/xintong/data/processed/t1_freesurfer'))
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--covariates', type=Path)
    parser.add_argument('--limit', type=int, help='Process only the first N manifest subjects for validation')
    parser.add_argument('--zero-length-tolerance-mm', type=float, default=1e-6)
    parser.add_argument('--coverage-review-pct', type=float, default=95.0)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    import sys, json, shlex
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import nibabel as nib
    from nibabel.freesurfer import io as fsio
    from scipy import stats
    REPO = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(REPO))
    from qc.option3.scripts.compute_option3_qc import read_mincost, template_boundary_hops
    from qc.publication_style import set_publication_style
    MANIFEST = args.manifest
    SUBJECTS_DIR = args.subjects_dir
    TEMPLATE = SUBJECTS_DIR / 'fsaverage'
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'run_metadata.json').unlink(missing_ok=True)
    ZERO_MM = args.zero_length_tolerance_mm
    WEIGHT_TOL = 1e-6
    COVERAGE_REVIEW_PCT = args.coverage_review_pct
    ROBUST_Z = 3.5
    COVARIATES = args.covariates
    manifest = pd.read_csv(MANIFEST, dtype={'subject_id': str})
    assert manifest.subject_id.is_unique
    if args.limit is not None:
        manifest = manifest.iloc[:args.limit].copy()
    print(f'Requested: {len(manifest)} subjects; outputs: {OUT}', flush=True)
    set_publication_style()
    FIGURE_DIR = OUT / 'figures'
    TABLE_DIR = OUT / 'tables'
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    figure_names = []

    def save_current_figure(name):
        figure_names.append(name)
        figure = plt.gcf()
        figure.savefig(FIGURE_DIR / f'{name}.png', dpi=300, bbox_inches='tight')
        figure.savefig(FIGURE_DIR / f'{name}.pdf', bbox_inches='tight')
        plt.close()
    def display(value):
        print(value.to_string() if hasattr(value, 'to_string') else value, flush=True)


    def load_npz(path):
        with np.load(path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    def text_scalar(x):
        x = np.asarray(x).item()
        return x.decode() if isinstance(x, bytes) else str(x)

    def pct(x, mask):
        return 100 * np.mean(x[mask]) if mask.any() else np.nan

    def agreement(length, thickness):
        ok = np.isfinite(length) & np.isfinite(thickness) & (thickness > 0)
        x, y = length[ok], thickness[ok]
        variable = len(x) > 2 and np.std(x) > 0 and np.std(y) > 0
        return dict(pearson_r=stats.pearsonr(x,y).statistic if variable else np.nan,
                    spearman_rho=stats.spearmanr(x,y).statistic if variable else np.nan,
                    mae_mm=np.mean(abs(x-y)) if len(x) else np.nan,
                    median_difference_mm=np.median(x-y) if len(x) else np.nan,
                    median_length_thickness_ratio=np.median(x/y) if len(x) else np.nan)

    # Fixed template mask built independently of each subject's sampling outcome.
    fixed_masks = {}
    for hemi in ('lh','rh'):
        labels, _, names = fsio.read_annot(str(TEMPLATE/'label'/f'{hemi}.aparc.annot'))
        names = [n.decode().lower() for n in names]
        excluded = {'unknown','corpuscallosum','medial_wall','medialwall','background','???'}
        cortex = np.zeros(len(labels), bool)
        cortex[fsio.read_label(str(TEMPLATE/'label'/f'{hemi}.cortex.label')).astype(int)] = True
        fixed_masks[hemi] = cortex & (labels >= 0) & np.isin(labels, [i for i,n in enumerate(names) if n not in excluded])

    def inspect_hemi(row, hemi, spatial=False):
        # Cohort loops yield namedtuples; spatial selection yields a Series.
        if hasattr(row, "_asdict"):
            row = pd.Series(row._asdict())
        c, p = load_npz(row[f'column_{hemi}']), load_npz(row[f'profile_{hemi}'])
        ids = c['target_vertex_id'].astype(int)
        assert np.array_equal(ids, np.arange(len(fixed_masks[hemi]))), 'Expected full fsaverage indexing'
        assert np.array_equal(ids, p['target_vertex_id']), 'Column/profile vertex mismatch'
        mask = fixed_masks[hemi]
        assert np.array_equal(c['in_analysis_mask'].astype(bool), mask), 'Cached mask differs from cortex ∩ valid DK mask'
        d = np.linspace(0,1,21)
        for z in (c,p):
            assert text_scalar(z['depth_direction']) == 'pial_to_white'
            assert np.allclose(z['depth_fractions'], d, atol=1e-8, rtol=0)
        points, values = c['column_points'], p['profiles']
        assert points.shape == (len(ids),21,3) and values.shape == (len(ids),21)
        w, src = c['barycentric_weights'], c['source_vertex_ids'].astype(int)
        sphere, _ = fsio.read_geometry(str(Path(row.freesurfer_subject_dir)/'surf'/f'{hemi}.sphere.reg'))
        # VTK mapping uses the centered, normalized sphere, not raw ~100 mm coordinates.
        sphere = (sphere - c['source_sphere_center']) / float(c['source_sphere_radius'])
        legal_ids = ((src >= 0) & (src < len(sphere))).all(1)
        legal_w = np.isfinite(w).all(1) & (w >= -WEIGHT_TOL).all(1) & (w <= 1+WEIGHT_TOL).all(1)
        sumerr = abs(w.sum(1)-1)
        mapping = c['mapping_valid'].astype(bool) & (c['source_face_id'] >= 0) & legal_ids & legal_w & (sumerr <= WEIGHT_TOL)
        recon = np.full(len(ids), np.nan)
        recon[legal_ids] = np.linalg.norm(np.einsum('ni,nij->nj', w[legal_ids], sphere[src[legal_ids]]) - c['closest_point_on_source_sphere'][legal_ids], axis=1)
        a,b = c['pial_endpoint'], c['white_endpoint']
        length = np.linalg.norm(b-a, axis=1)
        finite = np.isfinite(points).all((1,2)) & np.isfinite(length)
        zero = np.isfinite(length) & (length < ZERO_MM)
        geometry = mapping & finite & ~zero
        xyz = p['dwi_voxel_coordinates']
        shape = np.array(nib.load(row.md_image).shape[:3])
        bounds = np.isfinite(xyz).all(2) & (xyz >= 0).all(2) & (xyz <= shape-1).all(2)
        assert np.array_equal(bounds, p['coordinate_in_bounds_mask'].astype(bool)), 'DWI bounds cache disagreement'
        nonfinite = ~np.isfinite(values)
        zeros = np.isfinite(values) & (values == 0)
        invalid = ~bounds | nonfinite | zeros | ~p['sampled_finite_mask'].astype(bool)
        final = geometry & ~invalid.any(1)
        expected = a[:,None]*(1-d[None,:,None]) + b[:,None]*d[None,:,None]
        steps = np.linalg.norm(np.diff(points,axis=1),axis=2)
        r = dict(subject_id=row.subject_id, hemisphere=hemi, n_mask=int(mask.sum()),
            mapping_full_pct=100*mapping.mean(), mapping_mask_pct=pct(mapping,mask),
            triangle_lookup_pct=100*np.mean(c['source_face_id'] >= 0),
            illegal_source_ids=int((~legal_ids).sum()), illegal_weights=int((~legal_w).sum()),
            weight_sum_error_max=float(np.nanmax(sumerr)), reconstruction_error_max=float(np.nanmax(recon)),
            geometry_pct=pct(geometry,mask), final_pct=pct(final,mask),
            zero_full_pct=100*zero.mean(), zero_mask_pct=pct(zero,mask),
            point_invalid_pct=100*invalid[mask].mean(), column_invalid_pct=pct(invalid.any(1),mask),
            sampling_retention_geometry_pct=pct(~invalid.any(1),mask & geometry),
            endpoint_pial_error_mm=float(np.nanmax(np.linalg.norm(points[:,0]-a,axis=1))),
            endpoint_white_error_mm=float(np.nanmax(np.linalg.norm(points[:,-1]-b,axis=1))),
            line_error_mm=float(np.nanmax(np.linalg.norm(points-expected,axis=2))),
            equal_spacing_error_mm=float(np.nanmax(abs(steps-length[:,None]/20))))
        pairs = np.c_[length[mask & geometry], c['cortical_thickness'][mask & geometry]]
        r.update(agreement(*pairs.T))
        depths = [dict(subject_id=row.subject_id, hemisphere=hemi, depth=float(d[j]),
            oob_pct=100*(~bounds[mask,j]).mean(), nan_pct=100*np.isnan(values[mask,j]).mean(),
            nonfinite_pct=100*nonfinite[mask,j].mean(), zero_pct=100*zeros[mask,j].mean()) for j in range(21)]
        maps = dict(mask=mask, mapping_failed=~mapping, zero=zero, invalid=~final,
                    length_minus_thickness=length-c['cortical_thickness'], xyz=xyz)
        return r, depths, pairs, maps if spatial else None

    rows, depth_rows, subjects, statuses = [], [], [], []
    for row in manifest.itertuples(index=False):
        print(f'Processing {row.subject_id}', flush=True)
        try:
            results = [inspect_hemi(row,h) for h in ('lh','rh')]
            rr = [x[0] for x in results]
            rows.extend(rr)
            for x in results: depth_rows.extend(x[1])
            # Same fixed template hemisphere sizes for every subject; weight by mask size.
            pooled = {'subject_id': row.subject_id}
            for key in ('mapping_mask_pct','geometry_pct','final_pct','zero_mask_pct','point_invalid_pct','column_invalid_pct'):
                pooled[key] = np.average([r[key] for r in rr], weights=[r['n_mask'] for r in rr])
            pooled.update(agreement(*np.concatenate([x[2] for x in results]).T))
            subjects.append(pooled)
            statuses.append(dict(subject_id=row.subject_id,status='complete',reason=''))
        except Exception as e:
            statuses.append(dict(subject_id=row.subject_id,status='failed',reason=f'{type(e).__name__}: {e}'))
    hemi_qc, subject_qc, depth_qc, status = map(pd.DataFrame, (rows, subjects, depth_rows, statuses))
    status.to_csv(TABLE_DIR/'status.csv',index=False)
    display(status.groupby('status').size().to_frame('n_subjects'))
    display(status.loc[status.status != 'complete'])
    if subject_qc.empty:
        reasons = status['reason'].value_counts().to_string()
        raise RuntimeError(f'No successful subjects. Failure reasons:\n{reasons}\nFull details: {OUT / "status.csv"}')

    def distribution(df, columns):
        return df[columns].agg(['median',lambda x:x.quantile(.25),lambda x:x.quantile(.75),'min','max']).set_axis(['median','Q25','Q75','minimum','maximum'])

    display(distribution(subject_qc, ['mapping_mask_pct','geometry_pct','final_pct']))
    display(hemi_qc.groupby('hemisphere')[['mapping_full_pct','triangle_lookup_pct','illegal_source_ids','illegal_weights','weight_sum_error_max','reconstruction_error_max']].agg(['median','max']))

    display(hemi_qc.groupby('hemisphere')[['zero_full_pct','zero_mask_pct']].agg(['median','min','max']))
    display(distribution(subject_qc,['pearson_r','spearman_rho','mae_mm','median_difference_mm','median_length_thickness_ratio']))
    reg_rows=[]
    for row in manifest.itertuples(index=False):
        try: reg_rows.append(dict(subject_id=row.subject_id, **read_mincost(Path(row.mincost_path)), registration_error=''))
        except Exception as e: reg_rows.append(dict(subject_id=row.subject_id, registration_error=str(e)))
    registration=pd.DataFrame(reg_rows)
    if 'bbregister_mincost' not in registration: registration['bbregister_mincost']=np.nan
    v=registration.bbregister_mincost
    mad=(v-v.median()).abs().median()
    registration['robust_z']=.6745*(v-v.median())/mad if mad > 0 else np.nan
    registration['cost_review_flag']=registration.robust_z.abs() > ROBUST_Z
    manual_path=OUT/'registration_manual_review.csv'
    if manual_path.exists():
        manual=pd.read_csv(manual_path,dtype={'subject_id':str})
        assert manual.subject_id.is_unique
        registration=registration.merge(manual,on='subject_id',how='left',validate='one_to_one')
    for key,default in [('visual_qc_status','not_reviewed'),('exclusion_reason',''),('gm_wm_boundary_offset_mm',np.nan)]:
        if key not in registration: registration[key]=default
        registration[key]=registration[key].fillna(default)
    registration['excluded']=registration.visual_qc_status.eq('exclude') & registration.exclusion_reason.str.strip().ne('')
    display(registration)
    review=registration.loc[registration.cost_review_flag | registration.registration_error.ne('') | registration.visual_qc_status.eq('not_reviewed')]
    review.to_csv(TABLE_DIR/'registration_review_queue.csv',index=False)
    valid_cost=registration.dropna(subset=['bbregister_mincost']).sort_values('bbregister_mincost')
    examples={}
    if len(valid_cost):
        examples={'low_cost':valid_cost.iloc[0].subject_id,'median_cost':valid_cost.iloc[len(valid_cost)//2].subject_id,'high_cost':valid_cost.iloc[-1].subject_id}
    print('Cost-ranked examples (not visual quality labels):',examples)

    def endpoint_overlay(subject_id):
        row=manifest.set_index('subject_id').loc[subject_id]
        image=nib.load(row.md_image)
        data=np.asanyarray(image.dataobj)
        assert data.ndim == 3
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        endpoints=[]
        for h in ('lh','rh'):
            p=load_npz(row[f'profile_{h}'])
            endpoints.append(p['dwi_voxel_coordinates'][fixed_masks[h]][:, [0,-1], :])
        xyz=np.concatenate(endpoints)
        center=np.nanmedian(xyz.reshape(-1,3),axis=0).round().astype(int)
        finite=data[np.isfinite(data) & (data>0)]
        vmax=np.quantile(finite,.98) if len(finite) else 1
        for axis,ax in enumerate(axes):
            k=int(np.clip(center[axis],0,data.shape[axis]-1))
            ax.imshow(np.take(data,k,axis=axis).T,origin='lower',cmap='gray',vmin=0,vmax=vmax)
            other=[i for i in range(3) if i != axis]
            for j,label,color in [(0,'pial','tomato'),(1,'white','cyan')]:
                pts=xyz[:,j]; near=np.isfinite(pts).all(1) & (abs(pts[:,axis]-k)<.5)
                ax.scatter(pts[near,other[0]],pts[near,other[1]],s=1,c=color,label=label)
            ax.set_title(f'voxel axis {axis} = {k}'); ax.set_aspect('equal')
        axes[0].legend(markerscale=4)
        fig.suptitle(subject_id+' · MD + endpoints'); plt.tight_layout()
        save_current_figure(f'option3_registration_overlay_{subject_id}')
    for sid in dict.fromkeys(examples.values()):
        if sid in set(subject_qc.subject_id): endpoint_overlay(sid)

    display(distribution(subject_qc,['point_invalid_pct','column_invalid_pct','geometry_pct','final_pct']))
    subject_qc['coverage_review_flag']=subject_qc.final_pct < COVERAGE_REVIEW_PCT
    display(subject_qc.nsmallest(5,'final_pct'))
    # Hemisphere sample sizes are fixed; first pool to subject/depth, then summarize subjects.
    depth_subject=depth_qc.groupby(['subject_id','depth']).apply(
        lambda g: pd.Series({k:np.average(g[k],weights=[fixed_masks[h].sum() for h in g.hemisphere])
                             for k in ['oob_pct','nan_pct','nonfinite_pct','zero_pct']}), include_groups=False).reset_index()
    trajectory_cols=['endpoint_pial_error_mm','endpoint_white_error_mm','line_error_mm','equal_spacing_error_mm']
    display(hemi_qc.groupby('hemisphere')[trajectory_cols].max())

    label_rows=[]
    for row in manifest.itertuples(index=False):
        if row.subject_id not in set(subject_qc.subject_id): continue
        for h in ('lh','rh'):
            c=load_npz(row._asdict()[f'column_{h}'])
            lab,_,names=fsio.read_annot(str(Path(row.freesurfer_subject_dir)/'label'/f'{h}.aparc.annot'))
            src=c['source_vertex_ids'].astype(int)
            valid=((src>=0)&(src<len(lab))).all(1)
            tri=np.full(src.shape,-1); tri[valid]=lab[src[valid]]
            all_same=(tri[:,0]==tri[:,1])&(tri[:,1]==tri[:,2])
            # All-different triangles have no unique majority; count as disagreement.
            majority=np.where(tri[:,0]==tri[:,1],tri[:,0],np.where(tri[:,0]==tri[:,2],tri[:,0],np.where(tri[:,1]==tri[:,2],tri[:,1],-1)))
            native_names=np.array([n.decode().lower() for n in names])
            majname=np.full(len(src),'no_majority',dtype=object)
            ok=majority>=0; majname[ok]=native_names[majority[ok]]
            mask=fixed_masks[h] & valid
            mixed=~all_same & valid
            hops=template_boundary_hops(SUBJECTS_DIR,'fsaverage',h)
            label_rows.append(dict(subject_id=row.subject_id,hemisphere=h,
                same_label_pct=pct(all_same,mask),mixed_pct=pct(mixed,mask),
                majority_agreement_pct=pct(majname==c['template_label_name'].astype(str),mask),
                mixed_near_template_boundary_pct=pct(hops<=1,mask&mixed)))
    label_qc=pd.DataFrame(label_rows)
    display(label_qc.groupby('hemisphere').median(numeric_only=True))

    def spatial_maps(sid):
        row=manifest.set_index('subject_id',drop=False).loc[sid]
        for h in ('lh','rh'):
            _,_,_,m=inspect_hemi(row,h,spatial=True)
            vertices,_=fsio.read_geometry(str(TEMPLATE/'surf'/f'{h}.inflated'))
            fig,axes=plt.subplots(1,5,figsize=(16,3))
            panels=[('zero',np.ones(len(vertices),bool),'Zero: full mesh'),('zero',m['mask'],'Zero: mask'),
                    ('mapping_failed',m['mask'],'Mapping failed'),('invalid',m['mask'],'Invalid final profile'),
                    ('length_minus_thickness',m['mask'],'Length − thickness (mm)')]
            for ax,(key,scope,title) in zip(axes,panels):
                ax.scatter(
                    vertices[:,1], vertices[:,2], s=.1, c='lightgray',
                    rasterized=True,
                )
                select=scope & (np.isfinite(m[key]) if key=='length_minus_thickness' else m[key])
                artist=ax.scatter(vertices[select,1],vertices[select,2],s=.5,
                    c=m[key][select] if key=='length_minus_thickness' else 'crimson',
                    cmap='coolwarm' if key=='length_minus_thickness' else None,
                    rasterized=True)
                if key=='length_minus_thickness': fig.colorbar(artist,ax=ax,shrink=.6)
                ax.set_title(title); ax.set_aspect('equal'); ax.axis('off')
            fig.suptitle(f'{sid} {h}'); plt.tight_layout()
            save_current_figure(f'option3_spatial_qc_{sid}_{h}')
    spatial_maps(subject_qc.nsmallest(1,'final_pct').iloc[0].subject_id)

    subject_table=manifest[['subject_id']].merge(status,on='subject_id',how='left',validate='one_to_one').merge(subject_qc,on='subject_id',how='left',validate='one_to_one').merge(registration,on='subject_id',how='left',validate='one_to_one')
    display(subject_table.nsmallest(5,'final_pct'))
    display(subject_table.nlargest(5,'bbregister_mincost'))
    for metric in ['final_pct','geometry_pct','zero_mask_pct']:
        wide=hemi_qc.pivot(index='subject_id',columns='hemisphere',values=metric)
        print(metric, 'LH − RH:'); display((wide.lh-wide.rh).describe())
    if COVARIATES is not None:
        cov=pd.read_csv(COVARIATES,dtype={'subject_id':str})
        joined=subject_table.merge(cov,on='subject_id',how='left',validate='one_to_one')
        joined['invalid_fraction']=1-joined.final_pct/100
        if 'age' in joined:
            z=joined[['age','invalid_fraction']].dropna()
            if len(z)>2: print('Age association:',stats.spearmanr(z.age,z.invalid_fraction))
        for key in ['diagnosis','scan_protocol']:
            if key in joined: display(joined.groupby(key).invalid_fraction.agg(['count','median','min','max']))
    else:
        print('Covariate analysis pending: set COVARIATES to an actual subject metadata CSV.')
    for name,table in [('subject_qc',subject_table),('hemisphere_qc',hemi_qc),('depth_qc',depth_subject),('label_qc',label_qc)]:
        table.to_csv(TABLE_DIR/f'{name}.csv',index=False)
    print('Requested:',len(manifest),'successful:',len(subject_qc),'failed:',sum(status.status=='failed'),
          'review flags:',int(subject_qc.coverage_review_flag.sum()),'reviewed exclusions:',int(registration.excluded.sum()))
    print('Boundary offset measured:',int(registration.gm_wm_boundary_offset_mm.notna().sum()),'/',len(manifest))

    # Persist every presentation table so the notebook only reads artifacts.
    status.to_csv(TABLE_DIR/'status.csv', index=False)
    review.to_csv(TABLE_DIR/'registration_review_queue.csv', index=False)
    status.groupby('status').size().rename('n_subjects').reset_index().to_csv(
        TABLE_DIR/'status_counts.csv', index=False
    )
    distribution(
        subject_qc, ['mapping_mask_pct','geometry_pct','final_pct']
    ).rename_axis('statistic').reset_index().to_csv(
        TABLE_DIR/'coverage_summary.csv', index=False
    )
    distribution(
        subject_qc,
        ['pearson_r','spearman_rho','mae_mm','median_difference_mm',
         'median_length_thickness_ratio'],
    ).rename_axis('statistic').reset_index().to_csv(
        TABLE_DIR/'geometry_summary.csv', index=False
    )
    distribution(
        subject_qc,
        ['point_invalid_pct','column_invalid_pct','geometry_pct','final_pct'],
    ).rename_axis('statistic').reset_index().to_csv(
        TABLE_DIR/'sampling_summary.csv', index=False
    )
    hemi_qc.groupby('hemisphere')[trajectory_cols].max().reset_index().to_csv(
        TABLE_DIR/'trajectory_summary.csv', index=False
    )
    subject_table.nsmallest(5,'final_pct').to_csv(TABLE_DIR/'lowest_coverage_subjects.csv',index=False)
    registration.to_csv(TABLE_DIR/'registration_qc.csv',index=False)
    mapping_geometry = hemi_qc.groupby('hemisphere')[[
        'mapping_full_pct','triangle_lookup_pct','illegal_source_ids',
        'illegal_weights','weight_sum_error_max','reconstruction_error_max',
        'zero_full_pct','zero_mask_pct',
    ]].agg(['median','max'])
    mapping_geometry.columns = [
        f'{metric}_{statistic}' for metric, statistic in mapping_geometry.columns
    ]
    mapping_geometry.reset_index().to_csv(
        TABLE_DIR/'mapping_geometry_by_hemisphere.csv', index=False
    )
    for metric in ['final_pct','geometry_pct','zero_mask_pct']:
        wide=hemi_qc.pivot(index='subject_id',columns='hemisphere',values=metric)
        (wide.lh-wide.rh).describe().rename_axis('statistic').rename(
            f'{metric}_lh_minus_rh'
        ).reset_index().to_csv(
            TABLE_DIR/f'hemisphere_difference_{metric}.csv', index=False
        )
    (OUT/'run_metadata.json').write_text(json.dumps(dict(
        status='complete', manifest=str(MANIFEST), n_requested=len(manifest),
        n_successful=len(subject_qc), n_failed=int(status.status.eq('failed').sum()),
        coverage_review_pct=COVERAGE_REVIEW_PCT,
        zero_length_tolerance_mm=ZERO_MM,
        covariates=str(COVARIATES) if COVARIATES else None,
        table_directory=str(TABLE_DIR), figure_directory=str(FIGURE_DIR),
        figures=figure_names,
    ),indent=2)+'\n')
    print('QC computation complete.', flush=True)


if __name__ == '__main__':
    main()
