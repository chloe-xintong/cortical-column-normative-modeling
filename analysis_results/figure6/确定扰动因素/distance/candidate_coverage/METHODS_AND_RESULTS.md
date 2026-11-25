# Ico4 within/across DK candidate coverage

## Methods

The cohort is the formal 313-person manifest (`/mnt/newStor/paros/paros_WORK/xintong/code/cortical_trajectory_analysis/manifests/figure4_coverage95_pass_313_subjects.csv`). Ico4 anchors are the 4,687 template-mask nodes in `/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs/template/ico4_vertex_order.csv`; neighbors remain all 163,842 vertices per hemisphere. Distances are shortest paths on the full-resolution fsaverage white mesh, in the right-closed bins (0,2], ..., (28,30] mm. The formal cortex mask and aparc/DK labels exclude medial wall/unknown/corpus-callosum vertices. Hemispheres are never combined as spatial neighborhoods.

`N_geometry` uses the template mask only. `N_available` additionally requires the formal clean-profile validity mask for both anchor and neighbor; invalid anchors are zero in the all-subject-anchor denominator and excluded from the valid-anchor denominator. No candidate set was truncated to K, no distance matching was performed, and no correlations/nulls/models were computed.

## Configuration difference from formal Figure 4

The formal notebook uses ico5 anchors. This requested pilot intentionally changes anchors to ico4 while retaining the formal full-resolution neighbors, surface, bins, labels, masks, and 313-person cohort.

## N_retained status

Not computed. The located formal fixed table is ico5 and within-DK only, so it is not compatible with this ico4 within/across coverage analysis. It was neither reused nor replaced.

## Result guide

Exact values are in the CSV files. Across-DK fractions reaching 15 candidates, comparing template geometry with subject availability among valid anchors:

```text
distance_bin  geometry  available_valid_anchor    loss
       (0,2]    0.0134                  0.0135 -0.0001
       (2,4]    0.2063                  0.2060  0.0004
       (4,6]    0.3734                  0.3729  0.0005
       (6,8]    0.5097                  0.5097  0.0000
      (8,10]    0.6311                  0.6304  0.0007
     (10,12]    0.7301                  0.7295  0.0006
     (12,14]    0.8127                  0.8117  0.0010
     (14,16]    0.8731                  0.8720  0.0011
     (16,18]    0.9223                  0.9215  0.0008
     (18,20]    0.9573                  0.9569  0.0005
     (20,22]    0.9757                  0.9754  0.0003
     (22,24]    0.9870                  0.9868  0.0002
     (24,26]    0.9942                  0.9942  0.0001
     (26,28]    0.9968                  0.9968  0.0000
     (28,30]    0.9989                  0.9989  0.0000
```

The five distance bins with the highest subject-level joint within/across K=15 coverage are:

```text
distance_bin  proportion_both_ge_15
     (24,26]                 0.9716
     (22,24]                 0.9713
     (20,22]                 0.9667
     (26,28]                 0.9633
     (28,30]                 0.9537
```

Regions with the highest mean across-bin subject-level across K=15 coverage are:

```text
hemisphere         anchor_dk_region  mean_across_ge_15
        lh              frontalpole             0.9179
        rh              frontalpole             0.8761
        rh            parsorbitalis             0.8627
        rh                 bankssts             0.8574
        rh             temporalpole             0.8559
        rh       transversetemporal             0.8533
        lh       transversetemporal             0.8510
        lh             temporalpole             0.8482
        lh            parsorbitalis             0.8436
        rh rostralanteriorcingulate             0.8381
        lh                   cuneus             0.8332
        lh               entorhinal             0.8330
```

These are coverage diagnostics only. They do not establish that K=15 is statistically stable; that question requires the planned local-feature stability analysis.

Subjects: 313. Template anchor-bins: 70305.
