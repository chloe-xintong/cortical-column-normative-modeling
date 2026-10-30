"""Build an editable academic PowerPoint from frozen Figure 4/5 tables.
Run with PYTHONPATH=/tmp/figure45_pptx python build_slides.py.
All slide text, tables, axes, curves and error bars are native PowerPoint objects.
"""
from pathlib import Path
import json, math, sys
import pandas as pd
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR

REPO=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
F4=REPO/'analysis_results/figure4/outputs/registration_qc_pass_320'
F5=Path('/mnt/newStor/paros/paros_WORK/xintong/results/figure5_age_prediction/outputs')
NAVY='152B45'; TEAL='007F86'; BLUE='367CB7'; ORANGE='C97532'; GRAY='617185'; LIGHT='EDF3F7'; INK='263648'; WHITE='FFFFFF'; RED='A34D4D'
prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5)
manifest=[]; sources=set()
def csv(p):
 p=Path(p); sources.add(str(p)); return pd.read_csv(p)
def txt(s,x,y,w,h,text,size=20,color=INK,bold=False):
 box=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); tf=box.text_frame; tf.word_wrap=True
 tf.margin_left=tf.margin_right=Inches(.025); tf.margin_top=tf.margin_bottom=0
 for i,line in enumerate(str(text).split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.text=line; p.font.name='Aptos'; p.font.size=Pt(size); p.font.bold=bold; p.font.color.rgb=RGBColor.from_string(color); p.space_after=Pt(10)
 return box

def line(s,x1,y1,x2,y2,color=GRAY,width=1.4):
 z=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2)); z.line.color.rgb=RGBColor.from_string(color); z.line.width=Pt(width);return z

def rect(s,x,y,w,h,color=LIGHT):
 z=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));z.fill.solid();z.fill.fore_color.rgb=RGBColor.from_string(color);z.line.fill.background();return z

def dot(s,x,y,color=TEAL,r=.045):
 z=s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x-r), Inches(y-r), Inches(2*r), Inches(2*r));z.fill.solid();z.fill.fore_color.rgb=RGBColor.from_string(color);z.line.fill.background()

def slide(title,section='FIGURES 4–5',subtitle=None,notes='',source=''):
 s=prs.slides.add_slide(prs.slide_layouts[6]);rect(s,0,0,13.333,.10,TEAL)
 txt(s,.5,.30,12,.3,section,11,TEAL,True);txt(s,.5,.78,12.3,.88,title,26,NAVY,True)
 if subtitle: txt(s,.52,1.70,12.1,.37,subtitle,14,GRAY)
 line(s,.5,7.05,12.83,7.05,'D7E0E8',.8);txt(s,.52,7.15,11.7,.2,source or 'Cortical MD profiles | Registration-QC cohort update | 320 participants',9,GRAY)
 txt(s,12.25,7.11,.5,.25,str(len(prs.slides)),11,GRAY)
 s.notes_slide.notes_text_frame.text=notes+ ('\n\nData provenance: '+source if source else '')
 manifest.append({'slide':len(prs.slides),'title':title,'notes':notes,'source':source});return s

def bullets(s,items,x=.65,y=2.1,w=12,size=22,gap=.9):
 for i,item in enumerate(items):
  txt(s,x,y+i*gap,.25,.4,'•',size,TEAL,True);txt(s,x+.35,y+i*gap,w-.35,gap-.1,item,size)

def takeaway(s,text,y=6.25):
 rect(s,.55,y,12.22,.57,LIGHT);txt(s,.75,y+.09,11.8,.45,text,16,TEAL,True)

def table(s,headers,rows,x=.65,y=2.12,w=12.0,h=3.6,size=18,widths=None):
 t=s.shapes.add_table(len(rows)+1,len(headers), Inches(x), Inches(y), Inches(w), Inches(h)).table
 if widths:
  for c,f in zip(t.columns,widths): c.width=Inches(w*f)
 for r,row in enumerate([headers]+list(rows)):
  for c,v in enumerate(row):
   z=t.cell(r,c);z.text=str(v);z.fill.solid();z.fill.fore_color.rgb=RGBColor.from_string(NAVY if r==0 else (LIGHT if r%2 else WHITE))
   z.margin_left=Inches(.12);z.margin_right=Inches(.10);z.margin_top=Inches(.10);z.margin_bottom=Inches(.04)
   for p in z.text_frame.paragraphs:
    p.font.name='Aptos';p.font.size=Pt(size);p.font.bold=r==0;p.font.color.rgb=RGBColor.from_string(WHITE if r==0 else INK)
 return t

def curve(s,xvals,series,x=.95,y=2.35,w=7.5,h=3.4,xlim=(0,30),ylim=(0,.2),xlabel='Geodesic distance (mm)',ylabel='Profile dissimilarity (1 − r)',xticks=None,yticks=None):
 X=lambda v:x+(v-xlim[0])/(xlim[1]-xlim[0])*w
 Y=lambda v:y+h-(v-ylim[0])/(ylim[1]-ylim[0])*h
 for v in (yticks if yticks is not None else np.linspace(*ylim,5)):
  line(s,x,Y(v),x+w,Y(v),'E2E8EE',.7);txt(s,x-.65,Y(v)-.12,.57,.25,f'{v:.3g}',11,GRAY)
 for v in (xticks if xticks is not None else np.linspace(*xlim,7)):
  txt(s,X(v)-.22,y+h+.10,.6,.25,f'{v:g}',11,GRAY)
 line(s,x,y+h,x+w,y+h,NAVY,1);line(s,x,y,x,y+h,NAVY,1)
 txt(s,x,y-.47,w,.35,ylabel,14,NAVY);txt(s,x+1.5,y+h+.47,w-2,.3,xlabel,13,GRAY)
 for vals,color,label,width in series:
  for j in range(len(xvals)-1):line(s,X(xvals[j]),Y(vals[j]),X(xvals[j+1]),Y(vals[j+1]),color,width)
  if label:
   k=sum(bool(q[2]) for q in series[:series.index((vals,color,label,width))]) if False else 0
 return X,Y

def forest(s,keys,title=None,x=3.35,y=2.6,w=5.8,h=2.5,limits=(4.4,7.2),ci=True):
 for v in np.arange(4.5,7.21,.5):
  xx=x+(v-limits[0])/(limits[1]-limits[0])*w;line(s,xx,y-.2,xx,y+h,'DFE6EC',.7);txt(s,xx-.18,y+h+.12,.5,.25,f'{v:.1f}',11,GRAY)
 for i,k in enumerate(keys):
  r=metrics.loc[k]; yy=y+i*h/max(len(keys)-1,1); X=lambda v:x+(v-limits[0])/(limits[1]-limits[0])*w
  txt(s,.65,yy-.15,2.6,.57,labels[k],15 if len(labels[k])>22 else 17,NAVY)
  if ci:
   a=cis[(cis.representation==k)&(cis.metric=='mae')].iloc[0];line(s,X(a.ci95_lower),yy,X(a.ci95_upper),yy,GRAY,2);line(s,X(a.ci95_lower),yy-.07,X(a.ci95_lower),yy+.07,GRAY);line(s,X(a.ci95_upper),yy-.07,X(a.ci95_upper),yy+.07,GRAY)
  dot(s,X(r.mae),yy,TEAL if k=='ico4_md' else BLUE,.065);txt(s,x+w+.28,yy-.17,1.05,.35,f'{r.mae:.3f}',18,NAVY,True)
 txt(s,x+1.25,y+h+.48,w,.3,'OOF MAE (years); 95% subject-bootstrap CI',12,GRAY)

real=csv(F4/'real/cohort/figure4A_ico5_k15_real_group_summary.csv')
rel=csv(F4/'relative_mae/cohort/figure4_relative_mae_group_summary.csv')
bd=csv(F4/'boundary_matched/formal/cohort/figure4D_group_summary.csv')
metrics=csv(REPO/'analysis_results/figure5/outputs/figures/Figure5_all_metrics.csv').set_index('representation')
cis=csv(REPO/'analysis_results/figure5/outputs/figures/Figure5_all_metrics_bootstrap_ci.csv')
labels={k:r['model'].replace('Regional','DK').replace('21-depth profile','depth profile') for k,r in metrics.iterrows()}
meta=csv(REPO/'analysis_results/figure5/cohort_selection/outputs/cohort_manifest.csv')
ids4=csv(REPO/'manifests/figure4_registration_qc_pass_320_subjects.csv');ids5=csv(F5/'features/subject_order.csv')
d4=meta[meta.subject_id.isin(ids4.subject_id)];d5=meta[meta.subject_id.isin(ids5.subject_id)]

s=slide('Spatial organization and age-prediction value\nof cortical column profiles','RESEARCH PROGRESS | FIGURES 4–5',notes='Advisor discussion deck. All results are from the latest saved formal Figure 4 and LSQR Figure 5 outputs. These analyses describe spatial profile organization and prediction under a fixed model; they do not establish independent biological units or a unique optimal scale.')
txt(s,.7,2.7,11.8,.8,'From local profile similarity to the value of finer spatial sampling',26,TEAL)
txt(s,.72,4.05,11.6,1.2,'Figure 4  |  Spatial similarity, sampling effects, and correlation length\nFigure 5  |  Depth, spatial density, and geometry in brain-age prediction',22)
txt(s,.72,6.1,10,.45,'Advisor meeting • 28 October 2026',17,GRAY)

s=slide('Two linked questions guide the analysis',subtitle='A finer representation is useful only if its additional spatial detail supports the modeling objective.')
rect(s,.7,2.35,5.8,2.6);rect(s,6.8,2.35,5.8,2.6)
txt(s,.95,2.6,5.3,.4,'Q1  Spatial organization',23,TEAL,True);txt(s,.95,3.27,5.1,1.4,'How do column-profile differences vary with cortical distance and DK boundaries?',24)
txt(s,7.05,2.6,5.1,.4,'Q2  Predictive value',23,TEAL,True);txt(s,7.05,3.27,5.1,1.4,'Does retaining depth and finer spatial sampling improve age prediction?',24)
takeaway(s,'Spatial correlation describes redundancy; held-out prediction tests practical value.')

s=slide('Cohort definition: 320 spatial profiles, 196 CN prediction participants',subtitle='The prediction cohort is a subset of the spatial-analysis cohort.',source='Sources: cohort_manifest.csv; exclusion_summary.csv; feature_build_report.json',notes='Metadata: 448; imaging available: 369; complete profiles: 325; remove five visually confirmed registration failures: 320; date eligible: 268; CN: 200; ≥95% coverage: 196. Original and updated cohort files are preserved separately. Counts are sequential, not independent exclusions.')
steps=[('448','Metadata'),('369','Imaging matched'),('325','Complete profiles'),('320','Figure 4: QC pass'),('268','Date eligible'),('200','CN eligible'),('196','Figure 5')]
for i,(n,l) in enumerate(steps):
 x=.6+i*1.82;rect(s,x,2.5,1.65,1.55,LIGHT);txt(s,x+.1,2.65,1.45,.6,n,31,TEAL,True);txt(s,x+.1,3.35,1.48,.6,l,14,NAVY)
bullets(s,['Date eligibility: HIGH / MEDIUM match, or prespecified LOW exception with multiple consistently CN visits.','Prediction inclusion: CN status, manual T1–DWI registration QC, and ≥95% valid-column coverage.'],y=4.55,size=20,gap=.8)

s=slide('Sample composition and age coverage',subtitle='Figure 4 includes diagnostic heterogeneity; Figure 5 trains and evaluates age prediction in CN participants only.',source='Source: cohort_manifest.csv joined to the Figure 4 and final Figure 5 subject lists.',notes='Diagnosis labels are harmonized at the QIAL-matched clinical visit; ambiguous/LOW matches occur in Figure 4. Age is VISIT_AGE at that visit, not an independently verified scan-date age. Sex codes remain numeric because their codebook was not verified; no male/female interpretation is assumed.')
rows=[]
for label in ['Normal','MCI','AD','Other/Impaired non-demented','Unknown']:
 rows.append([label,str((d4.diagnosis_at_scan==label).sum()),str((d5.diagnosis_at_scan==label).sum())])
table(s,['Matched-visit diagnosis','Fig. 4','Fig. 5'],rows,w=7.1,h=3.6,size=18,widths=[.66,.17,.17])
txt(s,8.15,2.3,4.45,2.1,'Age, mean ± SD (range)\nFigure 4: 57.8 ± 11.8 (28–80) y\nFigure 5: 58.1 ± 10.5 (28–78) y',20)
txt(s,8.15,4.55,4.4,1.25,'Age source: clinical VISIT_AGE at the QIAL-matched visit.',18,GRAY)
takeaway(s,'Each participant contributes one merged out-of-fold age prediction in Figure 5.')

s=slide('From full fsaverage to the analyzed column representations',subtitle='A column is a computational depth trajectory at a template vertex, not a histological cortical column.',source='Sources: build_features.py; feature_build_report.json; reference profile masks; spacing summary.',notes='Full fsaverage has 163,842 vertices per hemisphere (327,684 bilateral). Reference in_analysis_mask has 149,953 left +149,926 right =299,879 positions. This is a fixed template analysis-mask count, not the valid count for every participant. Subject-specific validity varies. The masks retain 1,175/4,687/18,742 ico3/4/5 centers, pooled across hemispheres. Official ico mesh totals before mask: 1,284/5,124/20,484. Depth fractions are 0, .05, ..., 1, directed pial to white.')
table(s,['Representation','Bilateral positions','MD features / subject'],[
 ['Full fsaverage mesh','327,684','327,684 × 21 grid'],['Template analysis mask','299,879','Subject-specific validity applied'],['DK regional profile','68 regions','1,428'],['ico3 / ico4 / ico5','1,175 / 4,687 / 18,742','24,675 / 98,427 / 393,582']],h=3.6,size=19,widths=[.34,.28,.38])
takeaway(s,'Each retained profile samples MD at 21 depths: 0–100%, in 5% steps, pial → white.')

s=slide('Figure 4 methods: distance-dependent profile differences',subtitle='Formal A–C: 320 participants; official ico5 anchors; dense within-DK candidates.',source='Sources: formal notebook; formal_pair_qc.json; Figure 4A group tables.',notes='Pair list contains 4,182,592 directed pairs across bins. K=15 is a target maximum per anchor-bin; sparse bins can contain fewer candidates. Self pairs and duplicates are excluded. Distances are mesh-edge geodesics on fsaverage white. Aggregation for A: pair metric → anchor × 2-mm-bin median → bilateral subject median → group median/IQR. Mean profile shape is preserved in raw r, so high far-distance r does not indicate independent profiles.')
bullets(s,['18,742 bilateral anchors; up to 15 candidates per anchor in each 2-mm bin, from 0–30 mm.','Distance: fsaverage white-surface mesh-edge geodesic distance.','Shape: D = 1 − Pearson r across the 21-depth MD profile.','Magnitude: symmetric relative MAE = 200 × MAE(x,y) / (median|x| + median|y|).'],y=2.1,size=21,gap=.82)
takeaway(s,'Summarize within participant first; group curves show median and participant-level IQR.')

s=slide('Figure 4A: nearby columns are more similar',subtitle='Within-DK profile-shape dissimilarity increases rapidly at short distances, then levels off.',source='Source: figure4A_ico5_k15_real_group_summary.csv.',notes='Native PowerPoint line segments reproduce the saved median and IQR curves. IQR describes between-participant dispersion, not the confidence interval of the median. The reference plateau is 20–30 mm; it is an approximate reference, not a claim of perfect flatness.')
x=real.geodesic_distance_mm_median.to_numpy();med=real.profile_dissimilarity_median.to_numpy();lo=real.profile_dissimilarity_q25.to_numpy();hi=real.profile_dissimilarity_q75.to_numpy()
curve(s,x,[(lo,'B8C7D4','',1),(hi,'B8C7D4','',1),(med,TEAL,'median',3)],ylim=(0,.20))
txt(s,9.0,2.55,3.6,2.5,'0–2 mm: D = 0.052\n8–10 mm: D = 0.138\n28–30 mm: D = 0.147',22,NAVY)
txt(s,9.0,5.0,3.6,.7,'Teal: median\nLight lines: IQR',16,GRAY)
takeaway(s,'Local profile-shape similarity is spatially structured; high correlation does not imply identical MD.')

s=slide('Figure 4A: correlated profiles still differ in MD magnitude',subtitle='Relative profile MAE provides a complementary view of numerical differences.',source='Source: figure4_relative_mae_group_summary.csv.',notes='Symmetric relative MAE is normalized by the sum of the two median absolute MD values. It is not the conventional per-depth MAPE and should not be described as such. Shape and magnitude metrics answer different questions.')
curve(s,np.arange(1,30,2),[(rel.q25.to_numpy(),'B8C7D4','',1),(rel.q75.to_numpy(),'B8C7D4','',1),(rel['median'].to_numpy(),BLUE,'median',3)],ylim=(0,26),ylabel='Symmetric relative profile MAE (%)')
txt(s,9.0,2.55,3.6,2.5,'0–2 mm: 10.31%\n8–10 mm: 19.90%\n28–30 mm: 21.24%',22,NAVY)
takeaway(s,'Spatial redundancy in profile shape coexists with measurable numerical variation.')

s=slide('Figure 4B: acquisition and sampling explain part of local similarity',subtitle='Primary null: native-DWI Gaussian noise, FWHM = 0 mm, 100 realizations per participant.',source='Sources: formal Figure 4B notebook and seed-fixed acquisition-null cohort tables.',notes='Generate independent Gaussian noise on each native DWI grid, sample at real O3 coordinates with trilinear interpolation, then recolor to match subject-specific depth mean and 21×21 covariance. FWHM=0 means no additional Gaussian PSF, not an absence of interpolation-induced smoothness. Local excess L(d)=D_far−D(d), far reference 20–30 mm. Residual=L_real−L_null. AUC is not a proportion of biological information. Nonzero PSF sensitivity from older analyses is not merged into this formal primary result.')
for i,(a,b) in enumerate([('Native DWI grid','Independent Gaussian fields'),('Real O3 sampling','Trilinear interpolation'),('Depth matching','Mean + 21 × 21 covariance')]):
 xx=.7+4.2*i;rect(s,xx,2.3,3.95,1.4);txt(s,xx+.15,2.5,3.65,.4,a,21,TEAL,True);txt(s,xx+.15,3.05,3.65,.5,b,17)
txt(s,.8,4.1,11.8,.6,'Residual local-excess AUC, 0–10 mm: 0.3554 [95% CI 0.3478–0.3683]',24,NAVY,True)
txt(s,.8,5.05,11.7,.65,'Positive in 320/320 participants; median residual local excess across 0–10 mm ≈ 0.0355.',20)
takeaway(s,'Residual structure exceeds this interpolation-only null; it is not a purely physiological estimate.')

s=slide('Figure 4C: a stable millimeter-scale summary of spatial similarity',subtitle='Exponential model: D(d) = D∞ − A exp(−d / λ).',source='Sources: figure4C_full_fit_summary.csv; figure4C_fixed_split_reliability.csv.',notes='λ is the e-folding length of fitted excess similarity, not a hard radius of independence. Fixed anchor split is stratified by hemisphere × DK region and reused across all bins. This is anchor split-half stability, not scan-rescan reliability. Full data: all 320 fits acceptable. Half fits: 320 paired acceptable; no retained fit hits a parameter boundary. Acceptance: R² ≥ .5 and no boundary hit.')
rect(s,.7,2.35,5.4,2.7);txt(s,1.0,2.65,4.8,.6,'λ = 2.87 mm',36,TEAL,True);txt(s,1.0,3.65,4.8,.85,'IQR 2.68–3.04 mm\nFull-data median fit R² = 0.9967',20)
table(s,['Anchor split-half measure','Result'],[['Pearson r / ICC','0.895 / 0.895'],['Spearman ρ','0.855'],['Median |Δλ|','0.103 mm'],['Acceptable paired fits','320 / 320']],x=6.5,y=2.35,w=6.1,h=2.7,size=18,widths=[.65,.35])
takeaway(s,'The fitted scale is stable to anchor splitting; λ does not define a hard sampling threshold.')

s=slide('Figure 4D: no consistent positive DK-boundary shape effect',subtitle='Same-anchor within/across candidates matched one-to-one; geodesic mismatch ≤0.1 mm.',source='Sources: formal boundary_matched cohort group and 0–10-mm summary tables.',notes='ΔD=(1−r_across)−(1−r_within). Aggregate matched-pair mean → anchor-bin mean → bilateral subject-region-bin mean → equal-region subject-bin mean → group median/IQR. Effect is conditional on boundary-eligible anchors and distance matching. Curve uses varying eligible anchors across bins. It is not directly comparable to A’s all-anchor within-region medians. Earlier positive-boundary conclusions are superseded. A small negative aggregate does not establish that DK anatomy is invalid or that cross-region profiles are intrinsically more alike.')
X,Y=curve(s,np.arange(1,10,2),[(bd.delta_d_q25.to_numpy(),'B8C7D4','',1),(bd.delta_d_q75.to_numpy(),'B8C7D4','',1),(bd.delta_d_median.to_numpy(),TEAL,'',3)],xlim=(0,10),ylim=(-.012,.009),xticks=[0,2,4,6,8,10],ylabel='ΔD = D across − D within',yticks=[-.01,-.005,0,.005])
line(s,X(0),Y(0),X(10),Y(0),GRAY,1.3)
txt(s,9.0,2.4,3.65,2.95,'0–10 mm median ΔD\n−0.00242\n95% CI\n[−0.00303, −0.00166]\nPositive: 111/320 (34.7%)',20,NAVY)
takeaway(s,'DK boundaries do not consistently separate more dissimilar local profile shapes under this design.')

s=slide('Figure 5 methods: a controlled representation comparison',subtitle='Same 196 CN participants; shared nested folds; fixed LSQR Ridge framework.',source='Sources: run_ridge.py; saved fold definitions; submission_lsqr_20261028.json.',notes='Outer 5-fold CV, inner 5-fold CV, stratified by age quantile bins. Median imputation and StandardScaler fit within each training fold. Tune Ridge alpha over logspace(−4,6,21) using inner MAE; evaluate held-out outer folds. Metrics are recomputed from merged OOF predictions. Subject-bootstrap intervals condition on the saved OOF predictions; they do not include model retraining or resampling of the full CV pipeline. Age is from the matched clinical visit. Use fixed LSQR outputs consistently.')
bullets(s,['Outer 5-fold CV estimates prediction error; inner 5-fold CV selects regularization by MAE.','Median imputation and feature standardization are fitted within training folds.','Ridge α: 21 logarithmically spaced values from 10⁻⁴ to 10⁶.','Primary metric: merged out-of-fold MAE; r, RMSE, and R² are secondary.'],size=21,gap=.83)
takeaway(s,'95% CIs: 10,000 subject-level bootstrap resamples of saved OOF predictions.')

s=slide('Feature comparisons separate depth, spatial density, and geometry',subtitle='All feature dimensions below are bilateral and are measured per participant.',source='Sources: feature_build_report.json; column-mean and geometry build reports.',notes='Mean MD collapses depth within each column; regional mean MD also averages spatially within each DK region. Full ico4 depth features: 4,687×21=98,427. Adding length or curvature adds one scalar per center, 4,687 features each. Length is trajectory length; it is not automatically identical to cortical thickness. Native curvature is mapped using stored barycentric weights.')
table(s,['Scientific comparison','Representations','Feature count'],[
 ['Depth at DK scale','Mean MD → 21-depth profile','68 → 1,428'],['Spatial sampling with depth','DK → ico3 → ico4 → ico5','1,428 → 24,675 → 98,427 → 393,582'],['Depth at ico4 scale','Column mean → full profile','4,687 → 98,427'],['Geometry at ico4 scale','Full profile + length / curvature','103,114 each; 107,801 combined']],h=3.75,size=18,widths=[.29,.35,.36])
takeaway(s,'Spatial resolution and feature dimensionality change together; interpret gains under the fixed model.')

s=slide('Depth profiles improve prediction over DK mean MD',subtitle='A regional baseline establishes whether depth-dependent information is useful.',source='Sources: Figure5_all_metrics.csv; original predefined_paired_bootstrap.csv.',notes='Original seven-test comparison family: regional_mean_md−regional_profile ΔMAE=.442738, 95% CI [.149596,.739245], p=.002600, BH q=.018198. These are the original correction-family statistics, not the redefined family in the newer exploratory table.')
forest(s,['regional_mean_md','regional_profile'],y=2.7,h=1.2)
txt(s,.85,5.05,11.8,.6,'MAE reduction: 0.443 years [95% CI 0.150–0.739]; FDR q = 0.018',23,NAVY,True)
takeaway(s,'At the DK scale, preserving depth provides supported predictive benefit.')

s=slide('Spatial density: ico4 improves numerically; ico5 adds no observed gain',subtitle='The spatial sequence is not a monotonic improvement with denser sampling.',source='Sources: Figure5_all_metrics.csv; Figure5_displayed_MAE_statistics.csv.',notes='DK profile−ico4 ΔMAE=.347735; original bootstrap CI [.057035,.652521], unadjusted p=.019798; original seven-test BH q=.052728. ico4−ico5 ΔMAE=−.072997; CI [−.177822,.029289], q=.226777. A CI excluding zero combined with q>.05 is possible because CI is unadjusted and q is multiplicity adjusted. No equivalence or noninferiority claim is made.')
forest(s,['regional_profile','ico3_md','ico4_md','ico5_md'],y=2.45,h=2.25)
txt(s,.85,5.55,11.7,.55,'DK → ico4: ΔMAE = −0.348 y, q = 0.053   |   ico4 vs ico5: q = 0.227',20,NAVY,True)
takeaway(s,'The ico4 spatial gain is suggestive; it does not meet the original FDR 0.05 threshold.')

s=slide('At ico4, depth and geometry yield modest additional changes',subtitle='Column mean, full depth, and geometry extensions address distinct information contributions.',source='Sources: latest Figure 5 metrics and displayed paired statistics.',notes='Column mean vs full depth added after reviewing results: exploratory unadjusted p=.161984, ΔMAE=.149993, CI [−.057946,.364215]. Full depth vs +length ΔMAE=.057147, CI [−.000321,.115386], original three-test geometry q=.160784. Curvature and combined geometry do not show supported additional improvements. Lowest numerical MAE should not be called statistically best.')
forest(s,['ico4_column_mean_md','ico4_md','ico4_md_length','ico4_md_curvature','ico4_md_length_curvature'],y=2.4,h=2.5)
txt(s,.85,5.65,11.75,.4,'Mean → depth: exploratory p = 0.162   |   Depth → +length: q = 0.161',20,NAVY,True)
takeaway(s,'ico4 + length has the lowest numerical MAE (5.233 y); extra-feature benefits remain uncertain.')

s=slide('Integrating spatial similarity with sampling density',subtitle='Sampling spacing and correlation length share units, but describe different quantities.',source='Sources: formal Figure 4A/C; ico_nearest_neighbor_summary.csv; Figure 5 OOF metrics.',notes='Nearest-neighbor distances are pooled over retained template centers in both hemispheres, not over participants. Bands in the table are spatial IQRs. All use fsaverage white mesh-edge geodesics. Vertical lines show median center spacing on the Figure 4A curve; they do not represent a direct mapping of λ onto the prediction curve. λ is e-folding of profile excess similarity, not a theoretical optimum. No ico3-vs-ico4 significance is asserted here.')
X,Y=curve(s,x,[(med,TEAL,'',3)],x=.95,y=2.55,w=6.0,h=2.8,xlim=(0,15),ylim=(0,.17),xticks=[0,3,6,9,12,15],yticks=[0,.05,.1,.15]) if False else (None,None)
# Restrict plotted samples to the displayed distance range.
keep=x<=15
X,Y=curve(s,x[keep],[(med[keep],TEAL,'',3)],x=.95,y=2.55,w=5.65,h=2.8,xlim=(0,15),ylim=(0,.17),xticks=[0,3,6,9,12,15],yticks=[0,.05,.1,.15])
for v,label,c in [(2.120595,'ico5',BLUE),(4.235486,'ico4',ORANGE),(8.512779,'ico3',GRAY)]:
 line(s,X(v),Y(0),X(v),Y(.17),c,1.3);txt(s,X(v)-.15,2.32,.65,.23,label,12,c,True)
table(s,['Scale','NN median [IQR], mm','MAE, y'],[['ico3','8.51 [7.59–9.58]','5.687'],['ico4','4.24 [3.77–4.76]','5.291'],['ico5','2.12 [1.87–2.40]','5.364']],x=7.15,y=2.35,w=5.5,h=2.35,size=16,widths=[.2,.59,.21])
txt(s,7.2,5.0,5.35,.7,'Fitted λ: 2.87 [2.68–3.04] mm',20,TEAL,True)
takeaway(s,'Intermediate-density sampling may balance local spatial detail and redundancy in this setting.')

s=slide('Interpretation and questions for discussion',subtitle='The strongest evidence concerns depth information; the optimal spatial density remains open.',notes='Suggested advisor discussion: whether to prioritize ico4 as a pragmatic working representation, whether the formal DK-boundary result merits a main or supplementary panel, and whether further validation should focus on repeated CV/external generalization rather than additional model ranking. Do not present planned analyses as completed.')
bullets(s,['Established: local profiles show distance-dependent similarity; DK depth profiles outperform DK mean MD.','Suggestive: ico4 provides a favorable numerical balance of prediction error and feature dimension.','Unresolved: added depth at ico4 and geometry yield modest changes with uncertain incremental benefit.','Discussion: use ico4 as the working representation, and prioritize generalization and sensitivity checks?'],size=21,gap=.87)
takeaway(s,'A practical working scale is supported more clearly than a unique biological or predictive optimum.')

s=slide('Appendix | Complete prediction metrics',section='APPENDIX A',subtitle='All 11 representations; same 196 merged OOF predictions.',source='Source: Figure5_all_metrics.csv; metric CIs available in companion CSV.')
rows=[]
for k,r in metrics.sort_values('mae').iterrows():rows.append([labels[k],f'{r.mae:.3f}',f'{r.rmse:.3f}',f'{r.pearson_r:.3f}',f'{r.r2:.3f}'])
table(s,['Representation','MAE, y','RMSE, y','Pearson r','R²'],rows,y=2.03,h=4.7,size=14,widths=[.48,.13,.13,.13,.13])

s=slide('Appendix | Statistical contrasts and correction families',section='APPENDIX B',subtitle='Use original FDR families; the added column-mean comparison remains exploratory.',source='Sources: original representation / geometry paired-bootstrap tables; Figure5_displayed_MAE_statistics.csv.',notes='Δ is MAE(A)−MAE(B); positive favors B. Intervals are pointwise unadjusted percentile 95% intervals. Existing formal contrasts retain original multiplicity families: seven representation tests and three ico4 geometry tests. Column mean versus depth is exploratory and receives no q value. Fixed-OOF bootstrap does not quantify all training-set/CV variability.')
table(s,['A − B','ΔMAE, y','95% CI','Statistical status'],[
 ['DK mean − DK depth','0.443','[0.150, 0.739]','q=.018; 7-test family'],['DK depth − ico4','0.348','[0.057, 0.653]','q=.053; 7-test family'],['ico4 − ico5','−0.073','[−0.178, 0.029]','q=.227; 7-test family'],['ico4 mean − depth','0.150','[−0.058, 0.364]','Exploratory p=.162'],['ico4 − ico4 + length','0.057','[−0.0003, 0.1154]','q=.161; 3-test family']],h=3.7,size=16,widths=[.33,.13,.24,.30])
takeaway(s,'No formal all-model pairwise ranking or equivalence claim is made.')

s=slide('Appendix | Cohort and metadata qualifications',section='APPENDIX C',source='Sources: cohort_manifest.csv; cohort_selection/README.md; feature_build_report.json.')
table(s,['Detail','Figure 4 (n=320)','Figure 5 (n=196)'],[
 ['HIGH / MEDIUM date match','105 / 144','71 / 107'],['LOW / ambiguous date match','45 / 26','18 / 0'],['Sex code 1 / 2 / 9','117 / 202 / 1','65 / 130 / 1'],['Age source','Matched-visit VISIT_AGE','Matched-visit VISIT_AGE']],h=2.9,size=18,widths=[.42,.29,.29])
txt(s,.8,5.35,11.8,.8,'LOW matches in Figure 5 require multiple distinct clinical dates with consistently CN status. Sex codes are reported as stored; the codebook was not verified.',18,GRAY)

s=slide('Appendix | Analysis provenance and scope',section='APPENDIX D',subtitle='Formal outputs take precedence over historical notebook results.',notes='Full absolute paths are included in the companion data_sources.json and per-slide speaker notes. This presentation changes no analysis outputs and does not retrain models.')
bullets(s,['Figure 4: figure4_formal_ico5_k15.ipynb; frozen within-DK A–C and matched-boundary D.','Figure 5: plot_figure5.ipynb; fixed-LSQR outputs; 11-representation OOF summaries.','Spacing: ico_nearest_neighbor_summary.csv; fsaverage white-surface mesh-edge geodesics.','Limitations: mixed diagnosis in Figure 4, clinical-visit age proxy, high feature dimension, and internal validation only.'],size=21,gap=.85)
takeaway(s,'All slide text, tables, plotted curves, axes, points, and intervals are editable PowerPoint objects.')

path=OUT/'Figure4_5_Advisor_Update_EN.pptx';prs.save(path)
(OUT/'slide_notes.txt').write_text('\n\n'.join(f"Slide {d['slide']}: {d['title']}\n{d['notes']}\n{d['source']}" for d in manifest))
(OUT/'data_sources.json').write_text(json.dumps({'review_date':'2026-10-28','sources':sorted(sources),'figure4_root':str(F4),'figure5_root':str(F5),'slides':manifest},indent=2))
# Validate the package, text presence, and shape bounds.
import zipfile
from lxml import etree
with zipfile.ZipFile(path) as z:
 assert z.testzip() is None
 for name in z.namelist():
  if name.endswith('.xml'):etree.fromstring(z.read(name))
issues=[]
for i,s in enumerate(prs.slides,1):
 for sh in s.shapes:
  if sh.left<0 or sh.top<0 or sh.left+sh.width>prs.slide_width+1000 or sh.top+sh.height>prs.slide_height+1000:issues.append((i,sh.name))
assert not issues,issues
print(json.dumps({'output':str(path),'slides':len(prs.slides),'shapes':sum(len(s.shapes) for s in prs.slides),'bounds_check':'pass','xml_check':'pass'}))
