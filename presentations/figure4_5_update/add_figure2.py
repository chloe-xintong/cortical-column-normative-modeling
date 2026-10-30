"""Insert Figure 2 mapping-validation slides, using existing exported figures."""
from pathlib import Path
import json,hashlib,zipfile
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE,MSO_CONNECTOR
ROOT=Path(__file__).resolve().parent
F2=Path('/mnt/newStor/paros/paros_WORK/xintong/results/qc/template_registration_comparison/figure2_method_comparison_fsaverage')
r=Presentation(ROOT/'Figure4_5_Advisor_Update_EN_NotebookFigures.pptx')
old=list(r.slides._sldIdLst);new=[];records=[]
NAVY='152B45';TEAL='007F86';GRAY='617185';LIGHT='EDF3F7'
def text(s,t,x,y,w,h,size=20,color=NAVY,bold=False):
 z=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=z.text_frame;tf.word_wrap=True;tf.margin_left=tf.margin_right=0;tf.margin_top=tf.margin_bottom=0
 for i,line in enumerate(t.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='Aptos';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color);p.space_after=Pt(8)
 return z

def rect(s,x,y,w,h,color=LIGHT):
 z=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));z.fill.solid();z.fill.fore_color.rgb=RGBColor.from_string(color);z.line.fill.background()

def slide(title,sub='',note=''):
 s=r.slides.add_slide(r.slide_layouts[6]);new.append(r.slides._sldIdLst[-1]);rect(s,0,0,13.333,.1,TEAL)
 text(s,'FIGURE 2. MAPPING VALIDATION',.5,.3,12,.3,11,TEAL,True);text(s,title,.5,.78,12.3,.88,26,NAVY,True)
 text(s,sub,.52,1.7,12.1,.4,14,GRAY)
 text(s,'Source: figure2_AB_method_comparison.ipynb and its exported analysis / QC outputs.',.52,7.15,11.7,.2,9,GRAY)
 s.notes_slide.notes_text_frame.text=note
 return s

def pic(s,name,x=.65,y=2.15,w=12,h=4):
 path=F2/'figures'/name;iw,ih=Image.open(path).size;scale=min(w/iw,h/ih);ww=iw*scale;hh=ih*scale
 s.shapes.add_picture(str(path),Inches(x+(w-ww)/2),Inches(y+(h-hh)/2),width=Inches(ww),height=Inches(hh))
 records.append({'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})

def band(s,t):rect(s,.55,6.25,12.22,.57);text(s,t,.75,6.34,11.8,.44,16,TEAL,True)

s=slide('Figure 2. Mapping validation','Validate recovered depth profiles and shared T1→DWI registration before spatial and prediction analyses.',note='Figure 2A/B use 50 participants with both method comparisons available (100 hemisphere records per comparison). C uses the 327-participant registration-QC manifest. These are different denominators. Figure 3 has not yet been performed and is omitted. O3 is a comparison reference, not biological ground truth. All metrics use canonical pial-to-white 21-depth orientation.')
for i,(title,body) in enumerate([
 ('O2-Column','Sample MD along native columns first; combine profile values using barycentric weights.'),('O2-FS','FreeSurfer volume-to-surface depth sampling, followed by spherical surface resampling.'),('O3','Interpolate native pial/white endpoints first; construct and sample the target column.')]):
 x=.65+i*4.2;rect(s,x,2.4,3.9,2.7);text(s,title,x+.18,2.65,3.5,.4,23,TEAL,True);text(s,body,x+.18,3.25,3.5,1.6,20)
text(s,'A/B: same 50 participants • 21 depths • pairwise valid anatomical mask\nC: 327 registration-QC participants • shared bbregister transform',.8,5.4,11.8,.7,18)

s=slide('Figure 2A. O2-Column versus O3: agreement on fsaverage','Per-vertex Pearson r across 21 depths; Fisher-z average across participants; inflated surfaces.',note='Exact exported surface map. Display criterion: defined correlations in at least 80% of the 50 participants. Cortex.label ∩ valid DK cortical labels; pairwise complete-profile validity. Constant profiles are excluded from Fisher averaging. Shared color range .8–1.0. Surface summary is not the subject-level median. Subject-level agreement: median .996497, IQR .995349–.997169.')
pic(s,'Figure2A_O2Column_vs_O3_inflated.png');band(s,'O2-Column closely preserves the O3 depth-profile shape; agreement is not proof of absolute accuracy.')

s=slide('Figure 2A. O2-FS versus O3: agreement on fsaverage','Same participants, depth direction, analysis mask, display threshold, and color range.',note='Exact exported O2FS surface map. The subject-median r has group median .870392, IQR .845704–.902817. Correlation alone does not quantify MD magnitude differences. Between-method discrepancies cannot by themselves identify which method is biologically correct.')
pic(s,'Figure2A_O2FS_vs_O3_inflated.png');band(s,'O2-FS shows lower profile-shape agreement with O3 than O2-Column does.')

s=slide('Figure 2A. Quantitative agreement across participants','One point per participant: median across pairwise-valid bilateral vertices (n = 50 per comparison).',note='Exact notebook-exported panels. O2Column r=.996497 [.995349,.997169], sNMAE=1.957145% [1.807125,2.396821]. O2FS r=.870392 [.845704,.902817], sNMAE=16.557108% [15.408177,18.793826]. Brackets here are subject IQRs. sNMAE=200*MAE/(median|A|+median|B|). Pairwise mean valid coverage 98.297% and 98.706%, respectively. Three-method common-mask sensitivity yields near-identical subject-median results.')
text(s,'O2-Column vs O3',.85,2.22,5.65,.35,20,TEAL,True);text(s,'O2-FS vs O3',7,2.22,5.65,.35,20,TEAL,True)
pic(s,'Figure2A2_O2Column_vs_O3_subject_level_profile_agreement.png',x=.6,y=2.8,w=6.0,h=2.8)
pic(s,'Figure2A2_O2FS_vs_O3_subject_level_profile_agreement.png',x=6.75,y=2.8,w=6,h=2.8)
band(s,'Median r / sNMAE: O2-Column 0.9965 / 1.96%; O2-FS 0.8704 / 16.56%.')

s=slide('Figure 2B. Profile difference versus cortical curvature','Ten subject-specific equal-frequency bins; group median and between-participant IQR.',note='x is barycentrically interpolated absolute native white-surface mean curvature, sum(w_i*abs(H_i)). y is mean absolute difference over 21 depths at a vertex. Within each subject, bilateral vertices are binned and summarized by median MAE; group aggregation weights subjects equally. Within-subject Spearman rho medians are .019080 and .039205 with bootstrap CIs [.007333,.025661] and [.030822,.046250]. These are small associations despite low Wilcoxon p-values. Do not attribute differences solely to interpolation order; O2FS also differs in projection/column construction.')
text(s,'O2-Column vs O3',.85,2.22,5.65,.35,20,TEAL,True);text(s,'O2-FS vs O3',7,2.22,5.65,.35,20,TEAL,True)
pic(s,'Figure2B_O2Column_vs_O3_MAE_vs_curvature.png',x=.6,y=2.75,w=6,h=2.9)
pic(s,'Figure2B_O2FS_vs_O3_MAE_vs_curvature.png',x=6.75,y=2.75,w=6,h=2.9)
band(s,'Differences increase modestly across curvature bins; participant-level associations are small (ρ≈.019 / .039).')

s=slide('Figure 2C. T1→DWI registration quality across participants','Shared boundary-based registration: 327 participants; 22 automatic flags; 5 confirmed visual exclusions.',note='Use the latest exported Figure2C_T1_DWI_registration_cost.png, not the older notebook-cache plot reporting zero exclusions. Latest tables: median cost .123435, IQR .106595–.160340; all327 final transforms and costs readable. Five reviewed failures: D0128,D0163,D0248,D0296,D0388. All five are present in the historical Figure4 325 manifest; the updated retained Figure4 cohort has 320. This does not imply 327−5=320: the registration-QC and complete-profile cohorts differ. Cost is dimensionless and not a displacement in mm. Robust z=3.5 is a review threshold, not automatic exclusion. The tutorial .5 reference appearing in the exported panel is not the cohort decision threshold. Other flagged participants are not necessarily visually cleared.')
pic(s,'Figure2C_T1_DWI_registration_cost.png');band(s,'Boundary-based cost is a QC score; final exclusion is based on visual review, not cost alone.')

s=slide('Figure 2D. Registration QC examples','Reserved for one visually acceptable case and one flagged case.',note='QC images intentionally left empty at the user’s request. Insert matched T1/DWI reference views and identical surface overlays later. Good/flagged labels are layout placeholders and do not assign status to any participant.')
for xx,label in [(.75,'Good alignment'),(6.9,'Flagged alignment')]:
 rect(s,xx,2.55,5.65,3.3);text(s,label,xx+.2,2.8,5.2,.5,23,TEAL,True);text(s,'QC image placeholder',xx+.2,4.0,5.2,.5,20,GRAY)

s=slide('From mapping validation to spatial organization','Figure 2 establishes method agreement and QC; Figure 4 asks what spatial detail is worth retaining.',note='Figure 3 is not yet performed. Transition directly to Figure4. Interpret Figure2 agreement as internal method comparison rather than validation against an external ground truth. Use the 320-participant registration-QC-passing cohort for updated Figure4 summaries and the unchanged 196-participant CN cohort for Figure5.')
for i,t in enumerate(['Mapping: O2-Column and O3 yield highly consistent profiles; O2-FS differs more.','QC: remove the five confirmed registration failures from the Figure 4 cohort.','Next: characterize spatial redundancy (Figure 4), then assess age-prediction value (Figure 5).']):
 text(s,'•',.7,2.5+i*1.0,.25,.4,22,TEAL,True);text(s,t,1.05,2.5+i*1.0,11.6,.8,22)
band(s,'Updated analysis scope: Figure 4 n = 320; Figure 5 n = 196 CN participants.')

# Insert immediately after the title slide; preserve all original slides and relationships.
for el in new:r.slides._sldIdLst.remove(el)
for j,el in enumerate(new,1):r.slides._sldIdLst.insert(j,el)
# Cover now includes the newly added mapping section.
for sh in r.slides[0].shapes:
 if sh.has_text_frame:
  for para in sh.text_frame.paragraphs:
   for run in para.runs:
    run.text=run.text.replace('RESEARCH PROGRESS | FIGURES 4–5','RESEARCH PROGRESS | FIGURES 2, 4–5').replace('From local profile similarity to the value of finer spatial sampling','From mapping validation to spatial organization and age prediction')
for sh in r.slides[0].shapes:
 if sh.has_text_frame and sh.text.startswith('Figure 4  |'):
  sh.text='Figure 2  |  Mapping validation and registration QC\nFigure 4  |  Spatial similarity and correlation length\nFigure 5  |  Representation contributions to age prediction'
  for pp in sh.text_frame.paragraphs:pp.font.name='Aptos';pp.font.size=Pt(21);pp.font.color.rgb=RGBColor.from_string(NAVY)
# Number all slides after insertion.
for i,s in enumerate(r.slides,1):
 found=False
 for sh in s.shapes:
  if sh.has_text_frame and sh.left/Inches(1)>12 and sh.top/Inches(1)>7:
   sh.text=str(i);found=True
   for pp in sh.text_frame.paragraphs:pp.font.size=Pt(11);pp.font.color.rgb=RGBColor.from_string(GRAY)
 if not found:text(s,str(i),12.25,7.11,.5,.25,11,GRAY)
path=ROOT/'Figure2_4_5_Advisor_Update_EN.pptx';r.save(path)
with zipfile.ZipFile(path) as z:
 assert z.testzip() is None
 hashes={hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('ppt/media/')}
 assert all(a['sha256'] in hashes for a in records)
(ROOT/'figure2_slide_sources.json').write_text(json.dumps(records,indent=2))
print(f'Saved {path}; {len(r.slides)} slides; 8 Figure 2 / transition slides inserted.')
