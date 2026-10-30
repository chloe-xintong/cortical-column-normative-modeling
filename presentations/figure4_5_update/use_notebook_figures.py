"""Replace reconstructed result plots with the exact notebook-exported PNGs."""
from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
import json, hashlib, zipfile
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
F4=REPO/'analysis_results/figure4/outputs/registration_qc_pass_320'
F5=REPO/'analysis_results/figure5/outputs/figures'
p=ROOT/'Figure4_5_Advisor_Update_EN.pptx'
r=Presentation(p)

def clear(slide,keep_table=False):
 for sh in list(slide.shapes):
  top=sh.top/914400
  if 1.85<=top<6.24 and not (keep_table and sh.has_table):
   sh._element.getparent().remove(sh._element)

def picture(s,path,x=.65,y=2.12,w=12.0,h=4.04):
 iw,ih=Image.open(path).size; scale=min(w/iw,h/ih);ww=iw*scale;hh=ih*scale
 return s.shapes.add_picture(str(path), Inches(x+(w-ww)/2), Inches(y+(h-hh)/2),width=Inches(ww),height=Inches(hh))

def text(s,t,x,y,w,h,size=16):
 z=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=z.text_frame;tf.word_wrap=True;tf.margin_left=tf.margin_right=0
 z.text=t
 for pp in tf.paragraphs:
  pp.font.name='Aptos';pp.font.size=Pt(size);pp.font.color.rgb=RGBColor.from_string('263648')

figs={7:F4/'real/cohort/Figure4A_ico5_K15_with_relative_MAE.png',8:F4/'real/cohort/Figure4A_ico5_K15_with_relative_MAE.png',9:F4/'acquisition_null_fwhm0_seedfix/cohort/Figure4B_ico5_K15_formal_main.png',10:F4/'correlation_length/cohort/Figure4C_ico5_K15_correlation_length.png',11:F4/'boundary_matched/formal/cohort/Figure4D_boundary_matched_0_10.png',14:F5/'Figure5A_MAE.png',15:F5/'Figure5B_spatial_resolution_MAE.png',16:F5/'Figure5C_ico4_features_MAE.png',17:F4/'real/cohort/Figure4A_ico5_K15_with_relative_MAE.png'}
records=[]
for num,path in figs.items():
 s=r.slides[num-1];clear(s,keep_table=num==17)
 if num==17:
  picture(s,path,x=.6,y=2.2,w=6.35,h=2.9)
  text(s,'Spatial profile scale: λ = 2.87 [2.68–3.04] mm\nCompare this continuous spatial structure with sampling spacing and OOF MAE.',.8,5.15,6.15,.9,17)
  text(s,'Fitted λ: 2.87 [2.68–3.04] mm',7.2,5.0,5.35,.7,20)
 elif num in [14,15,16]:
  picture(s,path,y=2.09,h=3.85)
  annotation={14:'DK mean → depth: MAE reduction 0.443 y; original FDR q = 0.018.',15:'DK depth → ico4: q = 0.053   |   ico4 vs ico5: q = 0.227.',16:'Mean → depth: exploratory p = 0.162   |   Depth → +length: q = 0.161.'}[num]
  text(s,annotation,.8,5.96,11.9,.28,15)
 else:
  picture(s,path)
 s.notes_slide.notes_text_frame.text+='\n\nDisplayed figure: exact notebook-exported PNG, without redrawing or modification. Source: '+str(path)
 records.append({'slide':num,'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
# Keep descriptions of editability accurate throughout the deck.
for s in r.slides:
 for sh in s.shapes:
  if sh.has_text_frame:
   for para in sh.text_frame.paragraphs:
    for run in para.runs:
     if 'All slide text, tables, plotted curves, axes, points, and intervals are editable PowerPoint objects.' in run.text:
      run.text='Text and tables are editable; result panels use the exact notebook-exported figures.'
 for phrase in ['All slide text, tables, plotted curves, axes, points, and intervals are editable PowerPoint objects.']:
  if phrase in s.notes_slide.notes_text_frame.text:s.notes_slide.notes_text_frame.text=s.notes_slide.notes_text_frame.text.replace(phrase,'Text and tables are editable; result panels are notebook-exported images.')
# Original title remains meaningful; slide 14 now situates its contrast in the complete overview.
s=r.slides[13]
for sh in s.shapes:
 if sh.has_text_frame and sh.text.startswith('A regional baseline establishes'):
  sh.text='Notebook overview of all 11 representations; the DK mean-to-depth contrast provides the baseline.'
  for pp in sh.text_frame.paragraphs:pp.font.size=Pt(14);pp.font.color.rgb=RGBColor.from_string('617185')
new=ROOT/'Figure4_5_Advisor_Update_EN_NotebookFigures.pptx';r.save(new)
# Update the original deliverable too, so its existing link also resolves to the corrected version.
r.save(p)
(ROOT/'notebook_figure_sources.json').write_text(json.dumps(records,indent=2))
# Verify that every exact source PNG was embedded and the package is intact.
with zipfile.ZipFile(new) as z:
 assert z.testzip() is None
 hashes={hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('ppt/media/')}
 assert all(d['sha256'] in hashes for d in records)
print(f'Saved {new}; {len(records)} result panels replaced; all source-image hashes verified.')
