"""Insert an extra slide with editable scale markers over the unchanged notebook image."""
from pathlib import Path
import numpy as np,pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.enum.shapes import MSO_SHAPE,MSO_CONNECTOR
from pptx.dml.color import RGBColor
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
source=ROOT/'Figure2_4_5_Advisor_Update_EN_NotebookOnly.pptx';r=Presentation(source)
image=REPO/'analysis_results/figure4/outputs/registration_qc_pass_320/real/cohort/Figure4A_ico5_K15_with_relative_MAE.png'
a=pd.read_csv(REPO/'analysis_results/figure4/outputs/registration_qc_pass_320/real/cohort/figure4A_ico5_k15_real_group_summary.csv')
b=pd.read_csv(REPO/'analysis_results/figure4/outputs/registration_qc_pass_320/relative_mae/cohort/figure4_relative_mae_group_summary.csv')
s=r.slides.add_slide(r.slide_layouts[6])
def text(t,x,y,w,h,size=18,c='152B45',bold=False):
 z=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));z.text=t;z.text_frame.word_wrap=True
 for pp in z.text_frame.paragraphs:pp.font.name='Aptos';pp.font.size=Pt(size);pp.font.bold=bold;pp.font.color.rgb=RGBColor.from_string(c)
def line(x1,y1,x2,y2,c):
 z=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2));z.line.color.rgb=RGBColor.from_string(c);z.line.width=Pt(1.2)
text('FIGURE 4 | ADDITIONAL ANNOTATED VIEW',.5,.3,12,.3,11,'007F86',True)
text('Figure 4A: sampling scales on the original notebook figure',.5,.78,12.3,.8,26,bold=True)
text('Colored points locate median nearest-neighbor spacing; the original curves and IQR bands are unchanged.',.52,1.7,12.1,.4,14,'617185')
iw,ih=Image.open(image).size;w=11.2;h=w*ih/iw;px=1.06;py=2.15
s.shapes.add_picture(str(image),Inches(px),Inches(py),width=Inches(w),height=Inches(h))
# Pixel calibration from original axes; points are editable overlays, image bytes are unchanged.
dark=(np.array(Image.open(image).convert('RGB'))[:,:,:3]<80).all(2)
cols=np.where(dark.sum(0)>ih*.65)[0];clusters=np.split(cols,np.where(np.diff(cols)>1)[0]+1);lefts=[float(np.median(q)) for q in clusters]
rows=np.where(dark.sum(1)>iw*.5)[0];bottom=float(np.median(rows));top=float(np.where(dark[:,int(lefts[0])])[0].min())
# Infer right endpoints from contiguous black bottom-axis segments.
xs=np.where(dark[int(bottom)])[0];runs=np.split(xs,np.where(np.diff(xs)>1)[0]+1);runs=[q for q in runs if len(q)>1000];ends=[float(q.max()) for q in runs]
X=lambda pix:px+pix/iw*w
Y=lambda pix:py+pix/ih*h
spacing=[('ico5',2.120595,'007F86'),('ico4',4.235486,'C97532'),('ico3',8.512779,'7455A3')]
for j,(med,lo,hi) in enumerate([(a.profile_dissimilarity_median.to_numpy(),a.profile_dissimilarity_q25.to_numpy(),a.profile_dissimilarity_q75.to_numpy()),(b['median'].to_numpy(),b.q25.to_numpy(),b.q75.to_numpy())]):
 mn=float(lo.min());mx=float(hi.max());yl=mn-.05*(mx-mn);yh=mx+.05*(mx-mn)
 for k,(name,d,c) in enumerate(spacing):
  v=float(np.interp(d,np.arange(1,30,2),med));xx=X(lefts[j]+(ends[j]-lefts[j])*d/30);yy=Y(bottom-(v-yl)/(yh-yl)*(bottom-top))
  z=s.shapes.add_shape(MSO_SHAPE.OVAL,Inches(xx-.055),Inches(yy-.055),Inches(.11),Inches(.11));z.fill.solid();z.fill.fore_color.rgb=RGBColor.from_string(c);z.line.color.rgb=RGBColor(255,255,255);z.line.width=Pt(1)
  # Label in the clear space below each curve with a leader back to its marker.
  tx=X(lefts[j]+(ends[j]-lefts[j])*(.31+.20*k));ty=Y(bottom-(bottom-top)*(.10+.12*k))
  line(xx,yy,tx+.12,ty+.1,c);text(name,tx,ty,.65,.3,13,c,True)
for k,(name,d,c) in enumerate(spacing):text(f'{name}: {d:.2f} mm',1.4+k*3.8,6.55,3.3,.4,19,c,True)
text('Original notebook image with editable annotation overlays. Marker heights are interpolated on the binned curves.',.52,7.14,11.7,.25,9,'617185')
s.notes_slide.notes_text_frame.text='Additional annotated view, explicitly requested. The PNG is embedded byte-for-byte from the active 320-participant Figure4 notebook. Only colored markers, labels and leader lines are added as native editable PowerPoint objects. The marker x values are retained-center median nearest-neighbor geodesic distances; y values are linearly interpolated from binned group medians for illustrative positioning. The original unannotated slide is retained.'
el=r.slides._sldIdLst[-1];r.slides._sldIdLst.remove(el);r.slides._sldIdLst.insert(15,el)
for i,sl in enumerate(r.slides,1):
 found=False
 for sh in sl.shapes:
  if sh.has_text_frame and sh.left/914400>12 and sh.top/914400>7:
   sh.text=str(i);found=True
 if not found:
  z=sl.shapes.add_textbox(Inches(12.25),Inches(7.11),Inches(.5),Inches(.25));z.text=str(i)
  for pp in z.text_frame.paragraphs:pp.font.size=Pt(11)
out=ROOT.parent/'Advisor_Update_Figure4_Annotated.pptx';r.save(out);r.save(ROOT.parent/'Advisor_Update.pptx')
print(f'Saved {out.resolve()}; {len(r.slides)} slides; extra annotated Figure4A is slide16; original slide15 retained.')
