"""Create a spacing-annotated Figure 4A shape panel and replace slide 17 only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
F4=REPO/'analysis_results/figure4/outputs/registration_qc_pass_320'
data=pd.read_csv(F4/'real/cohort/figure4A_ico5_k15_real_group_summary.csv').sort_values('distance_bin_index')
spacing=pd.read_csv(REPO/'analysis_results/figure5/outputs/spatial_resolution/ico_nearest_neighbor_summary.csv')
spacing=spacing[spacing.hemisphere.eq('bilateral_pooled')].set_index('level')
# Match the notebook's Figure 4A plotting convention: 2-mm bin centers.
x=data.distance_bin_index.to_numpy()*2+1
median=data.profile_dissimilarity_median.to_numpy()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelsize':14,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(11.8,4.7))
ax.fill_between(x,data.profile_dissimilarity_q25.to_numpy(),data.profile_dissimilarity_q75.to_numpy(),color='#367CB7',alpha=.18,label='Participant IQR',linewidth=0)
ax.plot(x,median,'o-',color='#367CB7',lw=2.2,ms=4,label='Group median (n = 320)')
colors={'ico5':'#147F83','ico4':'#C97532','ico3':'#7455A3'}
positions={'ico5':(4.1,.025),'ico4':(11.2,.060),'ico3':(20.3,.096)}
rows=[]
for name in ['ico5','ico4','ico3']:
 d=float(spacing.loc[name,'median_nn_mm']);v=float(np.interp(d,x,median));c=colors[name]
 ax.vlines(d,0,v,color=c,linestyle=(0,(3,3)),lw=1.3,alpha=.75)
 ax.scatter([d],[v],s=105,c=c,edgecolor='white',linewidth=1.4,zorder=6)
 ax.annotate(f'{name}: {d:.2f} mm',xy=(d,v),xytext=positions[name],color=c,fontsize=14,fontweight='bold',ha='left',va='center',arrowprops={'arrowstyle':'-','color':c,'lw':1.25,'connectionstyle':'angle3,angleA=0,angleB=90'})
 rows.append({'representation':name,'nearest_neighbor_median_mm':d,'interpolated_dissimilarity':v})
ax.set(xlim=(0,30),ylim=(0,.20),xticks=np.arange(0,31,5),yticks=np.arange(0,.201,.05),xlabel='White-surface geodesic distance (mm)',ylabel='Profile-shape dissimilarity (1 − r)')
ax.grid(axis='y',alpha=.22);ax.legend(loc='lower right',frameon=False,fontsize=12)
fig.subplots_adjust(left=.09,right=.985,bottom=.17,top=.96)
for ext in ['png','pdf','svg']:fig.savefig(ROOT/f'Figure4A_sampling_scales_annotated.{ext}',dpi=300,facecolor='white')
plt.close(fig)
pd.DataFrame(rows).to_csv(ROOT/'Figure4A_sampling_scale_annotations.csv',index=False)
for name in ['Figure4_5_Advisor_Update_EN.pptx','Figure4_5_Advisor_Update_EN_NotebookFigures.pptx']:
 p=ROOT/name;r=Presentation(p);s=r.slides[16]
 for sh in list(s.shapes):
  # Preserve header, subtitle, page number and source footer only.
  if 2.08<=sh.top/914400<7.04:sh._element.getparent().remove(sh._element)
 for sh in s.shapes:
  if sh.has_text_frame and sh.text.startswith('Sampling spacing and correlation length'):
   sh.text='Markers indicate median nearest-neighbor spacing of retained ico3 / ico4 / ico5 centers.'
   from pptx.util import Pt
   from pptx.dml.color import RGBColor
   for para in sh.text_frame.paragraphs:
    para.font.name='Aptos';para.font.size=Pt(14);para.font.color.rgb=RGBColor.from_string('617185')
 s.shapes.add_picture(str(ROOT/'Figure4A_sampling_scales_annotated.png'),Inches(.65),Inches(2.15),width=Inches(12.0))
 s.notes_slide.notes_text_frame.text='Only the Figure 4A profile-shape panel is shown. Reproduced from the exact formal notebook source table using the notebook bin-center convention (1,3,...,29 mm). The blue curve and participant IQR use the saved group summary. Colored markers locate the median nearest-neighbor geodesic spacing of retained template centers: ico5 2.12 mm, ico4 4.24 mm, ico3 8.51 mm. Marker y positions are linearly interpolated on the binned group-median curve for illustration; they are not new empirical estimates or model-performance points. Spacings refer to template centers, not participants. This annotation does not identify an optimal sampling threshold. Standalone PNG, PDF, SVG and plotting code accompany the deck.'
 r.save(p)
print('Updated slide 17 in both decks; exported annotated PNG/PDF/SVG and annotation data.')
