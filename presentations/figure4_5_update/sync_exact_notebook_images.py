"""Use exact embedded notebook outputs throughout; remove duplicate Figure 4A slide."""
from pathlib import Path
import json,base64,hashlib,io,zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from PIL import Image
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
F2=REPO/'analysis_results/figure2/figure2_AB_method_comparison.ipynb'
F4=REPO/'analysis_results/figure4/notebooks/figure4_formal_ico5_k15.ipynb'
F5=REPO/'analysis_results/figure5/notebooks/plot_figure5.ipynb'
# Refresh only the original registration plotting cell from current saved tables.
# This runs the notebook's exact plotting code, rather than creating a new plot design.
n=json.loads(F2.read_text());rr=Path('/mnt/newStor/paros/paros_WORK/xintong/results/qc/template_registration_comparison/figure2_method_comparison_fsaverage')
ns={'np':np,'pd':pd,'plt':plt,'registration_costs':pd.read_csv(rr/'tables/figure2C_registration_cost_by_subject.csv'),'registration_summary':pd.read_csv(rr/'tables/figure2C_registration_cost_summary.csv')}
plt.show=lambda:None
exec(''.join(n['cells'][18]['source']),ns)
b=io.BytesIO();ns['fig'].savefig(b,format='png',dpi=180,bbox_inches='tight');plt.close(ns['fig'])
n['cells'][18]['outputs']=[{'output_type':'display_data','metadata':{},'data':{'image/png':base64.b64encode(b.getvalue()).decode(),'text/plain':['<Figure: notebook registration-QC panel>']}}]
# Refresh the corresponding summary-table output to match the plotted current QC state.
n['cells'][16]['outputs']=[{'output_type':'display_data','metadata':{},'data':{'text/plain':[ns['registration_summary'].to_string(index=False)]}}]
F2.write_text(json.dumps(n,ensure_ascii=False,indent=1))
notebooks={str(p):json.loads(p.read_text()) for p in [F2,F4,F5]}
assets=ROOT/'notebook_output_images';assets.mkdir(exist_ok=True)
def extract(path,cell,index=0):
 outs=[o for o in notebooks[str(path)]['cells'][cell].get('outputs',[]) if 'image/png' in o.get('data',{})];raw=outs[index]['data']['image/png'];raw=''.join(raw) if isinstance(raw,list) else raw
 data=base64.b64decode(raw);dest=assets/f'{path.stem}_cell{cell}_output{index}.png';dest.write_bytes(data)
 return dest,{'notebook':str(path.relative_to(REPO)),'cell':cell,'image_output_index':index,'sha256':hashlib.sha256(data).hexdigest()}
def pic(s,path,x=.65,y=2.12,w=12,h=4.04):
 iw,ih=Image.open(path).size;scale=min(w/iw,h/ih);ww,hh=iw*scale,ih*scale
 s.shapes.add_picture(str(path),Inches(x+(w-ww)/2),Inches(y+(h-hh)/2),width=Inches(ww),height=Inches(hh))
p=ROOT/'Figure2_4_5_Advisor_Update_EN.pptx';r=Presentation(p);records=[]
# Slide indices refer to the existing 30-slide presentation before deletion.
mapping={3:[(F2,8,0)],4:[(F2,8,1)],5:[(F2,10,0),(F2,10,1)],6:[(F2,13,0),(F2,13,1)],7:[(F2,18,0)],15:[(F4,3,0)],17:[(F4,5,0)],18:[(F4,7,0)],19:[(F4,9,0)],22:[(F5,14,0)],23:[(F5,16,0)],24:[(F5,16,1)],25:[(F4,3,0)]}
for num,items in mapping.items():
 s=r.slides[num-1]
 for sh in list(s.shapes):
  if sh.shape_type==13:sh._element.getparent().remove(sh._element)
 for k,(nb,cell,idx) in enumerate(items):
  path,record=extract(nb,cell,idx)
  if len(items)==2:pic(s,path,x=.6+6.15*k,y=2.75,w=6,h=2.9)
  elif num in [22,23,24]:pic(s,path,y=2.09,h=3.85)
  else:pic(s,path)
  record['original_slide']=num;records.append(record)
 s.notes_slide.notes_text_frame.text+='\nExact notebook-embedded image output used; no image redrawing or extra annotations.'
 if num==25:
  s.notes_slide.notes_text_frame.text='Notebook Figure 4A image reproduced exactly. Sampling distances can be discussed verbally: ico3 8.51 mm, ico4 4.24 mm, ico5 2.12 mm. No scale markers have been added to the original notebook image.'
  for sh in s.shapes:
   if sh.has_text_frame and sh.text.startswith('Markers indicate'):
    sh.text='Notebook Figure 4A: spatial profile differences across geodesic distances.'
    for pp in sh.text_frame.paragraphs:pp.font.size=Pt(14);pp.font.color.rgb=RGBColor.from_string('617185')
# Synchronize current notebook FDR values and provenance; benchmark remains historical.
repls={'q = 0.053':'q = 0.065','q = 0.227':'q = 0.238','q = 0.161':'q = 0.214','q=.053':'q=.065','q=.227':'q=.238','q=.161':'q=.214','original FDR 0.05 threshold':'exploratory FDR 0.05 threshold','original FDR families':'current exploratory FDR families','original multiplicity families: seven representation tests and three ico4 geometry tests':'current exploratory families: three spatial-resolution and four ico4-ablation tests','original seven-test BH q=.052728':'exploratory three-test BH q=.064794'}
for s in r.slides:
 for sh in s.shapes:
  groups=[sh.text_frame] if sh.has_text_frame else []
  if sh.has_table:groups += [c.text_frame for row in sh.table.rows for c in row.cells]
  for tf in groups:
   for para in tf.paragraphs:
    for run in para.runs:
     for a,b in repls.items():run.text=run.text.replace(a,b)
# Specific captions for current ablation family, replacing the outdated unadjusted comparison label.
for sh in r.slides[23].shapes:
 if sh.has_text_frame and sh.text.startswith('Mean → depth:'):
  sh.text='Mean → depth: exploratory q = 0.237   |   Depth → +length: exploratory q = 0.214'
  for pp in sh.text_frame.paragraphs:pp.font.size=Pt(15);pp.font.color.rgb=RGBColor.from_string('263648')
# Statistical appendix: current notebook families, while keeping the historical benchmark separate.
s=r.slides[27]
for sh in s.shapes:
 if sh.has_table:
  t=sh.table
  updates={1:('0.443','[0.150, 0.739]','Historical benchmark q=.018'),2:('0.348','[0.057, 0.665]','Exploratory q=.065; 3 tests'),3:('−0.073','[−0.175, 0.027]','Exploratory q=.238; 3 tests'),4:('0.150','[−0.058, 0.364]','Exploratory q=.237; 4 tests'),5:('0.057','[−0.0003, 0.1154]','Exploratory q=.214; 4 tests')}
  for row,vals in updates.items():
   for col,val in enumerate(vals,1):
    t.cell(row,col).text=val
    for pp in t.cell(row,col).text_frame.paragraphs:pp.font.name='Aptos';pp.font.size=Pt(14);pp.font.color.rgb=RGBColor.from_string('263648')
 if sh.has_text_frame and sh.text.startswith('Use original FDR'):
  sh.text='Current notebook: exploratory FDR families of 3 spatial and 4 ico4-ablation contrasts.'
  for pp in sh.text_frame.paragraphs:pp.font.size=Pt(14);pp.font.color.rgb=RGBColor.from_string('617185')
s.notes_slide.notes_text_frame.text='Current Figure 5 notebook uses post hoc exploratory/secondary BH-FDR families: spatial resolution (3 contrasts) and ico4 ablation (4 contrasts). Benchmark DK mean-to-depth contrast is retained separately as a historical benchmark and is not part of the current 3+4 families. See notebook cell12 output for the authoritative current table.'
# Delete the redundant second Figure 4A slide (old16), preserving the combined shape + magnitude panel on old15.
el=r.slides._sldIdLst[15];r.part.drop_rel(el.rId);r.slides._sldIdLst.remove(el)
for i,s in enumerate(r.slides,1):
 for sh in s.shapes:
  if sh.has_text_frame and sh.left/914400>12 and sh.top/914400>7:
   sh.text=str(i)
   for pp in sh.text_frame.paragraphs:pp.font.size=Pt(11);pp.font.color.rgb=RGBColor.from_string('617185')
for rec in records:rec['slide']=rec['original_slide']-(rec['original_slide']>16)
# Remove orphaned image relationships left by replacing picture elements.
for slide in r.slides:
 used=set(slide._element.xpath('//@r:embed'))
 for rel in list(slide.part.rels.values()):
  if rel.reltype.endswith('/image') and rel.rId not in used:slide.part.drop_rel(rel.rId)
r.save(p);alt=ROOT/'Figure2_4_5_Advisor_Update_EN_NotebookOnly.pptx';r.save(alt)
with zipfile.ZipFile(p) as z:
 assert z.testzip() is None
 hashes={hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('ppt/media/')}
 assert all(rec['sha256'] in hashes for rec in records)
 # Every remaining embedded image must be one of the explicitly audited notebook outputs.
 assert hashes <= {rec['sha256'] for rec in records}, hashes-{rec['sha256'] for rec in records}
(ROOT/'exact_notebook_image_audit.json').write_text(json.dumps(records,indent=2))
print(f'Updated {p}: {len(r.slides)} slides. Every embedded image verified against notebook output. Old25→24; duplicate old16 removed.')
