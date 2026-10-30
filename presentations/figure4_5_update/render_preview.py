from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
import math
import sys
import io
ROOT=Path(__file__).resolve().parent
r=Presentation(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'Figure4_5_Advisor_Update_EN.pptx')
S=110; W=round(13.333*S);H=round(7.5*S)
FONT='/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf'
import subprocess
FONT=subprocess.check_output(['fc-match','-f','%{file}','Arial']).decode()
BOLD=subprocess.check_output(['fc-match','-f','%{file}','Arial:style=Bold']).decode()
def co(c,default='#FFFFFF'):
 try:return '#'+str(c.rgb)
 except:return default

def text(draw,tf,x,y,w,h):
 yy=y
 for p in tf.paragraphs:
  try: sz=p.font.size.pt if p.font.size else 18
  except:sz=18
  font=ImageFont.truetype(BOLD if p.font.bold else FONT,round(sz*S/72))
  color=co(p.font.color,'#263648')
  words=p.text.split();line='';lines=[]
  for wd in words:
   trial=(line+' '+wd).strip()
   if draw.textlength(trial,font=font)>w and line:lines.append(line);line=wd
   else:line=trial
  lines.append(line)
  for line in lines:
   draw.text((x,yy),line,font=font,fill=color);yy+=sz*S/72*1.14
  yy+=8*S/72

for i,s in enumerate(r.slides,1):
 im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im)
 for z in s.shapes:
  x,y,w,h=[int(v/914400*S) for v in [z.left,z.top,z.width,z.height]]
  if z.shape_type==MSO_SHAPE_TYPE.PICTURE:
   pic=Image.open(io.BytesIO(z.image.blob)).convert('RGB');pic=pic.resize((w,h),Image.Resampling.LANCZOS);im.paste(pic,(x,y))
  elif z.shape_type==MSO_SHAPE_TYPE.LINE:
   d.line((x,y+h if z._element.spPr.xfrm.flipV else y,x+w,y if z._element.spPr.xfrm.flipV else y+h),fill=co(z.line.color,'#617185'),width=max(1,int((z.line.width.pt if z.line.width else 1)*S/72)))
  elif z.has_table:
   t=z.table;yy=y
   for ri,row in enumerate(t.rows):
    xx=x;hh=int(row.height/914400*S)
    for ci,c in enumerate(row.cells):
     ww=int(t.columns[ci].width/914400*S);d.rectangle((xx,yy,xx+ww,yy+hh),fill=co(c.fill.fore_color),outline='white')
     text(d,c.text_frame,xx+12,yy+10,ww-22,hh-14);xx+=ww
    yy+=hh
  elif z.has_text_frame:
   try:
    if z.fill.type: d.rectangle((x,y,x+w,y+h),fill=co(z.fill.fore_color))
   except:pass
   text(d,z.text_frame,x+3,y,w-6,h)
 im.save(ROOT/f'preview_{i:02}.png')
thumbw=400;thumbh=225;sheet=Image.new('RGB',(thumbw*4, (math.ceil(len(r.slides)/4))*(thumbh+28)),'#D9E1E8');d=ImageDraw.Draw(sheet)
for i in range(len(r.slides)):
 im=Image.open(ROOT/f'preview_{i+1:02}.png');im.thumbnail((thumbw,thumbh));xx=(i%4)*thumbw;yy=(i//4)*(thumbh+28);sheet.paste(im,(xx,yy));d.text((xx+10,yy+thumbh+5),f'Slide {i+1}',fill='#152B45')
sheet.save(ROOT/'slide_overview.png')
print(ROOT/'slide_overview.png')
