"""Build a 61 x 85 cm PDF, editable PowerPoint, SVG and preview using paper figures."""
from pathlib import Path
from xml.sax.saxutils import escape
from io import BytesIO
import base64
import pymupdf as fitz
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from pptx import Presentation
from pptx.util import Cm, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

OUT=Path(__file__).resolve().parent
PAPER=OUT.parent/'paper'
ASSETS=OUT/'assets'
ASSETS.mkdir(exist_ok=True)
CM=72/2.54
W,H=61*CM,85*CM
NAVY='#102C42'; TEAL='#007E87'; INK='#173449'; PALE='#EDF5F6'; WHITE='#FFFFFF'; MUTED='#516473'
for name,file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf')]:
    pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
c=canvas.Canvas(str(OUT/'poster_base.pdf'),pagesize=(W,H))
c.setTitle('LEAF — Do LLM Proposers Improve Budgeted Search for Ionizable Lipids?')
c.setAuthor('Harshit Singh; Rajeev Kumar Singh; Chundru Sharath Krishna; Satya Pratik Srivastava; Rohan Gorantla')
prs=Presentation(); prs.slide_width=Cm(61); prs.slide_height=Cm(85)
slide=prs.slides.add_slide(prs.slide_layouts[6])
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="610mm" height="850mm" viewBox="0 0 {W} {H}">']
placements=[]
def rgb(s): return RGBColor.from_string(s.lstrip('#'))
def rect(x,y,w,h,fill):
    c.setFillColor(HexColor(fill)); c.rect(x*CM,H-(y+h)*CM,w*CM,h*CM,fill=1,stroke=0)
    s=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,Cm(x),Cm(y),Cm(w),Cm(h))
    s.fill.solid(); s.fill.fore_color.rgb=rgb(fill); s.line.fill.background()
    svg.append(f'<rect x="{x*CM}" y="{y*CM}" width="{w*CM}" height="{h*CM}" fill="{fill}"/>')
def text(x,y,s,size=24,bold=False,color=INK):
    font='Arial-Bold' if bold else 'Arial'
    c.setFont(font,size); c.setFillColor(HexColor(color)); c.drawString(x*CM,H-y*CM-size,s)
    # Position each editable line independently to preserve wrapping across formats.
    width=min(61-x,pdfmetrics.stringWidth(s,font,size)/CM+.45)
    sh=slide.shapes.add_textbox(Cm(x),Cm(y),Cm(width),Pt(size*1.28))
    tf=sh.text_frame; tf.clear(); tf.word_wrap=False
    tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
    p=tf.paragraphs[0]; p.space_before=p.space_after=0
    r=p.add_run(); r.text=s; r.font.name='Arial'; r.font.size=Pt(size); r.font.bold=bold; r.font.color.rgb=rgb(color)
    svg.append(f'<text x="{x*CM}" y="{y*CM+size}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(s)}</text>')
def para(x,y,s,w,size=23,bold=False,color=INK):
    font='Arial-Bold' if bold else 'Arial'; words=s.split(); lines=[]; line=''
    for word in words:
        trial=(line+' '+word).strip()
        if line and pdfmetrics.stringWidth(trial,font,size)>w*CM: lines.append(line); line=word
        else: line=trial
    if line: lines.append(line)
    for i,line in enumerate(lines): text(x,y+i*size*1.25/CM,line,size,bold,color)
    return y+len(lines)*size*1.25/CM
def heading(x,y,n,title):
    text(x,y,n,25,True,TEAL); text(x+1.55,y,title,27,True)
def figure(name,x,y,w,crop=None,asset_name=None):
    source=fitz.open(PAPER/name); page=source[0]; clip=fitz.Rect(crop) if crop else page.rect
    h=w*clip.height/clip.width
    # Vector figure in PDF; high-resolution figure image in PowerPoint and SVG.
    pix=page.get_pixmap(clip=clip,matrix=fitz.Matrix(w/2.54*220/clip.width,w/2.54*220/clip.width),alpha=False)
    data=pix.tobytes('png'); path=ASSETS/((asset_name or Path(name).stem)+'.png'); path.write_bytes(data)
    pic=slide.shapes.add_picture(BytesIO(data),Cm(x),Cm(y),width=Cm(w),height=Cm(h))
    pic.name='Paper figure: '+name+(' (selected panel)' if crop else '')
    pic._element.nvPicPr.cNvPr.set('descr','Original paper figure, '+name+'; preserve aspect ratio.')
    placements.append((str(PAPER/name),fitz.Rect(x*CM,y*CM,(x+w)*CM,(y+h)*CM),clip))
    svg.append(f'<image x="{x*CM}" y="{y*CM}" width="{w*CM}" height="{h*CM}" href="data:image/png;base64,{base64.b64encode(data).decode()}"/>')
    return y+h

rect(0,0,61,85,WHITE)
rect(0,0,61,10.6,NAVY)
text(1.7,.75,'LEAF / LIPID EVOLUTIONARY ADAPTIVE FRAMEWORK',20,True,'#71D2D4')
text(1.7,2.0,'Do LLM Proposers Improve Budgeted',45,True,WHITE)
text(1.7,3.85,'Search for Ionizable Lipids?',45,True,WHITE)
text(1.7,6.15,'Harshit Singh¹ · Rajeev Kumar Singh¹ · Chundru Sharath Krishna¹',24,False,WHITE)
text(1.7,7.25,'Satya Pratik Srivastava¹ · Rohan Gorantla²',24,False,WHITE)
text(1.7,8.65,'¹ Computer Science and Engineering, Shiv Nadar University, Delhi-NCR, India',19,False,'#D1E3EB')
text(1.7,9.45,'² School of Informatics, University of Edinburgh, U.K.',19,False,'#D1E3EB')

rect(1.7,11.35,57.6,4.85,PALE)
text(2.4,11.8,'No established advantage under matched query budgets.',32,True,TEAL)
para(2.4,13.3,'The tested LLM proposers do not establish superiority over genetic-algorithm or random-search baselines. Repeated candidates consume queries; greater novelty does not establish a performance gain.',55.9,25)

heading(1.7,17.05,'01','Matched-budget search: only the proposer varies')
figure('Figure1_architecture.pdf',1.7,19.1,42,asset_name='paper_fig1_architecture')
text(49.0,19.2,'30,044',35,True,TEAL)
para(49,20.9,'candidate lipids',10.3,21)
text(49.0,23.0,'B = 40',35,True,TEAL)
para(49,24.7,'oracle queries per run',10.3,21)
text(49.0,27.6,'4 LLMs',35,True,TEAL)
para(49,29.3,'8 paired seeds across 3 splits',10.3,21)
text(49.0,32.8,'1,200',35,True,TEAL)
para(49,34.5,'AGILE labels; 960 train / 240 test',10.3,21)
para(1.7,38.0,'Paper Fig. 1. Frozen evaluation GP and refitted acquisition GP remain separate. Both objectives are potency readouts. The dashed Ledger pathway produced no validated hypotheses in the reported sweep.',57.6,21)

heading(1.7,40.7,'02','No established LLM benefit')
heading(31.3,40.7,'03','Repetition limits novelty')
figure('fig2_styled_panel.pdf',1.7,42.2,27.9,(423,0,930.29065,285),'paper_fig2B_comparisons')
figure('fig2_styled_panel.pdf',31.3,42.2,27.9,(0,295,435,600),'paper_fig2C_duplicates')
para(1.7,58.4,'Paper Fig. 2B. Random-split paired hypervolume differences with 95% bootstrap CIs (n = 8). Asterisks denote unadjusted p < 0.05. No primary comparison survives Holm correction; equivalence is not established.',27.9,21)
rect(1.7,62.6,27.9,2.4,PALE)
para(2.2,62.9,'Novelty control helps some runs, but intervals remain wide and proposal costs increase.',26.9,21,True,TEAL)
para(31.3,62.1,'Paper Fig. 2C shows proposal-pool duplicates. Separately, 79.1% of Llama 8B’s charged random-split queries repeat archive members (GA: 15.3%; random: 0%).',27.9,21)

# Lower row uses the actual molecular structures and reasoning-engagement panel.
heading(1.7,66.3,'04','Best-found lipid structures')
figure('fig5_lipid_structures.pdf',1.7,67.8,27.9,(0,35,936,325),'paper_fig5A_structures')
para(1.7,76.9,'Paper Fig. 5A. One scored-prompt seed; GP-predicted structures are illustrative, not wet-lab validation or a cross-seed ranking.',27.9,20)

heading(31.3,66.3,'05','Evidence use ≠ search gain')
figure('fig4_evidence_panel.pdf',31.3,67.8,17.2,(650,60,1140,398),'paper_fig4B_evidence_use')
para(49.0,68.3,'Paper Fig. 4B. Scored prompts raise potency mentions to 77.5%, yet reasoning-mode search still trails GA and random search (n = 4).',10.3,21)

rect(1.7,80.2,57.6,2.5,NAVY)
para(2.25,80.5,'LIMITS  Retrospective GP-oracle simulation; no wet-lab validation. Chemotype-cold RAW264.7 fails the validity gate. HeLa-only reanalysis retains a deficit. Non-significance does not establish parity.',56.4,21,False,WHITE)
text(1.7,83.1,'CODE  github.com/harshitsinghsnu/LEAF',20,True,TEAL)
text(1.7,84.0,'Contact: rajeev.kumar@snu.edu.in · gorantlarohan@gmail.com',18)
text(40.4,83.1,'Data: bowang-lab/AGILE',18)
text(40.4,84.0,'UKRI EP/S02431X/1 · Exscientia',16)

# Final positioning is adjusted below for a compact lower-row layout.
slide.notes_slide.notes_text_frame.text=(
    'Source: LEAF_biorxiv.pdf supplied by the author. Original repository figure PDFs are embedded as high-resolution pictures. '
    'Fig. 1: full architecture. Fig. 2B: random-split comparisons; Fig. 2C: proposal duplicate rates, distinct from charged-query rates. '
    'Fig. 5A: illustrative best-found structures, one scored-prompt seed. Fig. 4B: reasoning engagement, exploratory follow-up. '
    'Canvas is 610 x 850 mm, portrait. Text and layout shapes are editable; charts are original paper artwork.'
)

c.showPage(); c.save()
doc=fitz.open(OUT/'poster_base.pdf')
for name,target,clip in placements:
    src=fitz.open(name); doc[0].show_pdf_page(target,src,0,clip=clip,overlay=True)
doc.save(OUT/'LEAF_workshop_poster_61x85cm.pdf',garbage=4,deflate=True)
doc[0].get_pixmap(matrix=fitz.Matrix(.8,.8),alpha=False).save(OUT/'LEAF_workshop_poster_preview.png')
prs.save(OUT/'LEAF_workshop_poster_61x85cm.pptx')
svg.append('</svg>'); (OUT/'LEAF_workshop_poster_editable.svg').write_text('\n'.join(svg),encoding='utf-8')
doc.close()
(OUT/'poster_base.pdf').unlink()
print('Built PDF, editable PPTX, SVG and preview with original paper figures.')
