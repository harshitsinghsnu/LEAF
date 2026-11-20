"""Build a 610 x 850 mm poster. Editable text/graphics: SVG; print: PDF."""
from pathlib import Path
from xml.sax.saxutils import escape
import math
import pymupdf
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

OUT = Path(__file__).resolve().parent
CM = 72 / 2.54
W, H = 61 * CM, 85 * CM
for name, file in [('Arial','arial.ttf'), ('Arial-Bold','arialbd.ttf')]:
    pdfmetrics.registerFont(TTFont(name, 'C:/Windows/Fonts/' + file))
c = canvas.Canvas(str(OUT / 'LEAF_workshop_poster_61x85cm.pdf'), pagesize=(W,H))
c.setTitle('LEAF | Do LLM Proposers Improve Budgeted Search for Ionizable Lipids?')
c.setAuthor('Harshit Singh; Rajeev Kumar Singh; Chundru Sharath Krishna; Satya Pratik Srivastava; Rohan Gorantla')
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="610mm" height="850mm" viewBox="0 0 {W} {H}">']
NAVY='#102C42'; TEAL='#007E87'; INK='#173449'; MUTED='#516473'; PALE='#EDF5F6'; GOLD='#D38A25'; WHITE='#FFFFFF'; GRAY='#E3E9ED'
def rect(x,y,w,h,fill,stroke=None):
    x,y,w,h=[v*CM for v in (x,y,w,h)]
    c.setFillColor(HexColor(fill)); c.setStrokeColor(HexColor(stroke or fill))
    c.rect(x,H-y-h,w,h,fill=1,stroke=bool(stroke))
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke or fill}"/>')
def text(x,y,s,size=25,bold=False,color=INK):
    font='Arial-Bold' if bold else 'Arial'
    c.setFillColor(HexColor(color)); c.setFont(font,size); c.drawString(x*CM,H-y*CM-size,s)
    svg.append(f'<text x="{x*CM}" y="{y*CM+size}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(s)}</text>')
def para(x,y,s,w,size=24,bold=False,color=INK,leading=1.27):
    font='Arial-Bold' if bold else 'Arial'; words=s.split(); lines=[]; line=''
    for word in words:
        trial=(line+' '+word).strip()
        if pdfmetrics.stringWidth(trial,font,size)>w*CM and line:
            lines.append(line); line=word
        else: line=trial
    if line: lines.append(line)
    for i,line in enumerate(lines): text(x,y+i*size*leading/CM,line,size,bold,color)
    return y+len(lines)*size*leading/CM
def line(x1,y1,x2,y2,color=TEAL,width=2):
    c.setStrokeColor(HexColor(color)); c.setLineWidth(width); c.line(x1*CM,H-y1*CM,x2*CM,H-y2*CM)
    svg.append(f'<line x1="{x1*CM}" y1="{y1*CM}" x2="{x2*CM}" y2="{y2*CM}" stroke="{color}" stroke-width="{width}"/>')
def arrow(x1,y1,x2,y2,color=TEAL):
    line(x1,y1,x2,y2,color,3)
    a=math.atan2(y2-y1,x2-x1)
    for d in [-.5,.5]: line(x2,y2,x2-.38*math.cos(a+d),y2-.38*math.sin(a+d),color,3)
def heading(x,y,num,title):
    text(x,y,num,27,True,TEAL); text(x+1.7,y,title,29,True)

rect(0,0,61,85,WHITE)
rect(0,0,61,10.6,NAVY)
text(1.7,.75,'LEAF  /  LIPID EVOLUTIONARY ADAPTIVE FRAMEWORK',20,True,'#71D2D4')
text(1.7,2.0,'Do LLM Proposers Improve Budgeted',45,True,WHITE)
text(1.7,3.85,'Search for Ionizable Lipids?',45,True,WHITE)
text(1.7,6.15,'Harshit Singh¹ · Rajeev Kumar Singh¹ · Chundru Sharath Krishna¹',24,False,WHITE)
text(1.7,7.25,'Satya Pratik Srivastava¹ · Rohan Gorantla²',24,False,WHITE)
text(1.7,8.65,'¹ Computer Science and Engineering, Shiv Nadar University, Delhi-NCR, India',19,False,'#D1E3EB')
text(1.7,9.45,'² School of Informatics, University of Edinburgh, U.K.',19,False,'#D1E3EB')

rect(1.7,11.5,57.6,5.1,PALE)
text(2.5,11.95,'No established advantage under matched query budgets.',32,True,TEAL)
para(2.5,13.45,'Across the tested settings, LLM proposers do not establish superiority over genetic-algorithm or random-search baselines. Repetition is a major inefficiency; reducing it does not establish a performance gain.',55.6,25)
for x,big,small in [(1.7,'30,044','Ugi-3CR candidate lipids'),(16.5,'40','oracle queries per run'),(31.3,'4 models','primary LLM comparison'),(46.1,'8 × 3','paired seeds × splits')]:
    text(x,17.65,big,42,True,TEAL); text(x,19.5,small,20)
line(1.7,21.25,59.3,21.25,GRAY)

heading(1.7,22.1,'01','A controlled test: vary only the proposer')
para(1.7,23.65,'Can an LLM select better reagent combinations when every method shares the same chemistry, acquisition rule, evaluation oracle, and query budget?',57.6,25)
steps=[(1.7,10.4,'PROPOSE','LLM / GA / random','10 reagent triples'),(14.0,10.4,'SCORE + FILTER','Refitted Tanimoto GP','RDKit rule screen'),(26.3,10.4,'SELECT','UCB, β = 2','Up to 5 candidates'),(38.6,10.4,'EVALUATE','Frozen split-trained GP','HeLa + RAW264.7')]
for x,w,title,a,b in steps:
    rect(x,26.9,w,4.7,PALE if x!=38.6 else '#FFF3DF')
    text(x+.4,27.25,title,22,True,TEAL if x!=38.6 else '#956014')
    text(x+.4,28.75,a,18); text(x+.4,29.7,b,18)
    if x!=38.6: arrow(x+w+.15,29.2,x+w+1.55,29.2)
rect(50.9,26.9,8.4,4.7,NAVY)
text(51.35,27.3,'B = 40',30,True,WHITE); text(51.35,29.1,'All queries count',17,False,WHITE)
arrow(49.15,29.2,50.65,29.2,GOLD)
line(43.8,32,43.8,33.1); arrow(43.8,33.1,6.9,33.1); arrow(6.9,33.1,6.9,31.8)
text(14.4,33.4,'Acquired labels → Pareto archive update + surrogate refit',22,True,TEAL)
para(1.7,35.1,'AGILE: 1,200 labeled lipids; 960 train / 240 test per split. Random, scaffold-cold, and chemotype-cold splits test increasing extrapolation. Hypervolume measures two potency readouts, not toxicity.',57.6,22)

heading(1.7,38.0,'02','Primary result')
heading(31.5,38.0,'03','Where the budget goes')
para(1.7,39.55,'Random split · median paired hypervolume difference · n = 8. Positive values favor the LLM.',27.4,23)
rect(1.7,42.4,27.5,1.45,NAVY)
text(2.1,42.65,'Proposer',20,True,WHITE); text(17.0,42.65,'vs. GA',20,True,WHITE); text(23.0,42.65,'vs. random',19,True,WHITE)
rows=[('Llama 3.1 8B','−47.1','−42.0'),('Qwen2.5 7B','−40.9','−29.1'),('Qwen2.5 14B','+1.8','−4.3'),('Gemini Flash-Lite','−91.8','−85.5')]
for i,(a,b,d) in enumerate(rows):
    y=43.85+i*1.65; rect(1.7,y,27.5,1.65,PALE if i%2==0 else '#F8FAFB')
    text(2.1,y+.35,a,22); text(17.2,y+.35,b,24,True,TEAL if b.startswith('+') else INK); text(24.0,y+.35,d,24,True)
para(1.7,51.1,'None of the eight primary comparisons survives Holm correction. Equivalence at a 10% margin is also not established.',27.5,23,True)

text(31.5,39.7,'79.1%',66,True,TEAL)
para(43.4,40.0,'of Llama 8B’s selected queries repeat archive members.',15.6,24,True)
text(31.5,43.1,'Charged-query duplicates · random split · 8 seeds',20,False,MUTED)
for i,(label,val,col) in enumerate([('Llama 8B',79.1,TEAL),('Static-GA',15.3,GOLD),('Random',0.0,'#8298A7')]):
    y=44.6+i*1.85
    text(31.5,y,label,21)
    rect(40.0,y+.12,15,0.7,GRAY)
    if val: rect(40.0,y+.12,15*val/100,.7,col)
    text(55.6,y-.02,f'{val:.1f}%',21,True)
para(31.5,50.65,'Acquisition does not remove repetition. Lower duplication in larger models still does not establish superiority; duplication alone does not explain all deficits.',27.8,23)

line(1.7,55.0,59.3,55.0,GRAY)
heading(1.7,56.0,'04','Novelty helps, at a cost')
heading(31.5,56.0,'05','Evidence use ≠ better search')
para(1.7,57.65,'Stricter all-evaluated de-duplication improves Llama 8B’s median difference against GA from −33.5 to +20.1 (four random-split seeds).',27.5,24)
rect(1.7,61.45,27.5,3.3,PALE)
text(2.25,61.8,'+20.1 vs. GA',31,True,TEAL)
text(2.25,63.25,'95% bootstrap CI: [−46.6, 119.8]',22)
para(1.7,65.35,'The interval crosses zero; the difference against random search remains −15.4. Archive-only filtering increases proposal cost and leaves 2 of 8 random-split runs short of the query cap.',27.5,23)
para(31.5,57.65,'Adding objective scores raises potency mentions in Qwen3-8B reasoning traces from 0/155 to 124/160 (77.5%).',27.8,24)
rect(31.5,61.45,27.8,3.3,PALE)
text(32.05,61.8,'0% → 77.5%',34,True,TEAL)
text(32.05,63.25,'Evidence engagement, without a gain',21)
para(31.5,65.35,'Scored reasoning-mode search still trails GA (−19.1) and random search (−53.4), with four seeds. Shuffling scores changes query trajectories in 3 of 4 seeds despite similar final hypervolume.',27.8,23)

rect(1.7,71.0,57.6,8.25,NAVY)
text(2.4,71.5,'INTERPRETATION & LIMITS',25,True,'#71D2D4')
para(2.4,73.0,'LEAF is a reusable benchmark for proposal quality. These are retrospective GP-oracle simulations, not wet-lab validation. Most candidate lipids require oracle extrapolation.',55.9,24,False,WHITE)
para(2.4,75.6,'Chemotype-cold RAW264.7 fails the validity gate (ρ = 0.23). A validated HeLa-only reanalysis retains a Llama 8B deficit (−3.93 vs. both baselines; unadjusted p = 0.015625). The Ledger was empty; verification reduced to a deterministic rule.',55.9,22,False,WHITE)

text(1.7,80.15,'CODE & DATA',21,True,TEAL)
text(1.7,81.3,'github.com/harshitsinghsnu/LEAF',22,True)
text(1.7,82.4,'Data: github.com/bowang-lab/AGILE',19)
text(1.7,83.45,'Contact: rajeev.kumar@snu.edu.in · gorantlarohan@gmail.com',18)
text(40.3,80.2,'ACKNOWLEDGMENTS',18,True,TEAL)
para(40.3,81.3,'Shiv Nadar University; UKRI Biomedical AI CDT (EP/S02431X/1); Exscientia Plc.',18.8,18)
c.linkURL('https://github.com/harshitsinghsnu/LEAF',(1.7*CM,H-82.3*CM,28*CM,H-81.1*CM),relative=0)
c.showPage(); c.save()
svg.append('</svg>')
(OUT/'LEAF_workshop_poster_editable.svg').write_text('\n'.join(svg),encoding='utf-8')
doc=pymupdf.open(OUT/'LEAF_workshop_poster_61x85cm.pdf')
doc[0].get_pixmap(matrix=pymupdf.Matrix(.8,.8),alpha=False).save(OUT/'LEAF_workshop_poster_preview.png')
assert len(doc)==1
assert abs(doc[0].rect.width-W)<.01 and abs(doc[0].rect.height-H)<.01
print('Created one-page 610 x 850 mm PDF, editable SVG, and PNG preview.')
