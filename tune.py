import json, subprocess, os, re, sys
SOF='/root/.claude/skills/synced/419c058e-2d32-44aa-9f5c-d3295d2c53bb_44f6ecdc-3c0e-413c-a008-6d0fd86d46b2/docx/scripts/office/soffice.py'
def build(sh):
    json.dump(sh, open('/home/claude/figshift.json','w'))
    subprocess.run(['python3','make_paper.py'],cwd='/home/claude',capture_output=True,timeout=100)
    subprocess.run(['python3',SOF,'--headless','--convert-to','pdf','CrysFluid-ARC_Ubi-Media2027.docx'],cwd='/home/claude',capture_output=True,timeout=110)
    info=subprocess.run(['pdfinfo','CrysFluid-ARC_Ubi-Media2027.pdf'],cwd='/home/claude',capture_output=True,text=True).stdout
    n=int(re.search(r'Pages:\s+(\d+)',info).group(1))
    last=subprocess.run(['pdftotext','-f',str(n),'-l',str(n),'-layout','CrysFluid-ARC_Ubi-Media2027.pdf','-'],cwd='/home/claude',capture_output=True,text=True).stdout
    allt=subprocess.run(['pdftotext','-layout','CrysFluid-ARC_Ubi-Media2027.pdf','-'],cwd='/home/claude',capture_output=True,text=True).stdout
    order=[int(m) for m in re.findall(r'^\s*Fig\. (\d)\.',allt,re.M)]
    pen = 0 if order==sorted(order) else 5
    return n + len([l for l in last.splitlines() if l.strip()])/60.0 + pen
best={}; bs=build(best); print('base',bs,flush=True)
for k in range(6):
    cand=[]
    for d in [-3,-2,-1,1,2,3,4]:
        t=dict(best); t[str(k)]=d
        try: sc=build(t)
        except Exception as e: print('err',e,flush=True); continue
        print(k,d,round(sc,2),flush=True); cand.append((sc,d))
    if cand:
        sc,d=min(cand)
        if sc<bs-0.01: best[str(k)]=d; bs=sc
    print('FIG',k,'best',best,round(bs,2),flush=True)
json.dump(best, open('/home/claude/figshift.json','w'))
build(best); print('FINAL',best,bs,flush=True)
