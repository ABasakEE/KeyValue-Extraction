"""Study 34 post hoc (CPU): row F1 of the saved A2 answers after removing repeated rows, runaway pages (output cap hit) and pages without gold rows. Written after seeing the result."""
import sys,json,glob
from collections import Counter
sys.path[:0]=['scripts','src']
import adbuy_sg as S, adbuy_rows as R
from lir_metric import lir_counts, cell_counts, f1
prim=json.load(open('results/adbuy_sg/primary.json'))
recs={r['filename']:r for r in R.load_records() if r['filename'] in set(prim)}
files=sorted(glob.glob('results/adbuy_sg/run/rows_A2_*.jsonl'))
pages=S.read_pages(files)
cap={(r['doc_id'],r['page']) for p in files for r in map(json.loads,open(p,encoding='utf-8')) if (r.get('new_tokens') or 0)>=2400}
V={k:[0,0,0] for k in ('as is','dedupe rows','dedupe + drop capped pages','dedupe + drop pages without gold rows','dedupe + drop capped + no-gold pages')}
C={k:[0,0,0] for k in V}
n_docs=0; fp_pages=Counter()
for fn in prim:
    doc=S.make_doc(recs[fn])
    if not S.complete(doc,pages.get(fn,{})): continue
    n_docs+=1
    gold=R.gold_rows(recs[fn],doc); g=[{(f,ws) for f,ws,_,_ in r} for r in gold]
    gold_pages={r[0][2][0][2] for r in gold if r and r[0][2]}
    out={k:[] for k in V}
    for p in sorted(doc.lines):
        rows=[set(r) for r in S.decode_page(doc,p,pages[fn].get(p,'{"rows": []}'),Counter())]
        seen=[];dd=[]
        for r in rows:
            if r not in seen: seen.append(r); dd.append(r)
        if p not in gold_pages and rows: fp_pages['pages with predicted rows but no gold rows']+=1; fp_pages['rows on those pages']+=len(rows)
        out['as is']+=rows; out['dedupe rows']+=dd
        capped=(fn,p) in cap; nog=p not in gold_pages
        if not capped: out['dedupe + drop capped pages']+=dd
        if not nog: out['dedupe + drop pages without gold rows']+=dd
        if not capped and not nog: out['dedupe + drop capped + no-gold pages']+=dd
    for k in V:
        a=lir_counts(out[k],g); b=cell_counts(out[k],g)
        V[k]=[x+y for x,y in zip(V[k],a)]; C[k]=[x+y for x,y in zip(C[k],b)]
print(n_docs,'complete documents')
for k in V: print(f'{k:42s} row F1 {100*f1(V[k]):5.1f} (P {100*V[k][0]/max(1,V[k][1]):4.1f} R {100*V[k][0]/max(1,V[k][2]):4.1f}) | cell F1 {100*f1(C[k]):5.1f}')
print(dict(fp_pages))
