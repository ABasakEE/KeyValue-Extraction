import json
from collections import Counter, defaultdict
from pathlib import Path

D = Path(r'D:\KeyValue-Extraction\data\docile\annotations')
pool = json.load(open(r'D:\KeyValue-Extraction\results\docile_rows\pool.json'))
by_layout = defaultdict(list)
doc_use = {}
for i in pool:
    a = json.load(open(D / f'{i}.json'))
    ft = Counter(e['fieldtype'] for e in a['line_item_extractions'])
    n, g = ft['line_item_amount_net'], ft['line_item_amount_gross']
    doc_use[i] = 'net' if n and not g else 'gross' if g and not n else 'both' if n and g else 'neither'
    by_layout[a['metadata']['cluster_id']].append(i)
print('documents by amount column used:', Counter(doc_use.values()))
mixed = consistent = single = 0
for l, ds in by_layout.items():
    us = {doc_use[d] for d in ds if doc_use[d] in ('net', 'gross')}
    if len(ds) < 2 or not us:
        single += 1
    elif len(us) > 1:
        mixed += 1
    else:
        consistent += 1
print(f'layouts with a net/gross amount column in >=2 docs: consistent {consistent}, mixed (some docs net, some gross) {mixed}; others {single}')
# among docs that have net or gross only: share of docs whose layout has docs of the other kind
docs_net = [d for d, u in doc_use.items() if u == 'net']
share_net_in_mixed = sum(1 for d in docs_net if len({doc_use[x] for x in by_layout[json.load(open(D / f'{d}.json'))['metadata']['cluster_id']] if doc_use[x] in ('net', 'gross')}) > 1)
print('net-only docs:', len(docs_net), '| of those in a layout that also has gross-only docs:', share_net_in_mixed)
