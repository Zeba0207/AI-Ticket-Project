import json, os, sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score

path = '../data/annotated/seed_annotated_v1.jsonl'  # exported annotations go here
if not os.path.exists(path):
    print("ERROR: annotated file not found at", path)
    sys.exit(1)

anns = [json.loads(l) for l in open(path, 'r', encoding='utf8')]
rows = []
for a in anns:
    ticket_id = a.get('ticket_id') or a.get('id') or a.get('pk') or a.get('meta',{}).get('id')
    annotator = a.get('annotator') or a.get('user') or a.get('labeler') or a.get('meta',{}).get('annotator','annotator')
    category = a.get('category') or a.get('label') or a.get('annotation')

    # doccano-style extraction
    if category is None and 'annotations' in a:
        ann = a['annotations'][0]
        if 'result' in ann and len(ann['result']):
            v = ann['result'][0].get('value',{})
            if 'choices' in v: category = v['choices'][0]
            elif 'labels' in v: category = v['labels'][0]

    rows.append({'ticket_id':ticket_id,'annotator':annotator,'category':category})

df = pd.DataFrame(rows)
print("Total rows:", len(df))
print("Missing category labels:", df['category'].isnull().sum())
print("\nSample rows:")
print(df.head(10).to_string(index=False))

if df['annotator'].nunique() >= 2:
    pivot = df.pivot_table(index='ticket_id', columns='annotator', values='category', aggfunc='first').dropna()
    cols = pivot.columns.tolist()
    if len(cols) >= 2:
        a1, a2 = pivot[cols[0]].astype(str), pivot[cols[1]].astype(str)
        kappa = cohen_kappa_score(a1, a2)
        print(f"\nCohen's kappa between {cols[0]} and {cols[1]} = {kappa:.4f}")
    else:
        print("\nNot enough annotators to compute Cohen's kappa.")
else:
    print("\nNeed at least 2 annotators for agreement.")
