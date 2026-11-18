"""Optional independent comparison worker; uses the canonical model definitions.
Run only for a model the sequential runner has not reached. Checkpoint is published
atomically after fitting, so interrupted workers never supply partial results.
"""
import argparse, os
from pathlib import Path
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
from data_pipeline import FEATURE_COLUMNS
from models.harness import group_cv, evaluate
from mlops.train import models, save

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('name',choices=list(models())); ap.add_argument('--output',default='artifacts/experiments')
    args=ap.parse_args(); out=Path(args.output)
    if (out/f'{args.name}.json').exists(): return
    import json,hashlib
    manifest=json.loads((out/'run.json').read_text())
    source=Path('artifacts/data/features.parquet')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=manifest['data_sha256']: raise ValueError('Data changed')
    for path,digest in manifest['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest: raise ValueError('Source changed')
    df=pd.read_parquet(source); tr=df[df.split=='train'].reset_index(drop=True); st=df[df.split=='student'].reset_index(drop=True)
    family,model=models()[args.name]
    print('Independent comparison',args.name,flush=True)
    cv,folds=group_cv(model,tr,FEATURE_COLUMNS,n_splits=manifest['folds'])
    scores,_,p=evaluate(model,tr,st,FEATURE_COLUMNS)
    row={'family':family,'config':str(model),'seed':42,**cv,**{'student_'+k:v for k,v in scores.items()}}
    folds.to_csv(out/f'{args.name}_folds.csv',index=False)
    np.save(out/f'{args.name}_student_probabilities.npy',p.astype('float32'))
    tmp=out/f'{args.name}.json.tmp'; save(tmp,row); os.replace(tmp,out/f'{args.name}.json')
    print(row,flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=4): main()
