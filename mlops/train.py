"""Reproducible five-model comparison, group CV, Bayesian tuning and export."""
import argparse, hashlib, json, platform, subprocess, time
from pathlib import Path
import joblib, numpy as np, pandas as pd, optuna, sklearn
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from data_pipeline import FEATURE_COLUMNS
from models.harness import group_cv, evaluate

SEED=42

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))

def models():
    return {
        'LogReg': ('Linear', make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, random_state=SEED))),
        'GaussianNB': ('Probabilistic', GaussianNB()),
        'DecisionTree': ('Tree', DecisionTreeClassifier(max_depth=12, min_samples_leaf=30, random_state=SEED)),
        'ExtraTrees': ('Ensemble', ExtraTreesClassifier(n_estimators=40, max_depth=16, min_samples_leaf=20, n_jobs=4, random_state=SEED)),
        'HistGB': ('Boosting', HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, early_stopping=False, random_state=SEED))}

def parameters(name, trial):
    if name=='LogReg': return {'logisticregression__C':trial.suggest_float('logisticregression__C', .01, 10, log=True)}
    if name=='GaussianNB': return {'var_smoothing':trial.suggest_float('var_smoothing', 1e-12, 1e-5, log=True)}
    if name in ('DecisionTree','ExtraTrees'):
        return {'max_depth':trial.suggest_int('max_depth', 8, 22), 'min_samples_leaf':trial.suggest_int('min_samples_leaf', 5, 100, log=True)}
    return {'learning_rate':trial.suggest_float('learning_rate', .03, .2, log=True),
            'max_leaf_nodes':trial.suggest_int('max_leaf_nodes', 8, 40),
            'l2_regularization':trial.suggest_float('l2_regularization', 1e-4, 10, log=True),
            'max_iter':trial.suggest_int('max_iter', 60, 160)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features', default='artifacts/data/features.parquet')
    ap.add_argument('--output', default='artifacts/experiments')
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--trials', type=int, default=8)
    args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    df=pd.read_parquet(args.features)
    if set(df.split)!= {'train','student'}: raise ValueError('Unexpected partitions')
    tr=df[df.split=='train'].reset_index(drop=True); st=df[df.split=='student'].reset_index(drop=True)
    if set(tr.trip_id)&set(st.trip_id): raise ValueError('Trip leakage')
    if tr.trip_id.nunique()!=80 or st.trip_id.nunique()!=10: raise ValueError('Expected 80/10 trips')
    manifest={'seed':SEED,'folds':args.folds,'train_trips':sorted(tr.trip_id.unique()),'student_trips':sorted(st.trip_id.unique()),
              'features':FEATURE_COLUMNS,'data_sha256':hashlib.sha256(Path(args.features).read_bytes()).hexdigest(),
              'python':platform.python_version(),'sklearn':sklearn.__version__, 'numpy':np.__version__,
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'selection':'highest mean group-CV observation top1; student results reported only'}
    manifest['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path('mlops/train.py'),Path('models/harness.py'),Path('data_pipeline.py'),Path('decoder.py')]}
    if (out/'run.json').exists():
        old=json.loads((out/'run.json').read_text())
        for key in ('data_sha256','source_sha256','folds','features','sklearn','numpy'):
            if old.get(key)!=manifest[key]: raise ValueError(f'Stale run ({key}); choose a fresh --output directory')
    save(out/'run.json',manifest)
    zoo=models(); results={}
    for name,(family,model) in zoo.items():
        checkpoint=out/f'{name}.json'
        if checkpoint.exists():
            results[name]=json.loads(checkpoint.read_text()); continue
        print('Training',name,flush=True)
        cv,folds=group_cv(model,tr,FEATURE_COLUMNS,n_splits=args.folds)
        scores,fitted,p=evaluate(model,tr,st,FEATURE_COLUMNS)
        row={'family':family,'config':str(model),'seed':SEED,**cv,**{'student_'+k:v for k,v in scores.items()}}
        folds.to_csv(out/f'{name}_folds.csv',index=False)
        np.save(out/f'{name}_student_probabilities.npy',p.astype('float32'))
        save(checkpoint,row); results[name]=row
        save(out/'model_comparison.json',results)
        print(name, row['cv_top1_mean'],row['student_top1'],flush=True)
    save(out/'model_comparison.json',results)
    name=max(results,key=lambda x:results[x]['cv_top1_mean'])
    base=zoo[name][1]
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED,n_startup_trials=3),
        storage='sqlite:///'+str((out/'study.db').resolve()),study_name='pathsnap-'+name,load_if_exists=True)
    def objective(trial):
        model=clone(base).set_params(**parameters(name,trial))
        cv,folds=group_cv(model,tr,FEATURE_COLUMNS,n_splits=args.folds)
        folds.to_csv(out/f'trial_{trial.number}_folds.csv',index=False)
        trial.set_user_attr('cv',cv)
        return cv['cv_top1_mean']
    t0=time.perf_counter()
    completed=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
    study.optimize(objective,n_trials=max(0,args.trials-completed))
    study.trials_dataframe().to_csv(out/'trials.csv',index=False)
    tuned=clone(base).set_params(**study.best_params)
    tuned_scores,tuned_fit,p=evaluate(tuned,tr,st,FEATURE_COLUMNS)
    tuned_cv=study.best_trial.user_attrs['cv']
    use_tuned=study.best_value>=results[name]['cv_top1_mean']
    if use_tuned: final=tuned_fit
    else: final=clone(base).fit(tr[FEATURE_COLUMNS],tr.label)
    summary={'selected_model':name,'best_params':study.best_params,'trials':len(study.trials),'search_seconds_this_invocation':time.perf_counter()-t0,
             'untuned':results[name],'tuned':{**tuned_cv,**{'student_'+k:v for k,v in tuned_scores.items()}},'deployed_tuned':use_tuned}
    save(out/'tuning.json',summary)
    dest=Path('artifacts/model'); dest.mkdir(parents=True,exist_ok=True)
    bundle={'model':final,'features':FEATURE_COLUMNS,'seed':SEED,'candidates':{'k':32,'radius':300},
            'decoder':{'beta_m':40.,'gap_s':10.,'max_distance_m':2000.}}
    joblib.dump(bundle,dest/'best_model.joblib',compress=3)
    checksum=hashlib.sha256((dest/'best_model.joblib').read_bytes()).hexdigest()
    manifest.update(run_id=name+'-'+checksum[:12],model=name,tuned=use_tuned,sha256=checksum)
    save(dest/'best_model.json',manifest)
    print('Exported verified model',name,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4): main()
