"""Generate the technical report solely from completed experiment artifacts."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether

ROOT=Path('artifacts/experiments')

def main():
    comparison=json.loads((ROOT/'model_comparison.json').read_text())
    tuning=json.loads((ROOT/'tuning.json').read_text())
    sequence=json.loads((ROOT/'sequence_summary.json').read_text())
    manifest=json.loads(Path('artifacts/model/best_model.json').read_text())
    validation=json.loads(Path('reports/release_validation.json').read_text())
    summary=json.loads(Path('artifacts/data/summary.json').read_text())
    styles=getSampleStyleSheet(); styles['BodyText'].leading=14
    styles['Heading1'].keepWithNext=True
    styles['Heading2'].keepWithNext=True
    story=[]; md=[]
    def text(s):
        md.append(s+'\n'); story.extend([Paragraph(escape(s),styles['BodyText']),Spacer(1,7)])
    def heading(s,level=1):
        md.append('#'*level+' '+s+'\n'); story.append(Paragraph(escape(s),styles['Heading'+str(level)]))
    def table(headers,rows,widths=None):
        md.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |')
        md.extend('| '+' | '.join(map(str,r))+' |' for r in rows); md.append('')
        cells=[[Paragraph(escape(str(c)),styles['BodyText']) for c in row] for row in [headers]+rows]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dfeafa')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.lightgrey),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]))
        story.extend([t,Spacer(1,10)])
    def picture(path,caption):
        from PIL import Image as PILImage
        w,h=PILImage.open(path).size
        width=6.5*inch; height=width*h/w
        if height>6*inch: width*=6*inch/height; height=6*inch
        story.extend([KeepTogether([Image(str(path),width=width,height=height),Paragraph(escape(caption),styles['Caption'] if 'Caption' in styles else styles['BodyText'])]),Spacer(1,10)])
        md.append(f'![{caption}](../{path.as_posix()})\n')
    heading('PathSnap Technical Report')
    text('Distributed GPS map matching | Complete A-D implementation and measured student-development evaluation | 18 November 2026')
    text('This report is generated from repository artifacts. It reports actual experiments, not target scores. The available dataset contains 80 training trips and 10 student-testing trips; the reserved evaluator partition has not been used. Codex assisted implementation and documentation.')
    heading('2.2 Dataset understanding and exploratory analysis')
    text('The raw formats are longitude/latitude/time tracks, longitude/latitude nodes, directed endpoint-index arcs, and chronological reference arc sequences. The pipeline validates dimensions, finite values, coordinate ranges, integer IDs and strictly increasing timestamps; it fails instead of silently cleaning malformed data. All 360 source files are inventoried with SHA-256, size, shape and dtype.')
    table(['Partition','Trips','Observations','Candidate miss rate'],[[p,v['trips'],v['observations'],f"{v['miss_rate']:.4%}"] for p,v in summary['partitions'].items()],[150,60,110,148])
    text('EDA found 617 gaps above 10 seconds across 42 trips, with a largest gap of 153,064 seconds. There were no nonfinite/out-of-range coordinates or non-increasing timestamps. Networks span both hemispheres, so each uses its median-coordinate UTM zone. There are 90 distinct network hashes, 106 zero-length arcs, and 14 disconnected consecutive reference pairs across 11 routes. These are source-data findings from train and student partitions, not evaluator analysis.')
    text('Reference routes do not contain per-observation labels. A monotone dynamic program aligns each GPS point to route occurrences using squared spatial distance and a weak elapsed-progress tie breaker. Labels are proxies with ambiguous junctions, loops and displaced observations. High residual observations remain in the data. There is no ground-truth insertion into candidates.')
    text('Nine manual features describe distance, projected offset and offset fraction, heading difference, measured speed, arc length, distance rank, and endpoint degree. Distances and headings provide local geometric agreement; speed and graph degree describe motion and junction ambiguity. Undefined speed/heading use -1. The classifier input is an explicit allowlist; IDs, labels and route-derived audit fields are excluded. No automated feature-selection claim is made for this release.')
    picture(Path('artifacts/data/eda.png'),'EDA from the supplied training and student partitions.')
    heading('3. Model experimentation and comparison')
    text(f"All five models were fitted on all 80 training trips and evaluated with {manifest['folds']}-fold GroupKFold. Scaling is fitted inside each training fold. Model choice maximizes mean CV observation Top-1 accuracy, matching the road-selection objective. Student data is evaluated in full and does not choose hyperparameters. Seed is 42. Candidate F1, precision and recall are macro averages at a fixed 0.5 threshold; log loss measures binary candidate probability quality.")
    headers=['Model','CV F1','Student Top-1','Student F1','Log loss','Continuity']
    table(headers,[[n,f"{r['cv_macro_f1_mean']:.4f}",f"{r['student_top1']:.4f}",f"{r['student_macro_f1']:.4f}",f"{r['student_log_loss']:.4f}",f"{r['student_route_continuity']:.4f}"] for n,r in comparison.items()],[100,65,83,72,65,83])
    table(['Model','CV Top-1','Fit seconds','Inference ms / 1k candidates'],[[n,f"{r['cv_top1_mean']:.4f}",f"{r['student_train_time_s']:.2f}",f"{r['student_inference_ms_per_1k_cands']:.3f}"] for n,r in comparison.items()],[115,85,100,168])
    text('Timing values are local wall-clock measurements, not isolated hardware benchmarks. The HistGB comparison ran alongside another model to reduce total turnaround; resource contention can affect timing. Model selection uses CV quality rather than these timing differences.')
    best_f1=max(comparison,key=lambda n:comparison[n]['student_macro_f1'])
    best_top1=max(comparison,key=lambda n:comparison[n]['student_top1'])
    text(f'Student candidate macro F1 is highest for {best_f1}, while student observation Top-1 is highest for {best_top1}. These metrics ask different questions: thresholded classification across mostly negative candidate rows versus choosing one road per observation. High candidate accuracy can be dominated by negatives. The chosen primary metric is grouped-CV Top-1; log loss and F1 remain supporting diagnostics.')
    descriptions={
      'LogReg':('Linear','Logistic loss with L2 regularization; LBFGS optimization.','A scaled linear baseline tests whether geometric cues combine additively. Fast and interpretable, but cannot directly represent complex feature interactions.'),
      'GaussianNB':('Probabilistic','Gaussian class-conditional likelihood with conditional feature independence.','A low-cost probability baseline. Correlated distance, offset and rank violate its independence assumption and can make confidence poorly calibrated.'),
      'DecisionTree':('Tree','Greedy Gini impurity minimization.','Captures feature thresholds and interactions. A single tree is unstable and produces coarse probability estimates; depth and leaf-size constraints limit variance.'),
      'ExtraTrees':('Ensemble','Randomized split ensembles using Gini impurity.','Averages nonlinear trees to reduce variance. More memory and latency than linear methods; random thresholds may miss precise geometric boundaries.'),
      'HistGB':('Boosting','Binary log loss optimized by histogram gradient boosting.','Learns nonlinear interactions with bounded trees and efficient histogram bins. Sensitive to complexity and probability calibration; random row-level early stopping is disabled to avoid within-trip validation.')}
    initial={
      'LogReg':'C=1, L2 penalty, LBFGS, max_iter=300; StandardScaler in each fold.',
      'GaussianNB':'var_smoothing=1e-9; class priors estimated from each training fold.',
      'DecisionTree':'max_depth=12, min_samples_leaf=30, Gini criterion.',
      'ExtraTrees':'40 trees, max_depth=16, min_samples_leaf=20, max_features=1.0, four workers.',
      'HistGB':'100 iterations, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=20, L2=0, early_stopping=False.'}
    for name,r in comparison.items():
        heading(name,2)
        family,loss,reason=descriptions[name]
        text(f'Family: {family}. {reason} Objective: {loss}')
        text('Initial configuration: '+r['config'])
        text('Important initial values: '+initial[name])
        text(f"Student results: Top-1 {r['student_top1']:.4f}, macro F1 {r['student_macro_f1']:.4f}, log loss {r['student_log_loss']:.4f}, macro precision {r['student_precision']:.4f}, macro recall {r['student_recall']:.4f}, candidate accuracy {r['student_accuracy']:.4f}. CV F1 {r['cv_macro_f1_mean']:.4f} +/- {r['cv_macro_f1_std']:.4f}.")
        gap=r['student_macro_f1']-r['cv_macro_f1_mean']
        text(f'The student-minus-CV F1 gap is {gap:+.4f}. This compares different trip populations and label quality, so it cannot by itself establish overfitting or underfitting. Fold dispersion and student performance provide generalization evidence; no artificial training curves are produced.')
    plots=ROOT/'plots'; plots.mkdir(exist_ok=True)
    names=list(comparison); x=np.arange(len(names))
    fig,axes=plt.subplots(3,1,figsize=(8,8))
    axes[0].bar(names,[comparison[n]['cv_macro_f1_mean'] for n in names],yerr=[comparison[n]['cv_macro_f1_std'] for n in names],capsize=3); axes[0].set_title('Trip-grouped CV macro F1 (mean and standard deviation)')
    axes[1].bar(x-.18,[comparison[n]['student_top1'] for n in names],.36,label='Top-1'); axes[1].bar(x+.18,[comparison[n]['student_macro_f1'] for n in names],.36,label='Macro F1'); axes[1].set_xticks(x,names); axes[1].legend(loc='upper left',bbox_to_anchor=(1.01,1)); axes[1].set_title('Student performance')
    axes[2].bar(names,[comparison[n]['student_log_loss'] for n in names]); axes[2].set_title('Student log loss (lower is better)')
    fig.tight_layout(); fig.savefig(plots/'comparison.png',dpi=160); plt.close(fig)
    picture(plots/'comparison.png','Actual model comparison across all five algorithms.')
    fig,axes=plt.subplots(2,3,figsize=(10,6))
    for ax,(n,r) in zip(axes.flat,comparison.items()):
        cm=np.array(r['student_confusion_matrix']); ax.imshow(cm,cmap='Blues'); ax.set_title(n)
        for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha='center',color='red')
        ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted',ylabel='Reference proxy')
    axes.flat[-1].axis('off'); fig.tight_layout(); fig.savefig(plots/'confusion.png',dpi=160); plt.close(fig)
    picture(plots/'confusion.png','Candidate-level student confusion matrices, class order 0 and 1.')
    heading('4. Bayesian hyperparameter optimisation')
    ordered=sorted(comparison,key=lambda n:comparison[n]['cv_top1_mean'],reverse=True)
    lead=100*(comparison[ordered[0]]['cv_top1_mean']-comparison[ordered[1]]['cv_top1_mean'])
    text(f'The baseline CV leader exceeds the runner-up by only {lead:.3f} percentage points. This is a small observed advantage, not a statistical claim of definitive superiority. The predeclared selection rule identifies one family for the finite tuning budget.')
    text('Chosen model: '+tuning['selected_model']+'. The exact model-specific search space is encoded in mlops/train.py: LogReg C [0.01,10] log; GaussianNB smoothing [1e-12,1e-5] log; tree/ExtraTrees depth [8,22] and minimum leaf samples [5,100] log; HistGB learning rate [0.03,0.2] log, leaf nodes [8,40], L2 [1e-4,10] log, iterations [60,160]. Only the selected family is searched.')
    text(f"Optuna TPE completed {tuning['trials']} trials, including three startup trials. Objective: mean training-only group-CV Top-1. Further optimization was stopped to prioritize the working demonstration. One interrupted trial was marked failed and excluded from selection. Best parameters: {json.dumps(tuning['best_params'])}. Tuned model deployed: {tuning['deployed_tuned']}.")
    keys=['cv_macro_f1_mean','cv_top1_mean','student_top1','student_macro_f1','student_log_loss','student_inference_ms_per_1k_cands']
    table(['Metric','Untuned','Tuned'],[[k,f"{tuning['untuned'][k]:.5f}",f"{tuning['tuned'][k]:.5f}"] for k in keys],[290,89,89])
    text('Selection and promotion follow CV, not the student score. The tuned model is retained only when CV Top-1 is at least the untuned score. Final refitting uses all training trips. The limited completed search is not evidence of a global optimum; no further accuracy tuning is needed to run this demonstration.')
    heading('5. HMM and Viterbi reconstruction')
    text('Hidden states are directed candidate arcs. Emissions are clipped positive-class classifier probabilities normalized across candidates per observation. Transition weights are exp(-abs(network distance - GPS displacement)/40 m), normalized over reachable next candidates. Network distance includes the remaining previous arc, bounded directed Dijkstra distance between endpoints, and next-arc offset. Same-arc movement permits 5 m backward jitter. The shortest-path bound is 2,000 m.')
    text('Viterbi uses score[t,j] = log emission[t,j] + max_i(score[t-1,i] + log transition[i,j]), then backtracks the maximizing predecessors. Missing candidates output -1 and break the chain. Gaps above 10 seconds or fully unreachable transitions restart the chain and are explicitly reported. These parameters are fixed engineering defaults; no decoder tuning is claimed.')
    table(['Metric','Classifier','Classifier + HMM'],[[metric,f"{sequence['classifier_'+metric]:.5f}",f"{sequence['hmm_'+metric]:.5f}"] for metric in ['top1','continuity','route_jaccard','exact_sequence']],[230,119,119])
    delta=sequence['hmm_top1']-sequence['classifier_top1']
    text(f'HMM changes pointwise agreement by {delta:+.5f}. Improvement is not guaranteed: transition assumptions may disagree with proxy labels at parallel roads, sharp turns, long gaps or reference-route discontinuities. Per-trip evidence is in sequence_metrics.csv. Continuity is strict endpoint adjacency, not general graph reachability. Route Jaccard ignores order; exact sequence match uses the entire proxy arc sequence. Pointwise means are observation-weighted; continuity is pair-weighted; exact-sequence match is trip-weighted.')
    per=pd.read_csv(ROOT/'sequence_metrics.csv',dtype={'trip_id':str})
    table(['Trip','Classifier Top-1','HMM Top-1','Breaks'],[[r.trip_id,f'{r.classifier_top1:.4f}',f'{r.hmm_top1:.4f}',str(r.breaks)] for r in per.itertuples()],[115,125,125,103])
    heading('6. Geographical visualisation')
    text('artifacts/experiments/route_map.html is the interactive Folium comparison, with raw GPS, road network, reference and predicted segment layers plus layer controls. The static figure below is generated from the same actual coordinates and predictions. It is an exported geometry view; interactive basemap tiles need internet access.')
    picture(ROOT/'route_map.png','Student route: blue GPS, grey network, green reference, red HMM-selected segments.')
    if Path('reports/ui_screenshot.png').exists(): picture(Path('reports/ui_screenshot.png'),'Working evaluator UI on the local server.')
    if Path('reports/folium_screenshot.png').exists(): picture(Path('reports/folium_screenshot.png'),'Rendered interactive Folium map.')
    heading('7. API, UI, deployment and MLOps')
    text('Laptop 1 hosts the backend API, model and graph. Laptop 2 opens its independent frontend/index.html and connects to Laptop 1 via the configured LAN API URL. The portable ZIP includes local map assets; no model or reference data is bundled. The UI requests /trips and POST /predict/{trip_id}; only prediction geometry, input GPS and aggregate scores are returned. Prediction and reference scoring are separate. The API reports proxy-label point accuracy and explicitly named route Jaccard. No dummy fallback remains.')
    text('The release validates model SHA-256 and feature schema before startup, fails on invalid trips or incorrect trip count, and exports data/source fingerprints, dependency versions and seed. /health, /ready and /metrics support operations. Structured JSON prediction logs record request ID, model version, latency and score. Docker mounts data/references read-only; CI runs synthetic tests without raw data. README documents installation, reproduction, environment configuration and rollback.')
    text(f"Automated API verification passed for {len(validation['trips'])} real student trips, with one predicted arc per observation and no reference arrays in responses. Scope: {validation['scope']}. GitHub publication, reserved evaluator scores and a physical two-laptop LAN test are not established by these checks.")
    table(['Trip','HTTP','Points','API seconds'],[[r['trip_id'],r['status'],r['points'],f"{r['seconds']:.2f}"] for r in validation['trips']],[140,60,110,158])
    if Path('reports/api_screenshot.png').exists(): picture(Path('reports/api_screenshot.png'),'Successful API response evidence from the running local server.')
    heading('Reproducibility and remaining demonstration checks')
    text('Commands: python data_pipeline.py; python -m mlops.train; python -m mlops.evaluate; python -m unittest discover -s tests -v; python -m mlops.verify_release; python -m mlops.report. The README includes the complete file tree and operational runbook. All reported values can be traced to compact JSON/CSV artifacts; the model manifest identifies source hashes and dataset hash. No epoch-wise curves are claimed for the five selected algorithms.')
    text('Known limits: proxy labels, incomplete candidate coverage in training, a small student partition, fixed decoder defaults, bounded road-distance searches, and simplified straight arc geometry. This release predicts the observed segments; it does not insert unobserved connecting arcs into output geometry. No independently verified physical ground truth is available. Run the physical LAN/container checks and evaluator setup on the actual assessment machines, and publish the repository before submission.')
    Path('reports/technical_report.md').write_text('\n'.join(md),encoding='utf-8')
    from mlops.consolidate import consolidate
    consolidate('\n'.join(md))
    readme=Path('README.md')
    base=readme.read_text(encoding='utf-8').split('<!-- measured-results -->')[0].rstrip()
    measured=['','<!-- measured-results -->','## Measured release results','',
        '| Model | CV Top-1 | Student Top-1 | Student macro F1 | Student log loss |',
        '|---|---:|---:|---:|---:|']
    for name,r in comparison.items():
        measured.append(f"| {name} | {r['cv_top1_mean']:.4f} | {r['student_top1']:.4f} | {r['student_macro_f1']:.4f} | {r['student_log_loss']:.4f} |")
    measured.extend(['',f"Selected family: **{tuning['selected_model']}**. TPE trials: **{tuning['trials']}**. Tuned artifact deployed: **{tuning['deployed_tuned']}**.",
        '',f"Final classifier Top-1: **{sequence['classifier_top1']:.2%}**; classifier + HMM: **{sequence['hmm_top1']:.2%}**. Strict continuity changes from **{sequence['classifier_continuity']:.2%}** to **{sequence['hmm_continuity']:.2%}**. These are student results against route-derived proxy labels.",
        '',f"All **{len(validation['trips'])} student trips** passed API verification. The data and system/deployment test suites passed (4 + 11 tests). Docker/physical two-laptop deployment and reserved evaluator results remain unverified.",
        '', '[Consolidated report](REPORT.md) | [Interactive route map](artifacts/experiments/route_map.html) | [Release checks](reports/release_validation.json)', ''])
    readme.write_text(base+'\n'+'\n'.join(measured),encoding='utf-8')
    dest=Path('output/pdf'); dest.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8); canvas.drawString(45,24,'PathSnap | measured student-development evaluation'); canvas.drawRightString(550,24,str(doc.page))
    SimpleDocTemplate(str(dest/'PathSnap_Technical_Report.pdf'),rightMargin=45,leftMargin=45,topMargin=40,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
    print('Report written',flush=True)

if __name__=='__main__': main()
