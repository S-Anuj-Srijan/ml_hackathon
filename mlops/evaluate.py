"""Evaluate all ten student trips; generate reproducible map and measured results."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
import folium
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from api.data_loader import load_trip_data
from api.predictor import get_predictor
from api.scoring import score_trip
from data_pipeline import read_file, align_route, make_map
from decoder import continuity
from mlops.train import save

def main():
    out=Path('artifacts/experiments'); predictor=get_predictor()
    rows=[]
    features=pd.read_parquet('artifacts/data/features.parquet')
    student=features[features.split=='student'].reset_index(drop=True)
    comparison=json.loads((out/'model_comparison.json').read_text())
    model_cont={name:[] for name in comparison}
    for folder in sorted(Path('Student folder/Student folder/student_test').iterdir()):
        if not folder.is_dir(): continue
        meta=load_trip_data(folder)
        route=read_file(folder/(folder.name+'.route'),'route').ravel()
        labels,_,_=align_route(meta.network,meta.track_xy,meta.track_time,route)
        start=time.perf_counter(); result=predictor.predict(meta); elapsed=time.perf_counter()-start
        row={'trip_id':meta.trip_id,'observations':len(labels),'seconds':elapsed,'breaks':len(result['breaks'])}
        for mode,key in [('classifier','classifier_arcs'),('hmm','point_arcs')]:
            pred=result[key]
            row[mode+'_top1']=float(np.mean(pred==labels))
            row[mode+'_continuity']=continuity(pred,meta.network)
            row[mode+'_route_jaccard']=score_trip(pred[pred>=0],route)
            row[mode+'_exact_sequence']=bool(np.array_equal(pred,labels))
        rows.append(row)
        mask=student.trip_id==meta.trip_id
        for name in comparison:
            f=student.loc[mask,['obs_idx','arc_id']].copy()
            f['p']=np.load(out/f'{name}_student_probabilities.npy',mmap_mode='r')[mask]
            picks=f.loc[f.groupby('obs_idx').p.idxmax()].sort_values('obs_idx')
            pred=np.full(len(labels),-1,dtype=int); pred[picks.obs_idx]=picks.arc_id
            model_cont[name].append((continuity(pred,meta.network),len(labels)-1))
        if len(rows)==1:
            make_map(meta.network,np.column_stack([meta.track_lonlat,meta.track_time]),route,out/'source_map.html')
            m=folium.Map(location=meta.track_lonlat.mean(0)[::-1].tolist(),zoom_start=13,tiles=None)
            ends=meta.nodes_lonlat[meta.arcs]
            lo,hi=meta.track_lonlat.min(0)-.005,meta.track_lonlat.max(0)+.005
            visible=((ends.max(1)>=lo)&(ends.min(1)<=hi)).all(1)
            for title,lines,color,width in [('Roads',ends[visible],'#999',1),('Reference',ends[route],'green',5),('Predicted HMM',ends[result['point_arcs'][result['point_arcs']>=0]],'red',3)]:
                layer=folium.FeatureGroup(name=title)
                folium.GeoJson({'type':'MultiLineString','coordinates':lines.tolist()},style_function=lambda _,c=color,w=width:{'color':c,'weight':w}).add_to(layer)
                layer.add_to(m)
            gps=folium.FeatureGroup(name='Raw GPS (blue)')
            for lon,lat in meta.track_lonlat: folium.CircleMarker([lat,lon],radius=2,color='blue').add_to(gps)
            gps.add_to(m); folium.LayerControl().add_to(m)
            m.fit_bounds([lo[::-1].tolist(),hi[::-1].tolist()]); m.save(str(out/'route_map.html'))
            fig,ax=plt.subplots(figsize=(8,5))
            for lines,c,w,label in [(ends[visible],'#ccc',.5,'Road network'),(ends[route],'green',3,'Reference'),(ends[result['point_arcs'][result['point_arcs']>=0]],'red',1,'HMM')]:
                from matplotlib.collections import LineCollection
                ax.add_collection(LineCollection(lines,colors=c,linewidths=w,label=label))
            ax.scatter(*meta.track_lonlat.T,s=2,c='blue',label='GPS'); ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.legend()
            ax.set(xlabel='Longitude',ylabel='Latitude',title='Student trip '+meta.trip_id); fig.tight_layout(); fig.savefig(out/'route_map.png',dpi=170); plt.close(fig)
        print(meta.trip_id,row,flush=True)
        pd.DataFrame(rows).to_csv(out/'sequence_metrics.csv',index=False)
    frame=pd.DataFrame(rows)
    summary={'trips':len(rows),'observations':int(frame.observations.sum())}
    for col in frame.columns:
        if col.startswith(('classifier_','hmm_')):
            weights=frame.observations-1 if col.endswith('continuity') else frame.observations
            summary[col]=float(np.average(frame[col],weights=weights)) if not col.endswith('exact_sequence') else float(frame[col].mean())
    save(out/'sequence_summary.json',summary)
    for name,vals in model_cont.items(): comparison[name]['student_route_continuity']=float(np.average([v[0] for v in vals],weights=[v[1] for v in vals]))
    save(out/'model_comparison.json',comparison)

if __name__=='__main__': main()
