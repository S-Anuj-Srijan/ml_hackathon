"""FastAPI service: prediction has no reference input; scoring remains server-side."""
import json, logging, os, time, uuid
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from data_pipeline import align_route
from decoder import continuity
from .data_loader import load_trip_data
from .predictor import get_predictor
from .scoring import load_ground_truth, score_trip
from backend.paths import PROJECT_ROOT, resource_path
from backend.model_comparison import log_model_comparison

logger=logging.getLogger('pathsnap')
logging.basicConfig(level=logging.INFO,format='%(message)s')
DATA_DIR=resource_path('PATHSNAP_DATA_DIR','Student folder/Student folder/student_test')
GROUND_TRUTH_DIR=resource_path('PATHSNAP_TRUTH_DIR','ground_truth')
trip_registry={}
ground_truths={}
predictor=None
stats={'requests':0,'errors':0,'prediction_seconds':0.}

@asynccontextmanager
async def lifespan(app):
    global predictor
    trip_registry.clear(); ground_truths.clear()
    predictor=get_predictor()  # Fail closed: never silently substitute a heuristic.
    if not DATA_DIR.is_dir(): raise RuntimeError(f'Missing data directory: {DATA_DIR}')
    for path in sorted(DATA_DIR.iterdir()):
        if not path.is_dir(): continue
        meta=load_trip_data(path)
        trip_registry[meta.trip_id]=meta
        try: ground_truths[meta.trip_id]=load_ground_truth(meta.trip_id,ground_truth_dir=GROUND_TRUTH_DIR)
        except FileNotFoundError: pass
    expected=int(os.environ.get('PATHSNAP_EXPECTED_TRIPS','10'))
    if len(trip_registry)!=expected: raise RuntimeError(f'Expected {expected} trips; loaded {len(trip_registry)}')
    logger.info(json.dumps({'event':'ready','trips':len(trip_registry),'model_version':predictor.version}))
    log_model_comparison(logger, predictor.version)
    yield
    trip_registry.clear(); ground_truths.clear(); predictor=None

app=FastAPI(title='PathSnap',version='1.0.0',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=[origin.strip() for origin in os.environ.get('PATHSNAP_CORS_ORIGINS','*').split(',') if origin.strip()],
                   allow_credentials=False,allow_methods=['GET','POST'],allow_headers=['*'])
# Optional convenience for a local preview; the API runs without frontend files.
FRONTEND_DIR=PROJECT_ROOT/'frontend'
if (FRONTEND_DIR/'index.html').is_file():
    app.mount('/ui',StaticFiles(directory=str(FRONTEND_DIR),html=True),name='ui')

@app.get('/health')
def health(): return {'status':'ok'}

@app.get('/ready')
def ready():
    if predictor is None or not trip_registry: raise HTTPException(503,'Not ready')
    return {'status':'ready','trips_loaded':len(trip_registry),'predictor':type(predictor).__name__,'model_version':predictor.version}

@app.get('/metrics')
def metrics(): return stats

@app.get('/trips')
def trips():
    return {'count':len(trip_registry),'trips':[{'trip_id':tid,'n_points':m.n_points,'n_nodes':m.n_nodes,
             'n_arcs':m.n_arcs,'bbox':m.bbox,'track_preview':m.track_lonlat[:5].tolist()} for tid,m in trip_registry.items()]}

@app.post('/predict/{trip_id}')
def predict(trip_id: str):
    if predictor is None: raise HTTPException(503,'Not ready')
    if trip_id not in trip_registry: raise HTTPException(404,'Unknown trip')
    request_id=uuid.uuid4().hex
    stats['requests']+=1
    start=time.perf_counter(); meta=trip_registry[trip_id]
    try:
        result=predictor.predict(meta)
        point_arcs=result['point_arcs']
        geometry=[]
        for arc in dict.fromkeys(point_arcs.tolist()):
            if arc>=0:
                u,v=meta.arcs[arc]
                geometry.append({'arc_id':arc,'from':meta.nodes_lonlat[u].tolist(),'to':meta.nodes_lonlat[v].tolist()})
        response={'trip_id':trip_id,'request_id':request_id,'model_version':predictor.version,
                  'predicted_arcs':point_arcs.tolist(),'n_predicted_arcs':len(geometry),
                  'n_matched_observations':int((point_arcs>=0).sum()),
                  'route_geometry':geometry,'track':meta.track_lonlat.tolist(),'breaks':result['breaks'],
                  'route_continuity':continuity(point_arcs,meta.network)}
        if trip_id in ground_truths:
            route=ground_truths[trip_id]
            labels,_,_=align_route(meta.network,meta.track_xy,meta.track_time,route)
            response['accuracy']=float(np.mean(point_arcs==labels))
            response['accuracy_definition']='agreement with monotone route-derived proxy labels'
            response['route_jaccard']=score_trip(point_arcs[point_arcs>=0],route)
        response['prediction_time_ms']=round((time.perf_counter()-start)*1000,1)
        stats['prediction_seconds']+=time.perf_counter()-start
        logger.info(json.dumps({'event':'prediction','request_id':request_id,'trip_id':trip_id,
             'model_version':predictor.version,'elapsed_ms':response['prediction_time_ms'],'points':meta.n_points,
             'accuracy':response.get('accuracy'),'breaks':len(result['breaks'])}))
        return response
    except Exception:
        stats['errors']+=1
        logger.exception(json.dumps({'event':'prediction_failed','request_id':request_id,'trip_id':trip_id}))
        raise HTTPException(500,detail={'message':'Prediction failed','request_id':request_id})
