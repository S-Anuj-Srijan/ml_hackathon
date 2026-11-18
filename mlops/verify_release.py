"""Release verification over all ten available student trips, never evaluator data."""
import hashlib, json, os, platform, time
from pathlib import Path
from fastapi.testclient import TestClient
from api.main import app, DATA_DIR

def main():
    expected=Path('Student folder/Student folder/student_test').resolve()
    if DATA_DIR.resolve()!=expected:
        raise ValueError('Release-development verification only accepts the student_test partition')
    rows=[]
    with TestClient(app) as client:
        ready=client.get('/ready'); ready.raise_for_status()
        listing=client.get('/trips').json()
        assert listing['count']==10
        for trip in listing['trips']:
            start=time.perf_counter()
            response=client.post('/predict/'+trip['trip_id']); response.raise_for_status()
            data=response.json()
            assert len(data['predicted_arcs'])==trip['n_points']
            assert all(a==-1 or 0<=a<trip['n_arcs'] for a in data['predicted_arcs'])
            assert not ({'ground_truth','true_arc_id','reference_route','route_occurrence'}&set(data))
            rows.append({'trip_id':trip['trip_id'],'status':response.status_code,'points':len(data['predicted_arcs']),
                         'seconds':time.perf_counter()-start,'proxy_accuracy':data.get('accuracy')})
            print(rows[-1],flush=True)
        assert client.post('/predict/not-a-trip').status_code==404
        assert client.get('/ui/').status_code==200
        files=[Path('requirements-lock.txt'),Path('Dockerfile'),Path('compose.yaml'),Path('decoder.py'),Path('data_pipeline.py')]
        files += list(Path('api').rglob('*.py')) + list(Path('api/static').glob('*.html')) + list(Path('mlops').glob('*.py'))
        report={'readiness':ready.json(),'trips':rows,'all_passed':True,
                'runtime':{'python':platform.python_version(),'platform':platform.platform(),
                           'processor':platform.processor(),'logical_cpus':os.cpu_count()},
                'release_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'scope':'ten student-development trips via FastAPI TestClient; no physical two-laptop or Docker execution'}
    Path('reports').mkdir(exist_ok=True)
    Path('reports/release_validation.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__': main()
