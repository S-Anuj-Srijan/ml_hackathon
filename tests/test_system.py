import hashlib, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import joblib, numpy as np, pandas as pd
from sklearn.dummy import DummyClassifier
from data_pipeline import Network, FEATURE_COLUMNS, candidate_features, project
from decoder import ViterbiDecoder
from api.predictor import ModelPredictor

class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.net=Network(np.array([[8.,50.],[8.001,50.],[8.002,50.],[8.003,50.]]),np.array([[0,1],[1,2],[2,3]]))
    def decode(self, obs, arcs, times, probs):
        f=pd.DataFrame({'obs_id':obs,'arc_id':arcs,'offset_m':np.zeros(len(obs))})
        return ViterbiDecoder().decode(self.net,np.zeros((len(times),2)),np.array(times),f,np.array(probs))
    def test_global_path_not_greedy(self):
        # Arc 2 cannot travel backward to arc 1; Viterbi must choose arc 0 first.
        r=self.decode([0,0,1],[0,2,1],[0,1],[.4,.6,1.])
        self.assertEqual(r['classifier_arcs'].tolist(),[2,1])
        self.assertEqual(r['point_arcs'].tolist(),[0,1])
    def test_missing_and_time_gap(self):
        r=self.decode([0,2,3],[2,0,2],[0,1,2,40],[1,1,1])
        self.assertEqual(r['point_arcs'].tolist(),[2,-1,0,2])
        self.assertEqual([b['reason'] for b in r['breaks']],['no_candidates','time_gap'])
    def test_disconnected_restarts(self):
        r=self.decode([0,1],[2,0],[0,1],[1,1])
        self.assertEqual(r['point_arcs'].tolist(),[2,0])
        self.assertEqual(r['breaks'][0]['reason'],'disconnected')
    def test_empty_candidates(self):
        r=self.decode([],[],[0,1],[])
        self.assertEqual(r['point_arcs'].tolist(),[-1,-1])

class ArtifactTests(unittest.TestCase):
    def test_checksum_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.joblib'; p.write_bytes(b'bad')
            p.with_suffix('.json').write_text(json.dumps({'sha256':'wrong'}))
            with self.assertRaisesRegex(ValueError,'checksum'): ModelPredictor(p)
    def test_api_prediction_truth_isolation(self):
        from fastapi.testclient import TestClient
        import api.main as api
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); trip=root/'trips'/'trip'; trip.mkdir(parents=True)
            np.savetxt(trip/'trip.track',[[8.0002,50.,1],[8.0004,50.,2]])
            np.savetxt(trip/'trip.nodes',[[8.,50.],[8.001,50.]])
            np.savetxt(trip/'trip.arcs',[[0,1]],fmt='%d')
            # Malformed route in prediction directory must never be read.
            (trip/'trip.route').write_text('not input data')
            model=DummyClassifier(strategy='prior').fit(pd.DataFrame(np.zeros((2,9)),columns=FEATURE_COLUMNS),[0,1])
            bundle={'model':model,'features':FEATURE_COLUMNS,'decoder':{},'candidates':{'k':32,'radius':300}}
            p=root/'model.joblib'; joblib.dump(bundle,p)
            p.with_suffix('.json').write_text(json.dumps({'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'run_id':'test'}))
            with patch.dict('os.environ',{'PATHSNAP_MODEL':str(p),'PATHSNAP_EXPECTED_TRIPS':'1'}), patch.object(api,'DATA_DIR',root/'trips'), patch.object(api,'GROUND_TRUTH_DIR',root/'truth'):
                with TestClient(api.app) as client:
                    self.assertEqual(client.get('/ready').status_code,200)
                    self.assertEqual(client.post('/predict/unknown').status_code,404)
                    result=client.post('/predict/trip')
                    self.assertEqual(result.status_code,200,result.text)
                    payload=result.json()
                    self.assertEqual(payload['predicted_arcs'],[0,0])
                    self.assertNotIn('accuracy',payload)
                    self.assertNotIn('true_arc_id',result.text)
                    self.assertNotIn('ground_truth',result.text)
                    self.assertEqual(client.get('/ui/').status_code,200)

if __name__=='__main__': unittest.main()
