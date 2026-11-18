"""Verified artifact, shared feature extraction, classifier and HMM inference."""
import hashlib
import json
import os
from pathlib import Path
import joblib
import numpy as np
import sklearn
from data_pipeline import FEATURE_COLUMNS, candidate_features
from decoder import ViterbiDecoder
from backend.paths import resource_path

class ModelPredictor:
    def __init__(self, path):
        path = Path(path)
        manifest = json.loads(path.with_suffix('.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
            raise ValueError('Model checksum does not match manifest')
        if manifest.get('sklearn', sklearn.__version__) != sklearn.__version__:
            raise ValueError('Model scikit-learn version differs; install requirements-lock.txt')
        self.bundle = joblib.load(path)  # Trusted, locally trained artifacts only.
        if self.bundle['features'] != FEATURE_COLUMNS:
            raise ValueError('Model feature schema does not match inference')
        self.version = manifest['run_id']
        self.decoder = ViterbiDecoder(**self.bundle['decoder'])
    def predict(self, meta):
        track = np.column_stack([meta.track_lonlat, meta.track_time])
        f, _ = candidate_features(meta.network, track, **self.bundle['candidates'])
        f[FEATURE_COLUMNS] = f[FEATURE_COLUMNS].astype('float32')
        p = self.bundle['model'].predict_proba(f[FEATURE_COLUMNS])[:, 1] if len(f) else np.array([])
        out = self.decoder.decode(meta.network, meta.track_xy, meta.track_time, f, p)
        out['predicted_arcs'] = out['point_arcs'].tolist()
        return out

def get_predictor():
    return ModelPredictor(resource_path('PATHSNAP_MODEL', 'artifacts/model/best_model.joblib'))
