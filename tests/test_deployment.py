"""Checks for running a portable frontend separately from the model server."""
import importlib.util
import os
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import backend.paths as paths
from backend.api.main import app


class ResourcePathTests(unittest.TestCase):
    def test_default_and_relative_override_ignore_working_directory(self):
        variable = 'PATHSNAP_TEST_RESOURCE_PATH'
        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            with patch.dict(os.environ):
                os.environ.pop(variable, None)
                self.assertEqual(
                    paths.resource_path(variable, 'artifacts/model/best_model.joblib'),
                    paths.PROJECT_ROOT / 'artifacts/model/best_model.joblib',
                )
                os.environ[variable] = 'custom/data'
                self.assertEqual(
                    paths.resource_path(variable, 'unused'),
                    paths.PROJECT_ROOT / 'custom/data',
                )

    def test_absolute_override_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'external-model.joblib'
            with patch.dict(os.environ, {'PATHSNAP_TEST_RESOURCE_PATH': str(target)}):
                self.assertEqual(paths.resource_path('PATHSNAP_TEST_RESOURCE_PATH', 'unused'), target)


class SplitDeploymentTests(unittest.TestCase):
    def setUp(self):
        # No context manager: deployment HTTP checks do not start model/data loading.
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_frontend_origins_can_read_api_and_preflight_prediction(self):
        for origin in ('null', 'http://localhost:8080'):
            with self.subTest(origin=origin):
                response = self.client.get('/health', headers={'Origin': origin})
                self.assertEqual(response.status_code, 200)
                self.assertIn(response.headers.get('access-control-allow-origin'), ('*', origin))
                preflight = self.client.options('/predict/example', headers={
                    'Origin': origin,
                    'Access-Control-Request-Method': 'POST',
                    'Access-Control-Request-Headers': 'Content-Type',
                })
                self.assertEqual(preflight.status_code, 200, preflight.text)
                self.assertIn(preflight.headers.get('access-control-allow-origin'), ('*', origin))
                self.assertIn('POST', preflight.headers['access-control-allow-methods'])

    def test_frontend_preview_serves_portable_assets(self):
        response = self.client.get('/ui/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/html', response.headers['content-type'])
        for filename in ('config.js', 'app.js', 'styles.css', 'vendor/leaflet/leaflet.js', 'vendor/leaflet/leaflet.css'):
            with self.subTest(asset=filename):
                asset = self.client.get('/ui/' + filename)
                self.assertEqual(asset.status_code, 200, filename)
                self.assertTrue(asset.content, filename)

    def test_api_can_import_and_serve_without_frontend_directory(self):
        # Load a separate module so this check does not alter the shared test app.
        source = paths.PROJECT_ROOT / 'backend/api/main.py'
        spec = importlib.util.spec_from_file_location('backend.api._deployment_test_main', source)
        module = importlib.util.module_from_spec(spec)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(paths, 'PROJECT_ROOT', Path(directory)):
                spec.loader.exec_module(module)
            client = TestClient(module.app)
            self.addCleanup(client.close)
            self.assertEqual(client.get('/health').status_code, 200)
            self.assertEqual(client.get('/ui/').status_code, 404)


if __name__ == '__main__':
    unittest.main()
