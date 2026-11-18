"""Build the portable UI ZIP; never include model, data or Python environments."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / 'frontend'


def main():
    required = ['index.html', 'styles.css', 'app.js', 'config.js', 'README.md',
                'serve.py', 'start_frontend.bat', 'vendor/leaflet/leaflet.js',
                'vendor/leaflet/leaflet.css', 'vendor/leaflet/LICENSE']
    for name in required:
        if not (FRONTEND / name).is_file():
            raise FileNotFoundError(f'Missing frontend asset: {name}')
    files = sorted(path for path in FRONTEND.rglob('*') if path.is_file()
                   and '__pycache__' not in path.parts and path.suffix != '.pyc')
    dest = ROOT / 'dist'
    dest.mkdir(exist_ok=True)
    archive = dest / 'PathSnap-frontend.zip'
    with ZipFile(archive, 'w', compression=ZIP_DEFLATED) as output:
        for path in files:
            output.write(path, path.relative_to(ROOT).as_posix())
    with ZipFile(archive) as output:
        if output.testzip() is not None:
            raise ValueError('Frontend ZIP integrity check failed')
        assert all(name.startswith('frontend/') for name in output.namelist())
    summary = {'archive': archive.name, 'bytes': archive.stat().st_size,
               'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
               'files': {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in files}}
    (dest / 'frontend-manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'{archive} ({summary["bytes"]:,} bytes; {len(files)} files)')


if __name__ == '__main__':
    main()
