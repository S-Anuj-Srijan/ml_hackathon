"""Start the API from the repository root or from inside backend/."""
import argparse
import os
from pathlib import Path
import socket
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_lan_ips():
    addresses = set()
    try:
        addresses.update(ip for ip in socket.gethostbyname_ex(socket.gethostname())[2]
                         if not ip.startswith('127.'))
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            # UDP connect discovers the route's local address; no packet is sent.
            sock.connect(('8.8.8.8', 80))
            addresses.add(sock.getsockname()[0])
    except OSError:
        pass
    return sorted(addresses)


def main():
    parser = argparse.ArgumentParser(description='PathSnap backend for Laptop 1')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    import uvicorn

    print('\nPathSnap backend - leave this terminal running', flush=True)
    print(f'Readiness: http://localhost:{args.port}/ready', flush=True)
    print(f'API docs:  http://localhost:{args.port}/docs', flush=True)
    print('On Laptop 2: open frontend/index.html, then enter one of these API URLs:', flush=True)
    for ip in get_lan_ips():
        print(f'  http://{ip}:{args.port}', flush=True)
    if (PROJECT_ROOT/'frontend'/'index.html').is_file():
        print(f'Optional local UI: http://localhost:{args.port}/ui/', flush=True)
    uvicorn.run('backend.api.main:app', host=args.host, port=args.port,
                reload=args.reload, reload_dirs=[str(PROJECT_ROOT/'backend')] if args.reload else None,
                log_level='info')


if __name__ == '__main__':
    main()
