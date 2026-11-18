"""Optional local static server. Directly opening index.html needs no Python."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    server = ThreadingHTTPServer(("127.0.0.1", 8080), partial(SimpleHTTPRequestHandler, directory=str(root)))
    print("PathSnap frontend: http://localhost:8080")
    print("Enter Laptop 1's backend URL in the page. Ctrl+C stops this server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
