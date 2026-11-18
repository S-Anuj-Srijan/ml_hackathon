"""Run the PathSnap API server.

Usage:
    python run_server.py                       # binds 0.0.0.0:8000 (accessible on LAN)
    python run_server.py --port 9000           # custom port
"""
import argparse
import socket
import uvicorn


def get_lan_ips() -> list[str]:
    """Find local network IPv4 addresses to show Laptop 2 connection URLs."""
    ips = set()
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass
    # Fallback via dummy connection
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return sorted(list(ips))


def main():
    parser = argparse.ArgumentParser(description="PathSnap API Server")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0 for LAN / two-laptop setup)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development")
    args = parser.parse_args()

    lan_ips = get_lan_ips()
    lan_urls = [f"http://{ip}:{args.port}/ui" for ip in lan_ips]

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ⚡  PathSnap API Server  ⚡                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Local UI (Laptop 1):   http://localhost:{args.port}/ui              
║  API Docs:              http://localhost:{args.port}/docs            
║                                                                  
║  🌐 TWO-LAPTOP SETUP (Open this in browser on Laptop 2):        """)
    for url in lan_urls:
        print(f"║     👉 {url}")
    print(f"""╚══════════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
