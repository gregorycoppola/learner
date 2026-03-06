"""Health check command."""
from learner.cli.client import get_client


def add_subparser(subparsers):
    p = subparsers.add_parser("health", help="Check server health")
    p.add_argument("--url", default="http://localhost:8000", help="Server URL")
    p.set_defaults(func=cmd_health)


def cmd_health(args):
    try:
        with get_client(args.url) as client:
            resp = client.get("/api/health")
            resp.raise_for_status()
            data = resp.json()
            print(f"✓ {data['service']} v{data['version']} — {data['status']}")
    except Exception as e:
        print(f"✗ Could not reach server at {args.url}: {e}")
        raise SystemExit(1)