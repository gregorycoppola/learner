"""Server start/stop commands."""
import subprocess
import sys


def add_subparser(subparsers):
    p = subparsers.add_parser("server", help="Manage the Learner API server")
    sub = p.add_subparsers(dest="server_command", required=True)

    start = sub.add_parser("start", help="Start the server")
    start.add_argument("--host", default="0.0.0.0")
    start.add_argument("--port", type=int, default=8000)
    start.add_argument("--reload", action="store_true", default=True)
    start.set_defaults(func=cmd_start)


def cmd_start(args):
    print(f"🚀 Starting Learner API on {args.host}:{args.port}")
    cmd = [
        sys.executable, "-m", "uvicorn",
        "learner.server.main:app",
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.reload:
        cmd.append("--reload")
    subprocess.run(cmd)