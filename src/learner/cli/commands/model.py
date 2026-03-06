"""Model checkpoint CLI commands."""
from learner.cli.client import get_client


def add_subparser(subparsers):
    p = subparsers.add_parser("model", help="Manage model checkpoints")
    sub = p.add_subparsers(dest="model_command", required=True)

    list_p = sub.add_parser("list", help="List saved checkpoints")
    list_p.add_argument("--url", default="http://localhost:8000")
    list_p.set_defaults(func=cmd_list)

    del_p = sub.add_parser("delete", help="Delete a checkpoint")
    del_p.add_argument("name", help="Checkpoint name (without .pt)")
    del_p.add_argument("--url", default="http://localhost:8000")
    del_p.set_defaults(func=cmd_delete)


def cmd_list(args):
    with get_client(args.url) as client:
        resp = client.get("/api/model/list")
        resp.raise_for_status()
        data = resp.json()

    ckpts = data["checkpoints"]
    if not ckpts:
        print("No checkpoints saved yet.")
        return

    print(f"{'Name':<40} {'Machine':<14} {'Phase':<6} "
          f"{'Epoch':>6}  {'Val Acc':>8}  {'d_model':>7}")
    print("-" * 85)
    for c in ckpts:
        arch = c.get("arch", {})
        print(
            f"{c['name']:<40} {c['machine']:<14} {c['phase']:<6} "
            f"{c['epoch']:>6}  {c['val_acc']:>7.1%}  "
            f"{arch.get('d_model', '?'):>7}"
        )


def cmd_delete(args):
    with get_client(args.url) as client:
        resp = client.delete(f"/api/model/{args.name}")
        resp.raise_for_status()
    print(f"Deleted: {args.name}")