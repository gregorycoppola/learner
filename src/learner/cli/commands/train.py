"""Training CLI commands."""
from learner.cli.client import get_client
import json


def add_subparser(subparsers):
    p = subparsers.add_parser("train", help="Train the transformer")
    sub = p.add_subparsers(dest="train_command", required=True)

    run_p = sub.add_parser("run", help="Train on a TM and stream loss curve")
    run_p.add_argument("--machine", default="incrementer")
    run_p.add_argument("--samples", type=int, default=2000)
    run_p.add_argument("--epochs", type=int, default=100)
    run_p.add_argument("--batch-size", type=int, default=32)
    run_p.add_argument("--lr", type=float, default=1e-3)
    run_p.add_argument("--d-model", type=int, default=32)
    run_p.add_argument("--layers", type=int, default=2)
    run_p.add_argument("--url", default="http://localhost:8000")
    run_p.set_defaults(func=cmd_run)


def cmd_run(args):
    payload = {
        "machine": args.machine,
        "n_samples": args.samples,
        "n_epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "d_model": args.d_model,
        "n_layers": args.layers,
    }

    best_acc = 0.0
    best_epoch = 0

    with get_client(args.url) as client:
        with client.stream("POST", "/api/train/run", json=payload, timeout=None) as resp:
            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                if not raw_line.startswith("data: "):
                    continue
                event = json.loads(raw_line[6:])

                if event["type"] == "init":
                    print(f"Machine  : {event['machine']}")
                    print(f"Samples  : {event['n_samples']}  "
                          f"(train {event['n_train']} / val {event['n_val']})")
                    print(f"d_input  : {event['d_input']}")
                    print(f"Epochs   : {event['n_epochs']}")
                    print()
                    print(f"{'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} "
                          f"{'Val Acc':<10} {'Best':<6}")
                    print("-" * 56)

                elif event["type"] == "epoch":
                    ep = event["epoch"]
                    acc = event["val_acc"]
                    if acc > best_acc:
                        best_acc = acc
                        best_epoch = ep
                        best_marker = "★"
                    else:
                        best_marker = ""
                    print(
                        f"{ep:<8} "
                        f"{event['train_loss']:<14.6f} "
                        f"{event['val_loss']:<14.6f} "
                        f"{acc:<10.4f} "
                        f"{best_marker}",
                        flush=True,
                    )

                elif event["type"] == "done":
                    print()
                    print(f"Done. Best val accuracy: {best_acc:.1%} at epoch {best_epoch}")