"""Training CLI commands."""
from learner.cli.client import get_client


def add_subparser(subparsers):
    p = subparsers.add_parser("train", help="Train the transformer")
    sub = p.add_subparsers(dest="train_command", required=True)

    run_p = sub.add_parser("run", help="Train on a TM and print loss curve")
    run_p.add_argument("--machine", default="incrementer")
    run_p.add_argument("--samples", type=int, default=2000)
    run_p.add_argument("--epochs", type=int, default=20)
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

    print(f"Training on {args.machine} — {args.samples} samples, {args.epochs} epochs...")
    print()

    with get_client(args.url) as client:
        resp = client.post("/api/train/run", json=payload, timeout=300.0)
        resp.raise_for_status()
        data = resp.json()

    print(f"Machine  : {data['machine']}")
    print(f"d_input  : {data['d_input']}")
    print(f"Train/val: {data['n_train']} / {data['n_val']}")
    print()
    print(f"{'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} {'Val Acc':<10}")
    print("-" * 50)
    for row in data["history"]:
        print(f"{row['epoch']:<8} {row['train_loss']:<14.6f} {row['val_loss']:<14.6f} {row['val_acc']:<10.4f}")

    print()
    print(f"Final val accuracy : {data['final_val_acc']:.1%}")
    print(f"Final val loss     : {data['final_val_loss']:.6f}")