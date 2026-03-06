"""Training CLI commands."""
from learner.cli.client import get_client
import json


def add_subparser(subparsers):
    p = subparsers.add_parser("train", help="Train the transformer")
    sub = p.add_subparsers(dest="train_command", required=True)

    run_p = sub.add_parser("run", help="Train on a TM and stream loss curve")
    run_p.add_argument("--machine",         default="incrementer")
    run_p.add_argument("--samples",         type=int, default=2000)
    run_p.add_argument("--epochs",          type=int, default=100)
    run_p.add_argument("--batch-size",      type=int, default=32)
    run_p.add_argument("--lr",              type=float, default=1e-3)
    run_p.add_argument("--d-model",         type=int, default=32)
    run_p.add_argument("--layers",          type=int, default=2)
    run_p.add_argument("--analyze-every",   type=int, default=5,
                       help="Print error breakdown every N epochs")
    run_p.add_argument("--analyze-samples", type=int, default=500,
                       help="Samples to use for each analysis")
    run_p.add_argument("--url",             default="http://localhost:8000")
    run_p.set_defaults(func=cmd_run)


def cmd_run(args):
    payload = {
        "machine":          args.machine,
        "n_samples":        args.samples,
        "n_epochs":         args.epochs,
        "batch_size":       args.batch_size,
        "lr":               args.lr,
        "d_model":          args.d_model,
        "n_layers":         args.layers,
        "analyze_every":    args.analyze_every,
        "analyze_samples":  args.analyze_samples,
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
                    print(f"Machine        : {event['machine']}")
                    print(f"Samples        : {event['n_samples']}  "
                          f"(train {event['n_train']} / val {event['n_val']})")
                    print(f"d_input        : {event['d_input']}")
                    print(f"Epochs         : {event['n_epochs']}")
                    print(f"Analyze every  : {event['analyze_every']} epochs")
                    print()
                    print(f"{'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14} "
                          f"{'Val Acc':<10} {'Best'}")
                    print("-" * 56)

                elif event["type"] == "epoch":
                    ep  = event["epoch"]
                    acc = event["val_acc"]
                    if acc > best_acc:
                        best_acc   = acc
                        best_epoch = ep
                        marker = "★"
                    else:
                        marker = ""
                    print(
                        f"{ep:<8} "
                        f"{event['train_loss']:<14.6f} "
                        f"{event['val_loss']:<14.6f} "
                        f"{acc:<10.4f} "
                        f"{marker}",
                        flush=True,
                    )

                elif event["type"] == "analysis":
                    _print_analysis(event)

                elif event["type"] == "done":
                    print()
                    print(f"Done.  Best val accuracy: {best_acc:.1%} at epoch {best_epoch}")


def _print_analysis(event: dict):
    epoch = event["epoch"]
    overall = event["overall_acc"]
    table = event["table"]

    print()
    print(f"  ── Analysis at epoch {epoch}  (overall {overall:.1%}) ──")
    print(f"  {'Feature':<22} {'Value':<16} {'N':>6}  "
          f"{'Acc':>7}  {'Tape':>7}  {'Head':>7}  {'State':>7}")
    print("  " + "-" * 74)

    current_feature = None
    for row in table:
        feat = row["feature"]
        if feat != current_feature:
            if current_feature is not None:
                print()
            current_feature = feat

        n   = row["n"]
        acc = row["acc"]
        ta  = row["tape_acc"]
        ha  = row["head_acc"]
        sa  = row["state_acc"]
        marker = "  ◀" if acc < 0.85 and n > 10 else ""

        print(
            f"  {feat:<22} {str(row['value']):<16} {n:>6}  "
            f"{acc:>6.1%}  {ta:>6.1%}  {ha:>6.1%}  {sa:>6.1%}"
            f"{marker}",
            flush=True,
        )

    print()