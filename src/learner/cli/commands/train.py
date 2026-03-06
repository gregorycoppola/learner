"""Training CLI commands."""
from learner.cli.client import get_client
import json


def add_subparser(subparsers):
    p = subparsers.add_parser("train", help="Train the transformer")
    sub = p.add_subparsers(dest="train_command", required=True)

    run_p = sub.add_parser("run", help="Train on a TM and stream loss curve")
    run_p.add_argument("--machine",           default="incrementer")
    run_p.add_argument("--samples",           type=int,   default=2000)
    run_p.add_argument("--epochs",            type=int,   default=100)
    run_p.add_argument("--batch-size",        type=int,   default=32)
    run_p.add_argument("--lr",                type=float, default=1e-3)
    run_p.add_argument("--d-model",           type=int,   default=32)
    run_p.add_argument("--layers",            type=int,   default=2)
    run_p.add_argument("--analyze-every",     type=int,   default=5)
    run_p.add_argument("--analyze-samples",   type=int,   default=500)
    run_p.add_argument("--hard-multiplier",   type=float, default=4.0)
    run_p.add_argument("--easy-multiplier",   type=float, default=0.5)
    run_p.add_argument("--url",               default="http://localhost:8000")
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
        "hard_multiplier":  args.hard_multiplier,
        "easy_multiplier":  args.easy_multiplier,
    }

    best_acc   = 0.0
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
                    print(f"{'Ep':<6} {'T-Loss':<12} {'V-Loss':<12} "
                          f"{'V-Acc':<8} {'W-min':<8} {'W-max':<8} {'Best'}")
                    print("-" * 62)

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
                        f"{ep:<6} "
                        f"{event['train_loss']:<12.6f} "
                        f"{event['val_loss']:<12.6f} "
                        f"{acc:<8.4f} "
                        f"{event['w_min']:<8.3f} "
                        f"{event['w_max']:<8.3f} "
                        f"{marker}",
                        flush=True,
                    )

                elif event["type"] == "analysis":
                    _print_analysis(event)

                elif event["type"] == "done":
                    print()
                    print(f"Done.  Best val accuracy: {best_acc:.1%} "
                          f"at epoch {best_epoch}")


def _print_analysis(event: dict):
    epoch    = event["epoch"]
    overall  = event["overall_acc"]
    mastered = event["n_mastered"]
    total    = event["n_total"]
    w_min    = event["weight_min"]
    w_max    = event["weight_max"]
    table    = event["table"]
    cats     = event.get("category_summary", {})

    print()
    print(f"  ── Epoch {epoch}  overall {overall:.1%}  "
          f"mastered {mastered}/{total}  "
          f"weights [{w_min:.2f}–{w_max:.2f}] ──")

    # One-line category summary
    cat_str = "  ".join(f"{k}:{v:.1%}" for k, v in sorted(cats.items()))
    if cat_str:
        print(f"  States: {cat_str}")

    print(f"  {'Feature':<22} {'Value':<16} {'N':>5}  "
          f"{'Acc':>7}  {'Tape':>7}  {'Head':>7}  {'State':>7}")
    print("  " + "-" * 72)

    current_feature = None
    for row in table:
        feat = row["feature"]
        if feat != current_feature:
            if current_feature is not None:
                print()
            current_feature = feat

        n      = row["n"]
        acc    = row["acc"]
        marker = "  ◀" if acc < 0.90 and n > 10 else ""

        print(
            f"  {feat:<22} {str(row['value']):<16} {n:>5}  "
            f"{acc:>6.1%}  {row['tape_acc']:>6.1%}  "
            f"{row['head_acc']:>6.1%}  {row['state_acc']:>6.1%}"
            f"{marker}",
            flush=True,
        )

    print()