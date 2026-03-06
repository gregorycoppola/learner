"""Hybrid loss training CLI commands."""
import json
from learner.cli.client import get_client


def add_subparser(subparsers):
    p = subparsers.add_parser("hybrid", help="Hybrid loss: weighted CE for hard vs easy states")
    sub = p.add_subparsers(dest="hybrid_command", required=True)

    run_p = sub.add_parser("run", help="Train with hybrid loss and stream results")
    run_p.add_argument("--machine",          default="incrementer")
    run_p.add_argument("--samples",          type=int,   default=2000)
    run_p.add_argument("--epochs",           type=int,   default=100000)
    run_p.add_argument("--batch-size",       type=int,   default=32)
    run_p.add_argument("--lr",               type=float, default=1e-3)
    run_p.add_argument("--d-model",          type=int,   default=32)
    run_p.add_argument("--layers",           type=int,   default=2)
    run_p.add_argument("--carry-weight",     type=float, default=10.0)
    run_p.add_argument("--tape-weight",      type=float, default=1.0)
    run_p.add_argument("--head-weight",      type=float, default=1.0)
    run_p.add_argument("--state-weight",     type=float, default=1.0)
    run_p.add_argument("--analyze-every",    type=int,   default=5)
    run_p.add_argument("--analyze-samples",  type=int,   default=1000)
    run_p.add_argument("--url",              default="http://localhost:8000")
    run_p.set_defaults(func=cmd_run)


def cmd_run(args):
    payload = {
        "machine":         args.machine,
        "n_samples":       args.samples,
        "n_epochs":        args.epochs,
        "batch_size":      args.batch_size,
        "lr":              args.lr,
        "d_model":         args.d_model,
        "n_layers":        args.layers,
        "carry_weight":    args.carry_weight,
        "tape_weight":     args.tape_weight,
        "head_weight":     args.head_weight,
        "state_weight":    args.state_weight,
        "analyze_every":   args.analyze_every,
        "analyze_samples": args.analyze_samples,
    }

    best_val   = 0.0
    best_epoch = 0

    with get_client(args.url) as client:
        with client.stream(
            "POST", "/api/hybrid/run", json=payload, timeout=None
        ) as resp:
            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                if not raw_line.startswith("data: "):
                    continue
                event = json.loads(raw_line[6:])

                if event["type"] == "init":
                    print(f"Machine        : {event['machine']}")
                    print(f"Samples        : train {event['n_train']} "
                          f"/ val {event['n_val']}")
                    print(f"Carry weight   : {event['carry_weight']}x")
                    hard = event.get('hard_states', [])
                    if hard:
                        print(f"Hard states    : {', '.join(hard)}")
                    print(f"Head weights   : tape={event['tape_weight']}  "
                          f"head={event['head_weight']}  "
                          f"state={event['state_weight']}")
                    print(f"Analyze every  : {event['analyze_every']} epochs")
                    print()
                    print(f"{'Ep':<6} {'Loss':<12} {'L-Easy':<12} "
                          f"{'L-Hard':<12} {'Val Acc':<10} {'Best'}")
                    print("-" * 56)

                elif event["type"] == "epoch":
                    ep   = event["epoch"]
                    vacc = event["val_acc"]
                    if vacc > best_val:
                        best_val   = vacc
                        best_epoch = ep
                        marker = "★"
                    else:
                        marker = ""
                    print(
                        f"{ep:<6} "
                        f"{event['train_loss']:<12.6f} "
                        f"{event['loss_easy']:<12.6f} "
                        f"{event['loss_hard']:<12.6f} "
                        f"{vacc:<10.4f} "
                        f"{marker}",
                        flush=True,
                    )

                elif event["type"] == "analysis":
                    _print_analysis(event)

                elif event["type"] == "done":
                    print()
                    if event.get("stopped_early"):
                        print(f"✓ Solved in {event['epoch']} epochs — val acc 100%")
                    else:
                        print(f"Done.  Best val acc: {best_val:.1%}  at epoch {best_epoch}")


def _print_analysis(event: dict):
    epoch   = event["epoch"]
    overall = event["overall_acc"]
    cats    = event.get("category_summary", {})
    table   = event["table"]

    print()
    print(f"  ── Epoch {epoch}  overall {overall:.1%} ──")
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
        marker = "  ◀" if row["acc"] < 0.90 and row["n"] > 10 else ""
        print(
            f"  {feat:<22} {str(row['value']):<16} {row['n']:>5}  "
            f"{row['acc']:>6.1%}  {row['tape_acc']:>6.1%}  "
            f"{row['head_acc']:>6.1%}  {row['state_acc']:>6.1%}"
            f"{marker}",
            flush=True,
        )
    print()