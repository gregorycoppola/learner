"""GRPO training CLI commands."""
import json
from learner.cli.client import get_client


def add_subparser(subparsers):
    p = subparsers.add_parser("grpo", help="GRPO training with TM verifier")
    sub = p.add_subparsers(dest="grpo_command", required=True)

    run_p = sub.add_parser("run", help="Train with GRPO and stream results")
    run_p.add_argument("--machine",         default="incrementer")
    run_p.add_argument("--samples",         type=int,   default=2000)
    run_p.add_argument("--epochs",          type=int,   default=200)
    run_p.add_argument("--batch-size",      type=int,   default=16)
    run_p.add_argument("--lr",              type=float, default=1e-4)
    run_p.add_argument("--d-model",         type=int,   default=32)
    run_p.add_argument("--layers",          type=int,   default=2)
    run_p.add_argument("--K",               type=int,   default=8,
                       help="Candidates per example")
    run_p.add_argument("--kl-coef",         type=float, default=0.01)
    run_p.add_argument("--analyze-every",   type=int,   default=5)
    run_p.add_argument("--analyze-samples", type=int,   default=500)
    run_p.add_argument("--url",             default="http://localhost:8000")
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
        "K":               args.K,
        "kl_coef":         args.kl_coef,
        "analyze_every":   args.analyze_every,
        "analyze_samples": args.analyze_samples,
    }

    best_acc   = 0.0
    best_epoch = 0

    with get_client(args.url) as client:
        with client.stream("POST", "/api/grpo/run",
                           json=payload, timeout=None) as resp:
            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                if not raw_line.startswith("data: "):
                    continue
                event = json.loads(raw_line[6:])

                if event["type"] == "init":
                    print(f"Machine        : {event['machine']}")
                    print(f"Samples        : train {event['n_train']} "
                          f"/ val {event['n_val']}")
                    print(f"d_input        : {event['d_input']}  "
                          f"n_tape: {event['n_tape']}  "
                          f"n_states: {event['n_states']}")
                    print(f"K              : {event['K']} candidates/example")
                    print(f"Analyze every  : {event['analyze_every']} epochs")
                    print()
                    print(f"{'Ep':<6} {'Loss':<12} {'Train Rew':<12} "
                          f"{'Val Acc':<10} {'Best'}")
                    print("-" * 48)

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
                        f"{event['train_reward']:<12.4f} "
                        f"{acc:<10.4f} "
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
    epoch   = event["epoch"]
    overall = event["overall_acc"]
    cats    = event.get("category_summary", {})
    table   = event["table"]

    print()
    print(f"  ── GRPO Epoch {epoch}  overall {overall:.1%} ──")
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