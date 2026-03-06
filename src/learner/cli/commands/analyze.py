"""Error analysis CLI commands."""
from learner.cli.client import get_client


def add_subparser(subparsers):
    p = subparsers.add_parser("analyze", help="Analyze model errors by feature")
    sub = p.add_subparsers(dest="analyze_command", required=True)

    run_p = sub.add_parser("run", help="Train model and print error breakdown table")
    run_p.add_argument("--machine",  default="incrementer")
    run_p.add_argument("--samples",  type=int, default=2000,  help="Analysis samples")
    run_p.add_argument("--train-samples", type=int, default=10000)
    run_p.add_argument("--epochs",   type=int, default=50)
    run_p.add_argument("--d-model",  type=int, default=32)
    run_p.add_argument("--layers",   type=int, default=2)
    run_p.add_argument("--url",      default="http://localhost:8000")
    run_p.set_defaults(func=cmd_run)


def cmd_run(args):
    payload = {
        "machine":        args.machine,
        "n_samples":      args.samples,
        "n_train_samples": args.train_samples,
        "n_epochs":       args.epochs,
        "d_model":        args.d_model,
        "n_layers":       args.layers,
    }

    print(f"Training + analyzing {args.machine}...")
    print()

    with get_client(args.url) as client:
        resp = client.post("/api/analyze/run", json=payload, timeout=600.0)
        resp.raise_for_status()
        data = resp.json()

    print(f"Analyzed {data['n_analyzed']} steps  —  "
          f"overall accuracy {data['overall_acc']:.1%}")
    print()

    # Group rows by feature for display
    current_feature = None
    print(f"{'Feature':<22} {'Value':<16} {'N':>6}  "
          f"{'Acc':>7}  {'Tape':>7}  {'Head':>7}  {'State':>7}")
    print("-" * 78)

    for row in data["table"]:
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

        # Highlight bad rows
        marker = "  ◀" if acc < 0.85 and n > 10 else ""

        print(
            f"  {feat:<20} {str(row['value']):<16} {n:>6}  "
            f"{acc:>6.1%}  {ta:>6.1%}  {ha:>6.1%}  {sa:>6.1%}"
            f"{marker}"
        )