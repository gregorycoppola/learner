"""SFT then GRPO training CLI command."""
import json
from learner.cli.client import get_client


def add_subparser(subparsers):
    p = subparsers.add_parser("sft-grpo", help="SFT warmup then GRPO training")
    sub = p.add_subparsers(dest="sft_grpo_command", required=True)

    run_p = sub.add_parser("run", help="Run SFT then GRPO and stream results")
    run_p.add_argument("--machine",          default="incrementer")
    run_p.add_argument("--samples",          type=int,   default=2000)
    run_p.add_argument("--sft-epochs",       type=int,   default=50)
    run_p.add_argument("--sft-threshold",    type=float, default=0.90)
    run_p.add_argument("--sft-lr",           type=float, default=1e-3)
    run_p.add_argument("--grpo-epochs",      type=int,   default=100000)
    run_p.add_argument("--grpo-lr",          type=float, default=1e-4)
    run_p.add_argument("--K",                type=int,   default=8)
    run_p.add_argument("--kl-coef",          type=float, default=0.01)
    run_p.add_argument("--d-model",          type=int,   default=32)
    run_p.add_argument("--layers",           type=int,   default=2)
    run_p.add_argument("--analyze-every",    type=int,   default=5)
    run_p.add_argument("--analyze-samples",  type=int,   default=500)
    run_p.add_argument("--checkpoint-name",  default=None)
    run_p.add_argument("--url",              default="http://localhost:8000")
    run_p.set_defaults(func=cmd_run)


def cmd_run(args):
    payload = {
        "machine":          args.machine,
        "n_samples":        args.samples,
        "sft_max_epochs":   args.sft_epochs,
        "sft_threshold":    args.sft_threshold,
        "sft_lr":           args.sft_lr,
        "grpo_epochs":      args.grpo_epochs,
        "grpo_lr":          args.grpo_lr,
        "K":                args.K,
        "kl_coef":          args.kl_coef,
        "d_model":          args.d_model,
        "n_layers":         args.layers,
        "analyze_every":    args.analyze_every,
        "analyze_samples":  args.analyze_samples,
        "checkpoint_name":  args.checkpoint_name,
    }

    current_phase = None
    sft_best      = 0.0
    grpo_best     = 0.0
    grpo_best_ep  = 0

    with get_client(args.url) as client:
        with client.stream(
            "POST", "/api/sft-grpo/run", json=payload, timeout=None
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
                    print(f"SFT            : up to {event['sft_max_epochs']} epochs "
                          f"(threshold {event['sft_threshold']:.0%})")
                    print(f"GRPO           : {event['grpo_epochs']} epochs  "
                          f"K={event['K']}")
                    print(f"Analyze every  : {event['analyze_every']} epochs")
                    print()

                elif event["type"] == "phase":
                    current_phase = event["phase"]
                    if current_phase == "sft":
                        print("── Phase 1: SFT ─────────────────────────────────")
                        print(f"{'Ep':<6} {'Loss':<12} {'Val Acc':<10} {'Best'}")
                        print("-" * 36)
                    elif current_phase == "grpo":
                        print()
                        print("── Phase 2: GRPO ────────────────────────────────")
                        print(f"{'Ep':<6} {'Loss':<12} {'Reward':<10} "
                              f"{'Val Acc':<10} {'Best'}")
                        print("-" * 46)

                elif event["type"] == "epoch":
                    if event["phase"] == "sft":
                        acc = event["val_acc"]
                        if acc > sft_best:
                            sft_best = acc
                            marker = "★"
                        else:
                            marker = ""
                        print(
                            f"{event['epoch']:<6} "
                            f"{event['train_loss']:<12.6f} "
                            f"{acc:<10.4f} "
                            f"{marker}",
                            flush=True,
                        )
                    elif event["phase"] == "grpo":
                        acc = event["val_acc"]
                        if acc > grpo_best:
                            grpo_best    = acc
                            grpo_best_ep = event["epoch"]
                            marker = "★"
                        else:
                            marker = ""
                        print(
                            f"{event['epoch']:<6} "
                            f"{event['train_loss']:<12.6f} "
                            f"{event['train_reward']:<10.4f} "
                            f"{acc:<10.4f} "
                            f"{marker}",
                            flush=True,
                        )

                elif event["type"] == "sft_done":
                    print()
                    print(f"SFT complete — epoch {event['epoch']}  "
                          f"val_acc {event['val_acc']:.1%}  "
                          f"({event['reason']})")

                elif event["type"] == "checkpoint_saved":
                    print(f"Checkpoint saved: {event['name']}")
                    print()

                elif event["type"] == "analysis":
                    _print_analysis(event)

                elif event["type"] == "done":
                    print()
                    print(f"Done.  Best GRPO val accuracy: "
                          f"{event['best_grpo_acc']:.1%} "
                          f"at epoch {grpo_best_ep}")


def _print_analysis(event: dict):
    phase   = event.get("phase", "?")
    epoch   = event["epoch"]
    overall = event["overall_acc"]
    cats    = event.get("category_summary", {})
    table   = event["table"]

    print()
    print(f"  ── [{phase.upper()}] Epoch {epoch}  overall {overall:.1%} ──")
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