"""Turing machine CLI commands."""
from learner.cli.client import get_client


def add_subparser(subparsers):
    p = subparsers.add_parser("tm", help="Turing machine operations")
    sub = p.add_subparsers(dest="tm_command", required=True)

    # tm run
    run_p = sub.add_parser("run", help="Run a TM and display the trace")
    run_p.add_argument("--machine", default="incrementer")
    run_p.add_argument("--int", dest="input_int", type=int, default=None)
    run_p.add_argument("--tape", nargs="+", default=None)
    run_p.add_argument("--url", default="http://localhost:8000")
    run_p.set_defaults(func=cmd_run)

    # tm generate
    gen_p = sub.add_parser("generate", help="Generate training pairs")
    gen_p.add_argument("--machine", default="incrementer")
    gen_p.add_argument("--n", type=int, default=100)
    gen_p.add_argument("--max-val", type=int, default=255)
    gen_p.add_argument("--url", default="http://localhost:8000")
    gen_p.set_defaults(func=cmd_generate)

    # tm machines
    list_p = sub.add_parser("machines", help="List available machines")
    list_p.add_argument("--url", default="http://localhost:8000")
    list_p.set_defaults(func=cmd_machines)


def cmd_run(args):
    from learner.core.tm import format_tape

    payload = {"machine": args.machine}
    if args.input_int is not None:
        payload["input_int"] = args.input_int
    elif args.tape is not None:
        payload["tape"] = args.tape
    else:
        payload["input_int"] = 11  # default: increment 11 → 12

    with get_client(args.url) as client:
        resp = client.post("/api/tm/run", json=payload)
        resp.raise_for_status()
        data = resp.json()

    print(f"Machine : {data['machine']}")
    print(f"Input   : {data['input_int']}  →  {''.join(data['input_tape'])}")
    print(f"Output  : {data['output_int']}  →  {''.join(t for t in data['output_tape'] if t != '_')}")
    print(f"Steps   : {data['steps']}")
    print()
    print(f"{'Step':<6} {'State':<14} Tape")
    print("-" * 50)
    for entry in data["trace"]:
        tape_str = format_tape(entry["tape"], entry["head"])
        halted = " ✓" if entry["halted"] else ""
        print(f"{entry['step']:<6} {entry['state']:<14} {tape_str}{halted}")


def cmd_generate(args):
    payload = {
        "machine": args.machine,
        "n_samples": args.n,
        "max_val": args.max_val,
    }

    with get_client(args.url) as client:
        resp = client.post("/api/tm/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()

    print(f"Generated {data['n_samples']} step pairs from {data['machine']}")
    print()
    print(f"{'#':<5} {'state_before':<14} {'head':<6} tape_before  →  tape_after")
    print("-" * 65)
    for i, pair in enumerate(data["pairs"][:20]):
        tb = "".join(pair["tape_before"])
        ta = "".join(pair["tape_after"])
        print(f"{i:<5} {pair['state_before']:<14} {pair['head_before']:<6} {tb}  →  {ta}")
    if data["n_samples"] > 20:
        print(f"  ... ({data['n_samples'] - 20} more)")


def cmd_machines(args):
    with get_client(args.url) as client:
        resp = client.get("/api/tm/machines")
        resp.raise_for_status()
        data = resp.json()
    for m in data["machines"]:
        print(f"  • {m}")