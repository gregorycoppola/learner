"""Learner CLI."""
import argparse
import importlib
import pkgutil
from learner.cli import commands


def main():
    parser = argparse.ArgumentParser(prog="learner", description="Learner CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for _, module_name, is_pkg in pkgutil.iter_modules(commands.__path__):
        if is_pkg or module_name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"learner.cli.commands.{module_name}")
        except Exception as e:
            print(f"❌ Failed to import command module '{module_name}': {e}")
            continue
        if hasattr(module, "add_subparser"):
            try:
                module.add_subparser(subparsers)
            except Exception as e:
                print(f"⚠️  Failed to register subparser for '{module_name}': {e}")

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()