"""
Model checkpointing — save and load model weights with architecture metadata.

Format:
  {
    "model_class": "categorical" | "regression",
    "arch": { d_input, n_tape, n_states, d_model, n_layers, n_heads },
    "state_index": { state_name: int },
    "machine": str,
    "epoch": int,
    "val_acc": float,
    "phase": "sft" | "grpo",
    "state_dict": { ... }
  }
"""
import torch
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"


def models_dir() -> Path:
    MODELS_DIR.mkdir(exist_ok=True)
    return MODELS_DIR


def save(
    model,
    model_class: str,
    arch: dict,
    state_index: dict,
    machine: str,
    epoch: int,
    val_acc: float,
    phase: str = "sft",
    name: str = None,
) -> Path:
    """Save model checkpoint. Returns path to saved file."""
    if name is None:
        name = f"{machine}_{phase}_e{epoch}_acc{val_acc:.3f}"
    path = models_dir() / f"{name}.pt"
    torch.save({
        "model_class": model_class,
        "arch":        arch,
        "state_index": state_index,
        "machine":     machine,
        "epoch":       epoch,
        "val_acc":     val_acc,
        "phase":       phase,
        "state_dict":  model.state_dict(),
    }, path)
    return path


def load(name: str):
    """
    Load a checkpoint by name (without .pt extension).
    Returns (model, metadata_dict).
    """
    path = models_dir() / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint: {path}")

    data = torch.load(path, map_location="cpu")
    arch = data["arch"]

    if data["model_class"] == "categorical":
        from learner.core.model import TMTransformerCategorical
        model = TMTransformerCategorical(
            d_input=arch["d_input"],
            n_tape=arch["n_tape"],
            n_states=arch["n_states"],
            d_model=arch["d_model"],
            n_heads=arch["n_heads"],
            n_layers=arch["n_layers"],
        )
    else:
        from learner.core.model import TMTransformer
        model = TMTransformer(
            d_input=arch["d_input"],
            d_model=arch["d_model"],
            n_heads=arch["n_heads"],
            n_layers=arch["n_layers"],
        )

    model.load_state_dict(data["state_dict"])
    model.eval()

    return model, data


def list_checkpoints() -> list[dict]:
    """List all saved checkpoints with metadata."""
    out = []
    for path in sorted(models_dir().glob("*.pt")):
        try:
            data = torch.load(path, map_location="cpu")
            out.append({
                "name":        path.stem,
                "machine":     data.get("machine", "?"),
                "phase":       data.get("phase", "?"),
                "model_class": data.get("model_class", "?"),
                "epoch":       data.get("epoch", 0),
                "val_acc":     data.get("val_acc", 0.0),
                "arch":        data.get("arch", {}),
            })
        except Exception:
            pass
    return out