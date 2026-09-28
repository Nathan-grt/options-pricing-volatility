"""Style graphique commun du projet (ajout) + sauvegarde des figures."""
from pathlib import Path

import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
RES_DIR = Path(__file__).resolve().parents[1] / "results"

PALETTE = {
    "navy": "#1f3b73", "crimson": "#c0392b", "teal": "#16a085", "orange": "#e67e22",
    "purple": "#8e44ad", "grey": "#7f8c8d", "gold": "#d4a017", "sky": "#3498db",
}
CYCLE = [PALETTE[k] for k in ("navy", "crimson", "teal", "orange", "purple", "sky", "gold", "grey")]


def set_style():
    plt.rcParams.update({
        "figure.figsize": (10, 5.5), "figure.dpi": 110, "savefig.dpi": 150,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
        "legend.frameon": False, "font.size": 10,
        "axes.prop_cycle": plt.cycler(color=CYCLE),
    })


def savefig(fig, name):
    """Sauvegarde dans figures/ (PNG) et renvoie le chemin."""
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    return path


def save_table(df, name, index=False, floatfmt=".4f"):
    """Sauvegarde un DataFrame en CSV + Markdown dans results/."""
    RES_DIR.mkdir(exist_ok=True)
    df.to_csv(RES_DIR / f"{name}.csv", index=index)
    try:
        (RES_DIR / f"{name}.md").write_text(df.to_markdown(index=index, floatfmt=floatfmt))
    except ImportError:  # tabulate absent
        (RES_DIR / f"{name}.md").write_text(df.to_string(index=index))
