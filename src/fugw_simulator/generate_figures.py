# %%
import glob
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from itertools import product

resolution = "fsaverage5"
estimator = "FUGWBarycenter"
prefix = Path("/data/parietal/store3/work/pbarbara/fugw-simulator")
output_dir = prefix / f"output/{resolution}/{estimator}"

assert output_dir.exists()

# Glob all the files in the output directory recursively
files = glob.glob(str(output_dir / "**" / "*step_9.png"), recursive=True)


def extract_params_from_filename(
    filename: str,
) -> tuple[float, float, float, str]:
    params = Path(filename).parent.name.split("_")
    alpha = float(params[1])
    rho = float(params[3])
    eps = float(params[5])
    return alpha, rho, eps, str(filename)


df = pd.DataFrame(columns=["alpha", "rho", "eps", "path"])
for path in files:
    alpha, rho, eps, path = extract_params_from_filename(path)
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                columns=["alpha", "rho", "eps", "path"],
                data=[[alpha, rho, eps, path]],
            ),
        ],
        ignore_index=True,
    )

df = df.sort_values(by=["alpha", "rho", "eps"])

# %%
alpha = 0.1


def load_image(path: str) -> mpl.image.AxesImage:
    # Crop the image to the central part
    img = plt.imread(path)
    h, w, _ = img.shape
    img = img[
        int(h / 3) : int(3.2 * h / 4), int(w / 2.5) : int(2.45 * w / 4), :
    ]
    return img


def make_legend(
    ax: mpl.axes.Axes,
    title: str,
    labels: list[str],
) -> None:
    # Create a legend
    handles = [
        mpl.lines.Line2D([0], [0], color="black", lw=2, label=label)
        for label in labels
    ]
    ax.legend(
        handles=handles,
        title=title,
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )


def generate_figure(alpha: float, df: pd.DataFrame) -> None:
    # Create a 3x3 gridspec
    scale = 4
    fontsize = 12
    fig = plt.figure(figsize=(3 * scale, 3 * scale))
    grid_spec = gridspec.GridSpec(3, 3, wspace=0.1, hspace=0.1)
    for i, j in product(range(3), range(3)):
        rho = df.rho.unique()[i]
        eps = df.eps.unique()[j]
        img = load_image(
            df[
                (df.alpha == alpha) & (df.rho == rho) & (df.eps == eps)
            ].path.values[0]
        )
        ax = fig.add_subplot(grid_spec[i, j])
        ax.imshow(img)
        ax.axis("off")

        if i == 0:
            ax.title.set_text(f"eps={eps}")
        if j == 0:
            ax.text(
                -0.1,
                0.5,
                f"rho={rho}",
                rotation=90,
                verticalalignment="center",
                horizontalalignment="center",
                transform=ax.transAxes,
                fontsize=fontsize,
            )

    # Add colorbar
    ax = fig.add_subplot(grid_spec[1, :])
    ax.axis("off")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%")
    fig.add_axes(cax)
    fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=-1, vmax=1), cmap="coolwarm"
        ),
        cax=cax,
    )
    fig.suptitle(f"alpha={alpha}", x=0.51, y=0.90)
    fig.savefig(output_dir / f"alpha_{alpha}.png", dpi=300)


for alpha in df.alpha.unique():
    generate_figure(alpha, df)
