from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy.typing as npt
import torch
from fugw.utils import _make_tensor

from fugw_simulator.utils import surf_plot_wrapper


def callback_coarse_mapping(
    locals: dict[str, Any],
    source_features: npt.NDArray[Any],
    target_features: npt.NDArray[Any],
    fsaverage: Any,
    mesh: str,
    contrast_idx: int = 0,
    device: torch.device = torch.device("cpu"),
    output_dir: Path = Path("output"),
) -> None:
    # Get current transport plan and tensorize features
    pi = _make_tensor(locals["pi"], device)
    source_features_tensor = _make_tensor(source_features, device)
    target_features_tensor = _make_tensor(target_features, device)
    transformed_features = (
        pi.T @ source_features_tensor.T / pi.sum(dim=0).reshape(-1, 1)
    ).T

    fig = plt.figure(figsize=(3 * 3, 3))
    fig.suptitle(
        f"BCD step {locals['idx']}, alpha={locals['alpha']},"
        f" rho={locals['rho_s']}, eps={locals['eps']}"
    )
    grid_spec = gridspec.GridSpec(1, 3, figure=fig)

    ax = fig.add_subplot(grid_spec[0, 0], projection="3d")
    surf_plot_wrapper(
        fsaverage,
        source_features_tensor.cpu().numpy(),
        mesh=mesh,
        title="Source",
        axes=ax,
        colorbar=False,
    )

    ax = fig.add_subplot(grid_spec[0, 1], projection="3d")
    surf_plot_wrapper(
        fsaverage,
        transformed_features.cpu().numpy(),
        mesh=mesh,
        title="Projected source",
        axes=ax,
        colorbar=False,
    )

    ax = fig.add_subplot(grid_spec[0, 2], projection="3d")
    surf_plot_wrapper(
        fsaverage,
        target_features_tensor.cpu().numpy()[contrast_idx, :],
        mesh=mesh,
        title="Target",
        axes=ax,
        colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"bcd_step_{locals['idx']}.png")
    plt.close(fig)


def callback_fine_mapping(
    locals: dict[str, Any],
    source_features: npt.NDArray[Any],
    target_features: npt.NDArray[Any],
    fsaverage: Any,
    mesh: str,
    contrast_idx: int = 0,
    device: torch.device = torch.device("cpu"),
    output_dir: Path = Path("output"),
) -> None:
    # Get current transport plan and tensorize features
    pi = _make_tensor(locals["pi"], device).to_sparse_coo()
    source_features_tensor = _make_tensor(source_features, device)
    target_features_tensor = _make_tensor(target_features, device)
    transformed_features = torch.sparse.mm(
        pi.transpose(0, 1),
        source_features_tensor.T,
    ).to_dense() / (
        torch.sparse.sum(pi, dim=0).to_dense().reshape(-1, 1)
        # Add very small value to handle null rows
        + 1e-16
    )

    fig = plt.figure(figsize=(3 * 3, 3))
    fig.suptitle(
        f"BCD step {locals['idx']}, alpha={locals['alpha']},"
        f" rho={locals['rho_s']}, eps={locals['eps']}"
    )
    grid_spec = gridspec.GridSpec(1, 3, figure=fig)

    ax = fig.add_subplot(grid_spec[0, 0], projection="3d")
    surf_plot_wrapper(
        fsaverage,
        source_features_tensor.cpu().numpy(),
        mesh=mesh,
        title="Source",
        axes=ax,
        colorbar=False,
    )

    ax = fig.add_subplot(grid_spec[0, 1], projection="3d")
    surf_plot_wrapper(
        fsaverage,
        transformed_features.cpu().numpy(),
        mesh=mesh,
        title="Projected source",
        axes=ax,
        colorbar=False,
    )

    ax = fig.add_subplot(grid_spec[0, 2], projection="3d")
    surf_plot_wrapper(
        fsaverage,
        target_features_tensor.cpu().numpy()[contrast_idx, :],
        mesh=mesh,
        title="Target",
        axes=ax,
        colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"bcd_step_{locals['idx']}.png")
    plt.close(fig)


def callback_barycenter(
    locals: dict[str, Any],
    features_list: list[npt.NDArray[Any]],
    fsaverage: Any,
    mesh: str,
    contrast_idx: int = 0,
    device: torch.device = torch.device("cpu"),
    output_dir: Path = Path("output"),
) -> None:
    fig = plt.figure(figsize=(3 * 3, 3))
    fig.suptitle(f"Barycenter: iter {locals['idx']}")
    grid_spec = gridspec.GridSpec(1, 3, figure=fig)

    ax = fig.add_subplot(grid_spec[0, 0], projection="3d")
    surf_plot_wrapper(
        fsaverage,
        features_list[0][contrast_idx, :],
        mesh=mesh,
        title="1st source",
        axes=ax,
        colorbar=False,
    )

    ax = fig.add_subplot(grid_spec[0, 1], projection="3d")
    surf_plot_wrapper(
        fsaverage,
        locals["barycenter_features"].cpu().numpy()[contrast_idx, :],
        mesh=mesh,
        title="Barycenter",
        axes=ax,
        colorbar=False,
    )

    ax = fig.add_subplot(grid_spec[0, 2], projection="3d")
    surf_plot_wrapper(
        fsaverage,
        features_list[1][contrast_idx, :],
        mesh=mesh,
        title="2nd source",
        axes=ax,
        colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"barycenter_step_{locals['idx']}.png")
    plt.close(fig)
