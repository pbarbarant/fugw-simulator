# %%
import glob
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import gdist
import imageio.v2 as imageio
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from fugw.mappings import FUGW, FUGWBarycenter, FUGWSparseBarycenter
from fugw.scripts import coarse_to_fine, lmds
from fugw.utils import _make_tensor
from nilearn import datasets, plotting, surface


def sample_geometry(
    mesh_path: str,
    n_samples: int = 1000,
) -> Any:
    """Sample the geometry of the mask"""
    (coordinates, triangles) = surface.load_surf_mesh(mesh_path)
    geometry_embedding = lmds.compute_lmds_mesh(
        coordinates,
        triangles,
        n_landmarks=100,
        k=3,
        n_jobs=2,
        verbose=True,
    )
    mesh_sample = coarse_to_fine.sample_mesh_uniformly(
        coordinates,
        triangles,
        embeddings=geometry_embedding,
        n_samples=1000,
    )
    (
        geometry_embedding_normalized,
        _,
    ) = coarse_to_fine.random_normalizing(geometry_embedding)

    return geometry_embedding_normalized, mesh_sample


def generate_gaussian_single(
    vertices: npt.NDArray[Any], center_vertex_idx: int, sigma: float
) -> Any:
    center_vertex: int = vertices[center_vertex_idx]
    distances: npt.NDArray[Any] = np.linalg.norm(
        vertices - center_vertex, axis=1
    )
    return np.exp(-((distances) ** 2) / (2 * sigma**2))


def generate_gaussians(
    vertices: npt.NDArray[Any],
    centers_list: list[int],
    sigma: float,
) -> Any:
    gaussian_map = np.zeros_like(vertices[:, 0])
    for center_vertex_idx in centers_list:
        gaussian_map += generate_gaussian_single(
            vertices, center_vertex_idx, sigma
        )
    return gaussian_map


def normalize_map(vertices: npt.NDArray[Any]) -> Any:
    """Normalize the map between -1 and 1."""
    return (
        2 * (vertices - vertices.min()) / (vertices.max() - vertices.min()) - 1
    )


def generate_simulated_data(
    vertices: npt.NDArray[Any],
    centers_list: list[int],
    sigma: float,
    noise_level: float = 0.1,
) -> Any:
    gaussian_map = generate_gaussians(vertices, centers_list, sigma)
    noisy_map = add_noise(gaussian_map, noise_level=noise_level)
    normalized_map = normalize_map(noisy_map)
    return normalized_map.T


def surf_plot_wrapper(
    fsaverage: Any,
    surface_map: npt.NDArray[Any],
    mesh: str = "infl_left",
    colorbar: bool = True,
    **kwargs: Any,
) -> None:
    plotting.plot_surf_stat_map(
        fsaverage[mesh],
        surface_map,
        cmap="coolwarm",
        colorbar=colorbar,
        bg_map=fsaverage["sulc_left"],
        bg_on_data=True,
        darkness=0.5,
        **kwargs,
    )


def add_noise(vertices: npt.NDArray[Any], noise_level: float = 0.1) -> Any:
    return vertices + np.random.normal(0, noise_level, vertices.shape)


def compute_geometry_from_mesh(mesh_path: str) -> Any:
    """Util function to compute matrix of geodesic distances of a mesh."""
    (coordinates, triangles) = surface.load_surf_mesh(mesh_path)
    geometry = gdist.local_gdist_matrix(
        coordinates.astype(np.float64), triangles.astype(np.int32)
    ).toarray()

    return geometry


def callback_one_mapping(
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


def generate_gif(output_dir: Path, duration: float = 0.1) -> None:
    # Glob all the images
    image_paths = glob.glob(str(output_dir / "*.png"))
    # Sort the images
    image_paths.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    # Read the images
    image_arrays: Sequence[Any] = [
        imageio.imread(image_path) for image_path in image_paths
    ]
    imageio.mimsave(
        output_dir / "animation.gif", image_arrays, duration=duration
    )


def fugw_simple_mapping(
    output_dir: Path,
    source_features: npt.NDArray[Any],
    target_features: npt.NDArray[Any],
    fsaverage: Any,
    mesh: str = "infl_left",
    alpha: float = 0.5,
    rho: float = 1,
    eps: float = 1e-1,
    nits_bcd: int = 100,
    nits_uot: int = 1,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
    output_gif: bool = True,
) -> FUGW:
    mapping = FUGW(
        alpha=alpha,
        rho=rho,
        eps=eps,
    )

    geometry = compute_geometry_from_mesh(fsaverage[mesh])
    # Normalize geometry
    geometry = geometry / geometry.max()

    mapping.fit(
        source_features,
        target_features,
        source_geometry=geometry / geometry.max(),
        target_geometry=geometry / geometry.max(),
        verbose=verbose,
        solver_params={"nits_bcd": nits_bcd, "nits_uot": nits_uot},
        callback_bcd=partial(
            callback_one_mapping,
            fsaverage=fsaverage,
            mesh=mesh,
            source_features=source_features,
            target_features=target_features,
            device=device,
            output_dir=output_dir,
        ),
    )

    if output_gif:
        generate_gif(output_dir, duration=1)

    return mapping


def fugw_coarse_barycenter(
    output_dir: Path,
    features_list: list[npt.NDArray[Any]],
    weights_list: list[npt.NDArray[Any]],
    geometry_list: list[npt.NDArray[Any]],
    fsaverage: Any,
    mesh: str = "infl_left",
    alpha: float = 0.5,
    rho: float = 1,
    eps: float = 1e-4,
    nits_barycenter: int = 10,
    nits_bcd: int = 100,
    nits_uot: int = 1,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
    output_gif: bool = True,
) -> FUGW:
    fugw_barycenter = FUGWBarycenter(
        alpha=alpha,
        rho=rho,
        eps=eps,
    )
    barycenter = fugw_barycenter.fit(
        weights_list,
        features_list,
        geometry_list,
        nits_barycenter=nits_barycenter,
        device=device,
        verbose=verbose,
        init_barycenter_features=features_list[0],
        solver_params={"nits_bcd": nits_bcd, "nits_uot": nits_uot},
        callback_barycenter=partial(
            callback_barycenter,
            features_list=features_list,
            fsaverage=fsaverage,
            mesh=mesh,
            device=device,
            output_dir=output_dir,
        ),
    )

    if output_gif:
        generate_gif(output_dir, duration=1)

    return barycenter


def fugw_sparse_barycenter(
    output_dir: Path,
    features_list: list[npt.NDArray[Any]],
    weights_list: list[npt.NDArray[Any]],
    geometry_embedding: torch.Tensor,
    mesh_sample: npt.NDArray[Any],
    fsaverage: Any,
    mesh: str = "infl_left",
    alpha_coarse: float = 0.5,
    alpha_fine: float = 0.5,
    rho_coarse: float = 1,
    rho_fine: float = 1,
    eps_coarse: float = 1,
    eps_fine: float = 1,
    selection_radius: float = 5,
    nits_barycenter: int = 10,
    nits_bcd: int = 5,
    nits_uot: int = 100,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
    output_gif: bool = True,
) -> FUGW:
    sparse_barycenter = FUGWSparseBarycenter(
        alpha_coarse=alpha_coarse,
        alpha_fine=alpha_fine,
        rho_coarse=rho_coarse,
        rho_fine=rho_fine,
        eps_coarse=eps_coarse,
        eps_fine=eps_fine,
        selection_radius=selection_radius,
    )
    barycenter = sparse_barycenter.fit(
        weights_list,
        features_list,
        geometry_embedding,
        mesh_sample=mesh_sample,
        nits_barycenter=nits_barycenter,
        init_barycenter_features=features_list[0],
        coarse_mapping_solver_params={
            "nits_bcd": nits_bcd,
            "nits_uot": nits_uot,
        },
        fine_mapping_solver_params={
            "nits_bcd": nits_bcd,
            "nits_uot": nits_uot,
        },
        callback_barycenter=partial(
            callback_barycenter,
            features_list=features_list,
            fsaverage=fsaverage,
            mesh=mesh,
            device=device,
            output_dir=output_dir,
        ),
        device=device,
        verbose=verbose,
    )

    if output_gif:
        generate_gif(output_dir, duration=1)

    return barycenter


def main() -> None:
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    mesh = "infl_left"
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    vertices, _ = surface.load_surf_mesh(fsaverage[mesh])

    # Fsaverage3
    # simulated_source = generate_simulated_data(
    #     vertices, [300, 270], sigma=16, noise_level=0.2
    # )
    # simulated_target = generate_simulated_data(
    #     vertices, [302, 101, 272], sigma=16, noise_level=0.2
    # )

    # Fsaverage4
    # simulated_source = generate_simulated_data(
    #     vertices, [1000, 2000], sigma=16, noise_level=0.2
    # )
    # simulated_target = generate_simulated_data(
    #     vertices, [1010, 2010], sigma=16, noise_level=0.2
    # )

    # Fsaverage5
    simulated_source = generate_simulated_data(
        vertices, [4000, 5000], sigma=16, noise_level=0.2
    )
    simulated_target = generate_simulated_data(
        vertices, [4010, 5010], sigma=16, noise_level=0.2
    )

    n_vertices = vertices.shape[0]
    simulated_source = simulated_source.reshape(1, -1)
    simulated_target = simulated_target.reshape(1, -1)

    features_list = [simulated_source, simulated_target]
    weights_list = [
        np.ones(n_vertices) / n_vertices,
        np.ones(n_vertices) / n_vertices,
    ]

    alpha = 0.5
    rho = float("inf")
    eps = 1e-4

    # print("Computing geometry...")
    # geometry = compute_geometry_from_mesh(fsaverage[mesh])
    # Normalize geometry
    # geometry = geometry / geometry.max()
    # geometry_list = [geometry, geometry]

    # output_dir_one_mapping = Path(
    #     f"output/one_mapping_alpha_{alpha}_rho_{rho}_eps_{eps}"
    # )
    # output_dir_one_mapping.mkdir(exist_ok=True, parents=True)
    # _ = fugw_simple_mapping(
    #     output_dir_one_mapping,
    #     simulated_source,
    #     simulated_target,
    #     fsaverage,
    #     mesh=mesh,
    #     alpha=alpha,
    #     rho=rho,
    #     eps=eps,
    #     nits_bcd=100,
    #     nits_uot=1,
    #     device=device,
    #     verbose=True,
    #     output_gif=True,
    # )

    # output_dir_barycenter = Path(
    #     f"output/barycenter_alpha_{alpha}_rho_{rho}_eps_{eps}"
    # )
    # output_dir_barycenter.mkdir(exist_ok=True, parents=True)
    # _ = fugw_coarse_barycenter(
    #     output_dir_barycenter,
    #     features_list,
    #     weights_list,
    #     geometry_list,
    #     fsaverage,
    #     mesh=mesh,
    #     alpha=0.5,
    #     rho=1e-1,
    #     eps=1e-4,
    #     nits_barycenter=30,
    #     nits_bcd=5,
    #     nits_uot=100,
    #     device=device,
    #     verbose=False,
    #     output_gif=True,
    # )

    geometry_embedding, mesh_sample = sample_geometry(
        fsaverage[mesh], n_samples=1000
    )

    output_dir_barycenter = Path(
        f"output/bary_sparse_alpha_{alpha}_rho_{rho}_eps_{eps}"
    )
    output_dir_barycenter.mkdir(exist_ok=True, parents=True)
    _ = fugw_sparse_barycenter(
        output_dir_barycenter,
        features_list,
        weights_list,
        geometry_embedding,
        mesh_sample,
        fsaverage,
        mesh=mesh,
        alpha_coarse=alpha,
        alpha_fine=alpha,
        rho_coarse=rho,
        rho_fine=rho,
        eps_coarse=eps,
        eps_fine=eps,
        selection_radius=5,
        nits_barycenter=30,
        nits_bcd=5,
        nits_uot=100,
        device=device,
        verbose=False,
        output_gif=True,
    )


if __name__ == "__main__":
    main()
