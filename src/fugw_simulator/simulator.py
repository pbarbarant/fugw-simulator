# %%
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from fugw.mappings import (
    FUGW,
    FUGWBarycenter,
    FUGWSparse,
    FUGWSparseBarycenter,
)
from fugw.scripts import coarse_to_fine
from nilearn import datasets, surface

from fugw_simulator.callbacks import (
    callback_barycenter,
    callback_coarse_mapping,
    callback_fine_mapping,
)
from fugw_simulator.utils import (
    compute_geometry_from_mesh,
    generate_gif,
    generate_simulated_data,
    sample_geometry,
)


def fugw_dense_mapping(
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
            callback_coarse_mapping,
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


def fugw_sparse_mapping(
    output_dir: Path,
    source_features: npt.NDArray[Any],
    target_features: npt.NDArray[Any],
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
    nits_bcd: int = 100,
    nits_uot: int = 1,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
    output_gif: bool = True,
) -> FUGWSparse:
    coarse_mapping = FUGW(
        alpha=alpha_coarse,
        rho=rho_coarse,
        eps=eps_coarse,
    )

    fine_mapping = FUGWSparse(
        alpha=alpha_fine,
        rho=rho_fine,
        eps=eps_fine,
    )

    source_sample, target_sample, mask = coarse_to_fine.fit(
        source_features=source_features,
        target_features=target_features,
        source_geometry_embeddings=geometry_embedding,
        target_geometry_embeddings=geometry_embedding,
        source_sample=mesh_sample,
        target_sample=mesh_sample,
        coarse_mapping=coarse_mapping,
        coarse_mapping_solver="mm",
        coarse_mapping_solver_params={
            "nits_bcd": nits_bcd,
            "nits_uot": nits_uot,
        },
        coarse_pairs_selection_method="topk",
        source_selection_radius=selection_radius,
        target_selection_radius=selection_radius,
        fine_mapping=fine_mapping,
        fine_mapping_solver="mm",
        fine_mapping_solver_params={
            "nits_bcd": nits_bcd,
            "nits_uot": nits_uot,
        },
        fine_callback_bcd=partial(
            callback_fine_mapping,
            fsaverage=fsaverage,
            mesh=mesh,
            source_features=source_features,
            target_features=target_features,
            device=device,
            output_dir=output_dir,
        ),
        verbose=verbose,
    )

    if output_gif:
        generate_gif(output_dir, duration=1)

    return source_sample, target_sample, mask


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
) -> FUGWBarycenter:
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
) -> FUGWSparseBarycenter:
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
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage3")
    vertices, _ = surface.load_surf_mesh(fsaverage[mesh])

    # Fsaverage3
    simulated_source = generate_simulated_data(
        vertices, [300, 270], sigma=16, noise_level=0.2
    )
    simulated_target = generate_simulated_data(
        vertices, [302, 101, 272], sigma=16, noise_level=0.2
    )

    # Fsaverage4
    # simulated_source = generate_simulated_data(
    #     vertices, [1000, 2000], sigma=16, noise_level=0.2
    # )
    # simulated_target = generate_simulated_data(
    #     vertices, [1010, 2010], sigma=16, noise_level=0.2
    # )

    # Fsaverage5
    # simulated_source = generate_simulated_data(
    #     vertices, [4000, 5000], sigma=16, noise_level=0.2
    # )
    # simulated_target = generate_simulated_data(
    #     vertices, [4010, 5010, 10000], sigma=16, noise_level=0.2
    # )

    simulated_source = simulated_source.reshape(1, -1)
    simulated_target = simulated_target.reshape(1, -1)

    features_list = [simulated_source, simulated_target]
    n_vertices = vertices.shape[0]
    weights_list = [
        np.ones(n_vertices) / n_vertices,
        np.ones(n_vertices) / n_vertices,
    ]

    alpha = 0.5
    rho = 1.0
    eps = 1e-4

    print("Computing geometry...")
    geometry = compute_geometry_from_mesh(fsaverage[mesh])
    # Normalize geometry
    geometry = geometry / geometry.max()
    geometry_list = [geometry, geometry]

    output_dir = Path(f"output/FUGW/alpha_{alpha}_rho_{rho}_eps_{eps}")
    output_dir.mkdir(exist_ok=True, parents=True)
    _ = fugw_dense_mapping(
        output_dir,
        simulated_source,
        simulated_target,
        fsaverage,
        mesh=mesh,
        alpha=alpha,
        rho=rho,
        eps=eps,
        nits_bcd=100,
        nits_uot=1,
        device=device,
        verbose=True,
        output_gif=True,
    )

    output_dir = Path(
        f"output/FUGWBarycenter/alpha_{alpha}_rho_{rho}_eps_{eps}"
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    _ = fugw_coarse_barycenter(
        output_dir,
        features_list,
        weights_list,
        geometry_list,
        fsaverage,
        mesh=mesh,
        alpha=0.5,
        rho=1e-1,
        eps=1e-4,
        nits_barycenter=30,
        nits_bcd=5,
        nits_uot=100,
        device=device,
        verbose=False,
        output_gif=True,
    )

    geometry_embedding, mesh_sample = sample_geometry(
        fsaverage[mesh], n_samples=100
    )

    output_dir = Path(f"output/FUGWSparse/alpha_{alpha}_rho_{rho}_eps_{eps}")
    output_dir.mkdir(exist_ok=True, parents=True)
    _ = fugw_sparse_mapping(
        output_dir,
        simulated_source,
        simulated_target,
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
        selection_radius=2,
        nits_bcd=5,
        nits_uot=100,
        device=device,
        verbose=True,
        output_gif=True,
    )

    output_dir = Path(
        f"output/FUGWSparseBarycenter/alpha_{alpha}_rho_{rho}_eps_{eps}"
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    _ = fugw_sparse_barycenter(
        output_dir,
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
        selection_radius=6,
        nits_barycenter=30,
        nits_bcd=5,
        nits_uot=100,
        device=device,
        verbose=False,
        output_gif=True,
    )


if __name__ == "__main__":
    main()
