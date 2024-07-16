import glob
from pathlib import Path
from typing import Any, Sequence

import gdist
import imageio.v2 as imageio
import numpy as np
import numpy.typing as npt
from fugw.scripts import coarse_to_fine, lmds
from nilearn import plotting, surface


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
        n_jobs=10,
        verbose=True,
    )
    mesh_sample = coarse_to_fine.sample_mesh_uniformly(
        coordinates,
        triangles,
        embeddings=geometry_embedding,
        n_samples=n_samples,
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
