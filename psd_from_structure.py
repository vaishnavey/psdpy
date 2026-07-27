"""Compute a pore size distribution from a structure file using voxelization.

This is a practical Python implementation of the pseudocode in
psd_from_structure_pseudocode.py. It supports PDB and GRO inputs through
MDAnalysis, rasterizes atoms onto a 3D grid, computes the Euclidean distance
transform, and returns a pore-size histogram.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import MDAnalysis as mda
import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay

from plotting_psd import ensure_dir, plot_dist_slice, plot_psd_cdf, plot_psd_pdf, plot_timeseries


DEFAULT_RADII_TABLE = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "NA": 2.27,
    "MG": 1.73,
    "AL": 1.84,
    "SI": 2.10,
    "K": 2.75,
    "CA": 2.31,
    "MN": 1.97,
    "FE": 1.94,
    "CO": 1.92,
    "NI": 1.63,
    "CU": 1.40,
    "ZN": 1.39,
    "SE": 1.90,
}


@dataclass
class Grid:
    origin: np.ndarray
    dx: float
    shape: tuple[int, int, int]
    box: np.ndarray | None = None
    use_pbc: bool = False


@dataclass
class PSDParams:
    dx: float = 0.25
    probe_radius: float = 0.0
    use_pbc: bool = True
    padding: float = 2.0
    mode: str = "voxel"
    r_min: float = 0.1
    bins: np.ndarray = field(default_factory=lambda: np.linspace(0.0, 10.0, 201))
    selection: str = "all"
    keep_only_connected_pore: bool = False
    connectivity_criterion: str = "largest"
    peak_min_distance: int = 2
    frame_slice: tuple[int | None, int | None, int] = (0, None, 1)
    default_radius: float = 1.7
    radii_table: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RADII_TABLE))
    output_dir: str = "./psd_output"
    align_structure: bool = True
    align_axis: str = "z"


def run_psd_analysis(input_file: str, traj_file: str | None = None, params: PSDParams | None = None):
    params = params or PSDParams()
    universe = load_structure(input_file, traj_file)
    frames = get_frames(universe, params.frame_slice)

    all_hist_counts = np.zeros(len(params.bins) - 1, dtype=float)
    frame_metrics = []
    reference_dist = None

    for frame_index in frames:
        universe.trajectory[frame_index]
        coords, elements, box = extract_frame_data(universe, params.selection)
        if params.align_structure:
            coords = align_coords_to_axis(coords, target_axis=params.align_axis)
        radii = assign_vdw_radii(elements, params.radii_table, params.default_radius)
        r_eff = radii + params.probe_radius

        grid, effective_use_pbc = build_grid(coords, box, params.dx, params.use_pbc, params.padding, r_eff)
        solid = rasterize_atoms_to_solid(coords, r_eff, grid, effective_use_pbc)
        if not effective_use_pbc:
            solid, grid = crop_grid_to_occupied_region(solid, grid, padding=params.padding)
            analysis_mask = convex_hull_mask(coords, grid)
            solid = np.logical_or(solid, np.logical_not(analysis_mask))
        pore = np.logical_not(solid)

        if params.keep_only_connected_pore:
            pore = filter_connected_pore_regions(pore, criterion=params.connectivity_criterion)

        dist_vox = ndimage.distance_transform_edt(pore)
        dist_A = dist_vox * params.dx
        if reference_dist is None:
            reference_dist = dist_A

        if params.mode == "voxel":
            radii_samples = dist_A[pore]
        elif params.mode == "maximal_sphere":
            peaks_mask = find_local_maxima(dist_A, mask=pore, min_distance=params.peak_min_distance)
            radii_samples = dist_A[peaks_mask]
        else:
            raise ValueError(f"Unknown PSD mode: {params.mode}")

        radii_samples = radii_samples[radii_samples > params.r_min]
        hist_counts, _ = np.histogram(radii_samples, bins=params.bins, density=False)
        all_hist_counts += hist_counts

        porosity = float(np.count_nonzero(pore) / pore.size)
        frame_metrics.append(
            {
                "frame": int(frame_index),
                "porosity": porosity,
                "mean_radius": float(np.mean(radii_samples)) if radii_samples.size else float("nan"),
                "max_radius": float(np.max(radii_samples)) if radii_samples.size else float("nan"),
            }
        )

    bin_widths = np.diff(params.bins)
    total = float(np.sum(all_hist_counts))
    if total > 0.0:
        pdf = all_hist_counts / (total * bin_widths)
        cdf = np.cumsum(all_hist_counts) / total
    else:
        pdf = np.zeros_like(all_hist_counts)
        cdf = np.zeros_like(all_hist_counts)
    bin_centers = 0.5 * (params.bins[:-1] + params.bins[1:])

    outputs = {
        "bin_centers": bin_centers,
        "pdf": pdf,
        "cdf": cdf,
        "hist_counts": all_hist_counts,
        "frame_metrics": frame_metrics,
        "reference_dist": reference_dist,
        "params": dataclass_to_serializable(params),
    }

    save_outputs(outputs, params.output_dir)
    return outputs


def load_structure(input_file: str, traj_file: str | None = None):
    if traj_file is None:
        return mda.Universe(input_file)
    return mda.Universe(input_file, traj_file)


def get_frames(universe, frame_slice: tuple[int | None, int | None, int]):
    start, stop, step = frame_slice
    frame_indices = []
    for ts in universe.trajectory[start:stop:step]:
        frame_indices.append(int(ts.frame))
    return frame_indices


def extract_frame_data(universe, selection: str):
    atoms = universe.select_atoms(selection)
    coords = np.asarray(atoms.positions, dtype=float)
    elements = get_atom_elements(atoms)
    box = np.asarray(universe.dimensions, dtype=float) if universe.dimensions is not None else None
    return coords, elements, box


def align_coords_to_axis(coords, target_axis: str = "z"):
    centered = np.asarray(coords, dtype=float)
    centered = centered - np.mean(centered, axis=0, keepdims=True)
    if centered.shape[0] < 3:
        return centered

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    principal_axes = vh.T
    axis_map = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}
    target = axis_map.get(target_axis.lower())
    if target is None:
        raise ValueError(f"Unknown align_axis: {target_axis}")

    source = principal_axes[:, 0]
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)

    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if np.isclose(dot, 1.0):
        return centered
    if np.isclose(dot, -1.0):
        # 180-degree flip around any axis perpendicular to the source vector.
        helper = np.array([1.0, 0.0, 0.0]) if abs(source[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        rotation_axis = np.cross(source, helper)
        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation = rotation_matrix_from_axis_angle(rotation_axis, np.pi)
    else:
        rotation_axis = np.cross(source, target)
        rotation_axis /= np.linalg.norm(rotation_axis)
        angle = np.arccos(dot)
        rotation = rotation_matrix_from_axis_angle(rotation_axis, angle)

    aligned = centered @ rotation.T
    return aligned


def rotation_matrix_from_axis_angle(axis, angle):
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )


def get_atom_elements(atoms) -> list[str]:
    if hasattr(atoms, "elements"):
        elements = np.asarray(atoms.elements)
        if elements.size and not np.all(elements == ""):
            return [str(item).strip().upper() for item in elements]
    names = np.asarray(atoms.names)
    return [infer_element_from_name(name) for name in names]


def infer_element_from_name(name: str) -> str:
    cleaned = "".join(ch for ch in str(name).strip() if ch.isalpha())
    if not cleaned:
        return "C"
    if len(cleaned) >= 2 and cleaned[:2].upper() in DEFAULT_RADII_TABLE:
        return cleaned[:2].upper()
    return cleaned[0].upper()


def assign_vdw_radii(elements: Iterable[str], radii_table: dict[str, float], default_radius: float):
    return np.array([radii_table.get(str(element).strip().upper(), default_radius) for element in elements], dtype=float)


def build_grid(coords, box, dx, use_pbc, padding, r_eff):
    effective_use_pbc = bool(use_pbc and box_is_orthorhombic(box))
    if use_pbc and not effective_use_pbc:
        print(
            "[psdpy] Non-orthorhombic or invalid box detected; using bounding-box grid instead of simplified PBC.",
            file=sys.stderr,
        )

    if effective_use_pbc and box is not None:
        lengths = np.asarray(box[:3], dtype=float)
        origin = np.zeros(3, dtype=float)
        shape = tuple(int(math.ceil(length / dx)) for length in lengths)
    else:
        rmax = float(np.max(r_eff)) if len(r_eff) else 0.0
        minc = np.min(coords, axis=0) - (padding + rmax)
        maxc = np.max(coords, axis=0) + (padding + rmax)
        origin = minc.astype(float)
        lengths = maxc - minc
        shape = tuple(max(1, int(math.ceil(length / dx))) for length in lengths)

    return Grid(origin=origin, dx=float(dx), shape=shape, box=box, use_pbc=effective_use_pbc), effective_use_pbc


def box_is_orthorhombic(box) -> bool:
    if box is None:
        return False
    box = np.asarray(box, dtype=float)
    if box.size < 6:
        return False
    alpha, beta, gamma = box[3:6]
    return all(abs(angle - 90.0) < 1e-3 for angle in (alpha, beta, gamma))


def rasterize_atoms_to_solid(coords, r_eff, grid: Grid, use_pbc: bool):
    solid = np.zeros(grid.shape, dtype=bool)
    for idx, center in enumerate(coords):
        radius = float(r_eff[idx])
        idx_min, idx_max = sphere_index_bounds(center, radius, grid)
        for ix, iy, iz in iterate_indices(idx_min, idx_max, grid.shape, use_pbc):
            voxel_center = voxel_center_world(ix, iy, iz, grid)
            dvec = voxel_center - center
            if use_pbc:
                dvec = minimum_image(dvec, grid.box)
            if np.dot(dvec, dvec) <= radius * radius:
                solid[ix, iy, iz] = True
    return solid


def crop_grid_to_occupied_region(solid, grid: Grid, padding: float):
    occupied = np.argwhere(solid)
    if occupied.size == 0:
        return solid, grid

    margin = max(1, int(math.ceil(padding / grid.dx)))
    lower = np.maximum(np.min(occupied, axis=0) - margin, 0)
    upper = np.minimum(np.max(occupied, axis=0) + margin + 1, np.array(solid.shape))

    cropped = solid[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]]
    new_origin = grid.origin + grid.dx * lower.astype(float)
    new_grid = Grid(origin=new_origin, dx=grid.dx, shape=cropped.shape, box=grid.box, use_pbc=grid.use_pbc)
    return cropped, new_grid


def convex_hull_mask(coords, grid: Grid):
    if coords.shape[0] < 4:
        return np.ones(grid.shape, dtype=bool)

    try:
        hull = Delaunay(coords)
    except Exception:
        return np.ones(grid.shape, dtype=bool)

    axes = [grid.origin[i] + grid.dx * (np.arange(grid.shape[i], dtype=float) + 0.5) for i in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    mask = np.zeros(points.shape[0], dtype=bool)
    chunk_size = 100_000
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        mask[start:stop] = hull.find_simplex(points[start:stop]) >= 0
    return mask.reshape(grid.shape)


def sphere_index_bounds(center, radius, grid: Grid):
    lower = np.floor((center - radius - grid.origin) / grid.dx).astype(int)
    upper = np.ceil((center + radius - grid.origin) / grid.dx).astype(int)
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, np.array(grid.shape) - 1)
    return lower, upper


def iterate_indices(idx_min, idx_max, shape, use_pbc):
    x0, y0, z0 = [int(value) for value in idx_min]
    x1, y1, z1 = [int(value) for value in idx_max]
    for ix in range(x0, x1 + 1):
        wx = ix % shape[0] if use_pbc else ix
        if wx < 0 or wx >= shape[0]:
            continue
        for iy in range(y0, y1 + 1):
            wy = iy % shape[1] if use_pbc else iy
            if wy < 0 or wy >= shape[1]:
                continue
            for iz in range(z0, z1 + 1):
                wz = iz % shape[2] if use_pbc else iz
                if wz < 0 or wz >= shape[2]:
                    continue
                yield wx, wy, wz


def voxel_center_world(ix, iy, iz, grid: Grid):
    return grid.origin + grid.dx * (np.array([ix, iy, iz], dtype=float) + 0.5)


def minimum_image(dvec, box):
    box = np.asarray(box, dtype=float)
    lengths = box[:3]
    return dvec - lengths * np.round(dvec / lengths)


def find_local_maxima(dist_A, mask, min_distance):
    footprint = np.ones((2 * int(min_distance) + 1,) * 3, dtype=bool)
    filtered = ndimage.maximum_filter(dist_A, footprint=footprint, mode="nearest")
    peaks = (dist_A == filtered) & mask
    return peaks


def filter_connected_pore_regions(pore, criterion="largest"):
    labels, num = ndimage.label(pore)
    if num == 0:
        return pore
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    if criterion == "largest":
        keep_label = int(np.argmax(sizes))
        return labels == keep_label
    raise ValueError(f"Unknown connectivity criterion: {criterion}")


def save_outputs(outputs: dict, output_dir: str):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / "psd_outputs.npz", **{key: value for key, value in outputs.items() if key != "params"})
    with (outdir / "psd_outputs.json").open("w", encoding="utf-8") as handle:
        json.dump(make_jsonable(outputs), handle, indent=2)


def save_visualizations(outputs: dict, output_dir: str):
    outdir = Path(output_dir)
    ensure_dir(outdir)
    plot_psd_pdf(outputs["bin_centers"], outputs["pdf"], outdir / "psd_pdf.png")
    plot_psd_cdf(outputs["bin_centers"], outputs["cdf"], outdir / "psd_cdf.png")

    frame_metrics = outputs.get("frame_metrics", [])
    if frame_metrics:
        frame_idx = [item["frame"] for item in frame_metrics]
        mean_r = [item["mean_radius"] for item in frame_metrics]
        porosity = [item["porosity"] for item in frame_metrics]
        plot_timeseries(frame_idx, mean_r, porosity, outdir / "psd_frame_metrics.png")

    reference_dist = outputs.get("reference_dist")
    if reference_dist is not None:
        dist = np.asarray(reference_dist)
        vmax = float(np.nanmax(dist)) if np.isfinite(dist).any() else None
        plot_dist_slice(dist, outdir / "psd_slice_z.png", axis="z", vmax=vmax)
        plot_dist_slice(dist, outdir / "psd_slice_y.png", axis="y", vmax=vmax)
        plot_dist_slice(dist, outdir / "psd_slice_x.png", axis="x", vmax=vmax)


def dataclass_to_serializable(params: PSDParams):
    data = asdict(params)
    data["bins"] = np.asarray(data["bins"]).tolist()
    return data


def make_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Compute a pore size distribution from a structure file.")
    parser.add_argument("input_file", help="Input structure file, e.g. PDB or GRO")
    parser.add_argument("--traj-file", default=None, help="Optional trajectory file")
    parser.add_argument("--output-dir", default="./psd_output", help="Directory for outputs")
    parser.add_argument("--dx", type=float, default=0.25, help="Grid spacing in Angstrom")
    parser.add_argument("--probe-radius", type=float, default=0.0, help="Probe radius in Angstrom")
    parser.add_argument("--use-pbc", action="store_true", help="Use simplified PBC handling for orthorhombic boxes")
    parser.add_argument("--padding", type=float, default=2.0, help="Padding in Angstrom for non-PBC grids")
    parser.add_argument("--mode", choices=("voxel", "maximal_sphere"), default="voxel")
    parser.add_argument("--r-min", type=float, default=0.1, dest="r_min")
    parser.add_argument("--selection", default="all")
    parser.add_argument("--keep-only-connected-pore", action="store_true")
    parser.add_argument("--peak-min-distance", type=int, default=2)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--bins-start", type=float, default=0.0)
    parser.add_argument("--bins-stop", type=float, default=10.0)
    parser.add_argument("--bins-count", type=int, default=201)
    parser.add_argument("--align-axis", choices=("x", "y", "z"), default="z", help="Axis to align the longest principal axis onto")
    parser.add_argument("--no-align", action="store_true", help="Disable PCA-based structure alignment")
    return parser.parse_args()


def main():
    args = parse_args()
    params = PSDParams(
        dx=args.dx,
        probe_radius=args.probe_radius,
        use_pbc=args.use_pbc,
        padding=args.padding,
        mode=args.mode,
        r_min=args.r_min,
        bins=np.linspace(args.bins_start, args.bins_stop, args.bins_count),
        selection=args.selection,
        keep_only_connected_pore=args.keep_only_connected_pore,
        peak_min_distance=args.peak_min_distance,
        frame_slice=(args.frame_start, args.frame_stop, args.frame_step),
        output_dir=args.output_dir,
        align_structure=not args.no_align,
        align_axis=args.align_axis,
    )
    outputs = run_psd_analysis(args.input_file, args.traj_file, params)
    save_visualizations(outputs, args.output_dir)
    print(json.dumps({"frames": len(outputs["frame_metrics"]), "output_dir": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()