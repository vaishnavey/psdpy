params = {
    "dx": 0.25,                    # Å
    "probe_radius": 0.0,           # Å (set >0 for accessible pore definition)
    "use_pbc": True,
    "padding": 2.0,                # Å, only used if non-PBC
    "mode": "voxel",               # "voxel" or "maximal_sphere"
    "r_min": 0.1,                  # Å
    "bins": np.linspace(0.0, 10.0, 201),
    "selection": "all",
    "keep_only_connected_pore": False,
    "connectivity_criterion": "largest",
    "peak_min_distance": 2,        # voxels, for maximal_sphere mode
    "frame_slice": (0, None, 1),
    "default_radius": 1.7,
    "radii_table": {...},
    "output_dir": "./psd_output"
}