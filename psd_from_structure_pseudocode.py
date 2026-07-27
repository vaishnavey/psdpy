# Pseudocode for PSD from .pdb/.gro via voxelization + EDT

function run_psd_analysis(input_file, traj_file=None, params):
    # params:
    # dx, probe_radius, use_pbc, padding, bins, mode, r_min, selection, etc.

    universe = load_structure(input_file, traj_file)  # MDAnalysis-like object
    frames = get_frames(universe, params.frame_slice)

    all_hist_counts = zeros(len(params.bins)-1)
    frame_metrics = []

    for frame in frames:
        coords, elements, box = extract_frame_data(universe, frame, params.selection)
        radii = assign_vdw_radii(elements, params.radii_table, params.default_radius)
        r_eff = radii + params.probe_radius

        grid = build_grid(coords, box, params.dx, params.use_pbc, params.padding, r_eff)
        # grid contains origin, shape (Nx,Ny,Nz), spacing dx, maybe box vectors

        solid = rasterize_atoms_to_solid(coords, r_eff, grid, params.use_pbc)
        pore = logical_not(solid)

        if params.keep_only_connected_pore:
            pore = filter_connected_pore_regions(pore, criterion=params.connectivity_criterion)

        dist_vox = distance_transform_edt(pore)   # scipy.ndimage
        dist_A = dist_vox * params.dx

        if params.mode == "voxel":
            radii_samples = dist_A[pore]
        else if params.mode == "maximal_sphere":
            peaks_mask = find_local_maxima(dist_A, mask=pore, min_distance=params.peak_min_distance)
            radii_samples = dist_A[peaks_mask]
        else:
            raise ValueError("Unknown PSD mode")

        radii_samples = radii_samples[radii_samples > params.r_min]

        hist_counts = histogram(radii_samples, bins=params.bins, density=False)
        all_hist_counts += hist_counts

        porosity = count_true(pore) / pore.size
        frame_metrics.append({
            "frame": frame.index,
            "porosity": porosity,
            "mean_radius": mean(radii_samples),
            "max_radius": max(radii_samples)
        })

    # Normalize histogram to PDF
    bin_widths = diff(params.bins)
    total = sum(all_hist_counts)
    pdf = all_hist_counts / (total * bin_widths)
    cdf = cumulative_sum(all_hist_counts) / total
    bin_centers = 0.5 * (params.bins[:-1] + params.bins[1:])

    outputs = {
        "bin_centers": bin_centers,
        "pdf": pdf,
        "cdf": cdf,
        "frame_metrics": frame_metrics,
        "params": params
    }

    save_outputs(outputs, params.output_dir)
    return outputs


function build_grid(coords, box, dx, use_pbc, padding, r_eff):
    if use_pbc and box is valid:
        Lx, Ly, Lz = get_orthorhombic_lengths(box)
        origin = [0, 0, 0]
        Nx, Ny, Nz = ceil(Lx/dx), ceil(Ly/dx), ceil(Lz/dx)
    else:
        rmax = max(r_eff)
        minc = min(coords, axis=0) - (padding + rmax)
        maxc = max(coords, axis=0) + (padding + rmax)
        origin = minc
        lengths = maxc - minc
        Nx, Ny, Nz = ceil(lengths[0]/dx), ceil(lengths[1]/dx), ceil(lengths[2]/dx)

    return Grid(origin=origin, dx=dx, shape=(Nx,Ny,Nz), box=box, use_pbc=use_pbc)


function rasterize_atoms_to_solid(coords, r_eff, grid, use_pbc):
    solid = false_array(grid.shape)

    for i in range(len(coords)):
        c = coords[i]
        r = r_eff[i]

        # Determine index bounding box around atom
        idx_min, idx_max = sphere_index_bounds(c, r, grid)

        # Clamp or wrap index bounds based on boundary mode
        for ix, iy, iz in iterate_indices(idx_min, idx_max, grid.shape, use_pbc):
            x = voxel_center_world(ix, iy, iz, grid)

            dvec = x - c
            if use_pbc:
                dvec = minimum_image(dvec, grid.box)

            if dot(dvec, dvec) <= r*r:
                solid[ix, iy, iz] = True

    return solid