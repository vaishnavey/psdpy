# psdpy

Python implementation for computing pore size distribution in materials using
voxelization and the Euclidean distance transform.

## Usage

Run the analysis on a structure file such as `.pdb` or `.gro`:

```bash
python psd_from_structure.py <structure-file> --dx 1.0 --use-pbc --output-dir <output-dir>
```

Common options:

- `--dx`: grid spacing in Angstrom
- `--probe-radius`: probe radius added to atomic radii
- `--mode`: `voxel` or `maximal_sphere`
- `--selection`: MDAnalysis atom selection string
- `--align-axis`: axis to align the longest principal axis onto (`x`, `y`, or `z`)
- `--no-align`: disable the PCA-based alignment step

## Outputs

Each run writes a result directory containing:

- `psd_outputs.json`: serialized PDF, CDF, frame metrics, and run parameters
- `psd_outputs.npz`: NumPy archive with the same numeric arrays
- `psd_pdf.png`: pore size distribution plot
- `psd_cdf.png`: cumulative distribution plot
- `psd_frame_metrics.png`: per-frame porosity and mean radius trace
- `psd_slice_x.png`: EDT slice along the x axis
- `psd_slice_y.png`: EDT slice along the y axis
- `psd_slice_z.png`: EDT slice along the z axis

## Examples

See [examples/README.md](examples/README.md) for bundled input structures and
saved outputs.

## Reference

Bhattacharya, Supriyo, and Keith E. Gubbins. "Fast method for computing pore
size distributions of model materials." Langmuir 22, no. 18 (2006): 7726-7731.
