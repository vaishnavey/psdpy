# PSD Examples

This folder contains sample structures and the generated PSD outputs used to
test the voxelization pipeline.

## Inputs

- `inputs/1PHO.pdb`
- `inputs/2OMF.pdb`
- `inputs/em_test.gro`
- `inputs/packed_box_test.pdb`

## Results

Each subfolder under `results/` contains the output for one input structure:

- `psd_pdf.png`
- `psd_cdf.png`
- `psd_frame_metrics.png`
- `psd_slice_x.png`
- `psd_slice_y.png`
- `psd_slice_z.png`
- `psd_outputs.json`
- `psd_outputs.npz`

To regenerate an example, run:

```bash
python psd_from_structure.py <structure-file> --dx 1.0 --use-pbc --output-dir <output-dir>
```