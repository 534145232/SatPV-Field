# Release manifest

This repository was assembled as a code-only public release from the working research project. The source project was not modified.

## Included

- all Python modules and shell launchers required by the four-stage paper pipeline;
- model baselines and ablation implementations referenced by the manuscript;
- common-intersection evaluation and paired-bootstrap code;
- source-density, operating-condition, spatial-consistency, and matched graph-versus-field experiments;
- manuscript figure-generation scripts;
- portable path configuration, environment checks, and public documentation.

## Excluded

- raw PV, H8, NWP, and station metadata;
- generated NetCDF/NumPy/HDF5/raster artifacts;
- cached field tensors and future-SWR anchors;
- model checkpoints and optimizer states;
- result CSV/JSON files, figures, logs, and temporary diagnostics;
- Python bytecode, IDE metadata, and machine-specific virtual environments;
- third-party Natural Earth shapefiles and downloaded map tiles.

## Portability changes in the public copy

- the repository no longer needs to be named `Final_3_stage_pipeline`;
- shell launchers resolve their own directory and default to `python`/`$PYTHON`;
- data, cache, and checkpoint locations are controlled by environment variables;
- Stage 1--4 path modules accept a shared external `SATPV_DATA_ROOT`;
- the original server-specific source and data paths remain untouched in the private working project.

## Verification performed

- all copied Python files passed `compileall`;
- no data/checkpoint extension listed in `.gitignore` was present in the release tree;
- no original local Windows path, user-storage path, Bitahub host id, or obvious credential literal was found by the release scan;
- the repository was initialized locally with branch `main`, but no commit or remote was created.

## License

Repository source code is released under the MIT License. Data, trained weights, and third-party assets are not included and are not covered by that license.

