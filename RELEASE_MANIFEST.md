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

- the repository does not require a particular checkout directory name;
- shell launchers resolve their own directory and default to `python`/`$PYTHON`;
- data, cache, and checkpoint locations are controlled by environment variables;
- Stage 1--4 path modules accept a shared external `SATPV_DATA_ROOT`;
- the private working project remains untouched; portability changes are confined to this public copy.

## Verification performed

- all copied Python files passed `compileall`;
- no data/checkpoint extension listed in `.gitignore` was present in the release tree;
- no original local Windows path, user-storage path, Bitahub host id, or obvious credential literal was found by the release scan;
- the automated public-release audit checks syntax, path hygiene, accidental artifacts, secrets, and Markdown links on every push and pull request.

## License

Repository source code is released under the MIT License. Data, trained weights, and third-party assets are not included and are not covered by that license.

