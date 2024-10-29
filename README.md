## Name
Complex Contagions Package.

## Description
This Python package models the spread of information, infections, or other contagions within complex networks, with a focus on studying hysteresis effects. The simulations explore how various parameters—particularly the infection-to-recovery rate ratio (alpha)—impact the spread and reveal shifts in system behavior as the parameter t0 is adjusted. The t0 parameter specifies the minimum percentage of a node’s neighbors that must be infected for it to become infected in the next iteration step, and it can be varied either ascending or descending. The package supports multiple network structures and includes tools for tracking and analyzing hysteresis gaps.

Key features include:
- Simulations of contagion spread on different network structures.
- Analysis of hysteresis effects, focusing on critical points (`t0` values) and hysteresis gaps.
- Support for batch processing of simulations and incremental saving of results to create a netcdf dataset.
- Visualization of the spread of contagions and hysteresis gaps depending on t0 and alpha.

## Badges
![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Visuals
Please find the exemplary plots of the visualization notebook in the figs folder of the provided project.

## Installation
This is a Python package, using a `pyproject.toml` to define the project's dependencies and metadata,
and `poetry` to manage the Python package and its dependencies.

To install it:

- Install `poetry`, if it is not already installed.
- Create a new virtual environment and activate it.
- Pull the package from github.
- Go to the base directory of the package provided and call `poetry install`.
- After that, you can call the script in your shell with `ccsim
  <arguments>`; e.g.: `ccsim --help`.

The name `ccsim` of the script, and the entry point, are defined
in the `pyproject.toml` section `[tool.poetry.scripts]`.

## Usage
You can choose, wether you want the package to start a new simulation, to resume a simulation based on an existing checkpoint or delete data and batches.

To start a new simulation enter the command `ccsim  start` and follow the instructions for choosing a custom configuration. If confirmed existing data will be deleted.

To resume an existing simulation enter the command `ccsim  resume`. Based on an existing checkpoint the simulation continues.

To delete data and empty data and batch folders enter the command `ccsim  delete`.

Use the provided notebook to visualize the data of an existing netcdf dataset. The project provides an exemplary zipped dataset in the data folder.

## Support
For any issues or questions, please open an issue on the GitHub repository or contact the author.

## Roadmap
tbd

## Contributing
Please feel free to contribute by forking the repository and submitting a pull request.

## Authors and acknowledgment
I appreciate the contribution of Georg Jäger for providing the basic code for network generation, diffusion behavior and simulation logic.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Project status
initial