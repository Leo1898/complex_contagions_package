"""Dataset creation."""
import json
import os

import numpy as np
import xarray as xr
from tqdm import tqdm

from complex_contagions_package.analyser import calculate_hysteresis_gaps
from complex_contagions_package.logging import log_error, log_info


def create_data_directory(base_dir=None, batches_dir=None):
    """Create or return existing data and batch directory for storing results."""
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "data"
            ))
    if batches_dir is None:
        batches_dir = os.path.join(base_dir, "batches")

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if not os.path.exists(batches_dir):
        os.makedirs(batches_dir)

    return base_dir, batches_dir

def load_checkpoint(checkpoint_file):
    """Load the checkpoint from a JSON file."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            return json.load(f)
    return None

def save_checkpoint(checkpoint_file, checkpoint_data):
    """Save the checkpoint to a JSON file."""
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f)

def delete_batches(batch_files):
    """Delete batch files after saving the final alpha dataset."""
    for batch_file in batch_files:
        if os.path.exists(batch_file):
            os.remove(batch_file)
            log_info(f"Deleted batch file: {batch_file}")

def reload_last_batch(
        t0_values_ascending,
        t0_values_descending,
        steps,
        batch_files,
        n_simulations
    ):
    """Reload the last saved batch file or initialize empty dataset."""
    if batch_files:
        last_batch_file = batch_files[-1]
        ds_alpha = xr.open_dataset(last_batch_file)
        last_simulation_index = len(ds_alpha["simulation"])

        if last_simulation_index < n_simulations:
            # Create empty entries for remaining simulations
            ds_alpha = xr.concat(
                [ds_alpha, xr.Dataset(
                    {
                        "hysteresis_gaps": (("simulation"), np.empty(
                            n_simulations - last_simulation_index
                            )),
                        "inflist_asc": (("simulation", "t0", "steps"), np.empty(
                            (n_simulations - last_simulation_index,
                            len(t0_values_ascending), steps)
                            )),
                        "inflist_desc": (("simulation", "t0", "steps"), np.empty(
                            (n_simulations - last_simulation_index,
                            len(t0_values_descending), steps))),
                    },
                    coords={
                        "simulation": np.arange(
                            last_simulation_index + 1, n_simulations + 1
                            ),
                        "t0": t0_values_ascending,
                        "steps": np.arange(1, steps + 1),
                    }
                )], dim="simulation")

        log_info(f"Restored {last_simulation_index} simulations from last batch:\n"
                f"{last_batch_file}")

    else:
        ds_alpha = xr.Dataset(
            {
                "hysteresis_gaps": (("simulation"), np.empty(n_simulations)),
                "inflist_asc": (("simulation", "t0", "steps"), np.empty(
                    (n_simulations, len(t0_values_ascending), steps)
                    )),
                "inflist_desc": (("simulation", "t0", "steps"), np.empty(
                    (n_simulations, len(t0_values_descending), steps)
                    )),
            },
            coords={
                "simulation": np.arange(1, n_simulations+1),
                "t0": t0_values_ascending,
                "steps": np.arange(1, steps+1),
            }
        )
        last_simulation_index = 0

    return ds_alpha, last_simulation_index


def create_ds(network_type,
              hys_threshold, alphas, g_type,
              t0_values_ascending, t0_values_descending,
              inf_chance, steps, n_simulations,
              base_data_dir=None, base_batch_dir=None
              ):
    """Runs simulations for various alpha values and stores results in netCDF format.

    Simulates infection spread over a network for different alpha values,
    saving intermediate results in batch files. After all simulations for each alpha are
    completed, the results are merged into a final dataset.

    Parameters:
        network_type (str): Type of network used in the simulations.
        hys_threshold (int): Threshold for detecting hysteresis.
        alphas (list): List of alpha values (infection/recovery rate ratios).
        g_type (networkx.Graph): The network structure for simulations.
        t0_values_ascending (list): List of ascending `t0` values.
        t0_values_descending (list): List of descending `t0` values.
        inf_chance (float): Infection probability.
        steps (int): Number of steps in each simulation.
        n_simulations (int): Number of simulations to run per alpha.
        base_data_dir (str, optional): Directory to save the final results.
        base_batch_dir (str, optional): Directory to store intermediate batch files.
    """
    data_dir, batches_dir = create_data_directory(base_data_dir, base_batch_dir)
    checkpoint_file = os.path.join(data_dir, "checkpoint.json")
    checkpoint = load_checkpoint(checkpoint_file)

    start_alpha_index = checkpoint["alpha_index"] if checkpoint else 0

    batch_files = []

    #Loop through alphas list
    for alpha_index, alpha in enumerate(
        alphas[start_alpha_index:],
        start=start_alpha_index
        ):
        if checkpoint and checkpoint["alpha_index"] == alpha_index:
            start_simulation_index = checkpoint["simulation_index"]
            batch_files = sorted(
                [os.path.join(batches_dir, f)
                for f in os.listdir(batches_dir)
                if f"batch_simulation_alpha_{alpha}_" in f],
                key=lambda x: int(x.split("_")[-1].split(".")[0])
                )
            ds_alpha, completed_simulations = reload_last_batch(t0_values_ascending,
                                                                t0_values_descending,
                                                                steps,
                                                                batch_files,
                                                                n_simulations
                                                                )
        else:
            start_simulation_index = 0
            ds_alpha, completed_simulations = reload_last_batch(t0_values_ascending,
                                                                t0_values_descending,
                                                                steps,
                                                                batch_files,
                                                                n_simulations
                                                                )

        rec_chance = inf_chance / alpha

        #Progress bar
        with tqdm(
            total=n_simulations,
            desc=f"Simulation for alpha={alpha}",
            unit="sim",
            initial=start_simulation_index
            ) as pbar:

            #Loop for simulations
            for i in range(start_simulation_index, n_simulations):
                inflist_ascending, inflist_descending, gaps = calculate_hysteresis_gaps(
                    hys_threshold, alpha, g_type,
                    t0_values_ascending, t0_values_descending,
                    inf_chance, rec_chance, steps
                )

                inflist_asc = np.array([x[2] for x in inflist_ascending])
                inflist_desc = np.array([x[2] for x in inflist_descending])[::-1]

                ds_alpha["inflist_asc"][i, :, :] = inflist_asc
                ds_alpha["inflist_desc"][i, :, :] = inflist_desc

                if gaps:
                    ds_alpha["hysteresis_gaps"][i] = gaps[0]
                else:
                    ds_alpha["hysteresis_gaps"][i] = np.nan

                completed_simulations += 1
                pbar.update(1)

                # Batch saving every 10 simulations
                if (i + 1) % 10 == 0:
                    batch_filename = os.path.join(
                        batches_dir,
                        f"batch_simulation_alpha_{alpha}_to_{i + 1}.nc"
                        )
                    ds_alpha.isel(simulation=slice(0, i + 1)).to_netcdf(batch_filename)
                    batch_files.append(batch_filename)
                    log_info(f"Saved batch for alpha={alpha} after {i + 1} "
                             "simulations as\n"
                             f"{batch_filename}")

                    #Save checkpoint after every 10th simulation
                    save_checkpoint(checkpoint_file, {
                        "alpha_index": alpha_index + 1
                        if i + 1 == n_simulations
                        else alpha_index,
                        "simulation_index": 0
                        if i + 1 == n_simulations
                        else i + 1
                    })

        # Save the final dataset for the current alpha
        final_filename = os.path.join(data_dir, f"final_simulation_alpha_{alpha}.nc")
        ds_alpha.to_netcdf(final_filename)
        log_info(f"Final dataset for alpha={alpha} saved as\n"
                 f"{final_filename}")

        # Delete batch files after the final dataset is saved
        delete_batches(batch_files)
        batch_files.clear()

        del ds_alpha

    alpha_files = []
    alpha_files = sorted([f for f in os.listdir(data_dir)
                          if "final_simulation_alpha_" in f],
                        key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Merge all saved datasets
    if len(alpha_files) == len(alphas):
        datasets = []

        for idx, f in enumerate(alpha_files):
            ds = xr.open_dataset(os.path.join(data_dir, f))
            ds = ds.expand_dims("alpha")
            ds = ds.assign_coords(alpha=(["alpha"], [idx + 1]))
            datasets.append(ds)

        final_ds = xr.concat(datasets, dim="alpha")

        # Add attributes to the merged dataset
        final_ds.attrs = {
            "n_simulations": n_simulations,
            "iterations": steps,
            "hys_threshold": hys_threshold,
            "network_type": network_type,
        }

        merged_filename = os.path.join(data_dir, "final_simulation_dataset.nc")
        final_ds.to_netcdf(merged_filename)
        log_info(f"Merged datasets saved as {merged_filename} in {data_dir}")
    else:
        log_error("Final merge failed. Too few data files.")
