"""Dataset creation."""
import json
import os
import re
from multiprocessing import Pool

import numpy as np
import xarray as xr
from tqdm import tqdm

from complex_contagions_package.logging import log_error, log_info
from complex_contagions_package.simulator import run_hysteresis_simulation


def create_data_directory(base_dir=None, dataset_dir=None):
    """Create or return data, dataset and batch directory for storing results."""
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "dataForRec"
            ))

    if dataset_dir is None:
        dataset_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "datasetsForRec"
            ))

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    return base_dir, dataset_dir

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

def initialize_ds(
        t0_values_ascending,
        t0_values_descending,
        steps,
        n_simulations
    ):
    """Initialize empty dataset."""
    ds_rec = xr.Dataset(
        {
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

    return ds_rec

def run_simulation_wrapper(i, g_type, steps,
                           t0_values_ascending, t0_values_descending,
                           inf_chance, rec_chance
                           ):
    """Wrapper für die Simulation, der den Index und die Ergebnisse zurückgibt."""
    inflist_ascending, inflist_descending = run_hysteresis_simulation(
        g_type, steps,
        t0_values_ascending, t0_values_descending,
        inf_chance, rec_chance
        )
    inflist_asc = np.array([x[2] for x in inflist_ascending])
    inflist_desc = np.array([x[2] for x in inflist_descending])[::-1]
    return i, inflist_asc, inflist_desc

def parallel_simulations(steps, t0_values_ascending, t0_values_descending,
                         inf_chance, rec_chance, g_type, n_simulations, ds_rec):
    """Führt alle Simulationen für ein bestimmtes Alpha parallel aus."""
    pool = Pool(processes=os.cpu_count())  # Verwendet alle verfügbaren CPU-Kerne

    # Parallele Ausführung der Simulationen
    results = [
        pool.apply_async(
            run_simulation_wrapper,
            args=(i, g_type, steps,
                  t0_values_ascending, t0_values_descending,
                  inf_chance, rec_chance
                  )
        )
        for i in range(n_simulations)
    ]

    for result in results:
        idx, inflist_asc, inflist_desc = result.get()  # Ergebnis abrufen
        ds_rec["inflist_asc"][idx, :, :] = inflist_asc
        ds_rec["inflist_desc"][idx, :, :] = inflist_desc

    pool.close()
    pool.join()

    return ds_rec

def create_ds(network_type, average_degree, recovery_rates, g_type,
              t0_values_ascending, t0_values_descending,
              inf_chance, steps, n_simulations,
              base_data_dir=None, base_dataset_dir=None
              ):
    """Runs simulations for various alpha values and stores results in netCDF format.

    Simulates infection spread over a network for different alpha values,
    saving intermediate results in batch files. After all simulations for each alpha are
    completed, the results are merged into a final dataset.

    Parameters:
        network_type (str): Type of network used in the simulations.
        average_degree (int): network parameter defining the average number of links
                              associated with a node.
        alphas (list): List of alpha values (infection/recovery rate ratios).
        g_type (networkx.Graph): The network structure for simulations.
        t0_values_ascending (list): List of ascending `t0` values.
        t0_values_descending (list): List of descending `t0` values.
        inf_chance (float): Infection probability.
        steps (int): Number of steps in each simulation.
        n_simulations (int): Number of simulations to run per alpha.
        base_data_dir (str, optional): Directory to save the final results.
        base_dataset_dir (str, optional): Directory to store final dataset files.
    """
    data_dir, dataset_dir = create_data_directory(base_data_dir,
                                                  base_dataset_dir
                                                  )

    checkpoint_file = os.path.join(data_dir, "checkpoint.json")
    checkpoint = load_checkpoint(checkpoint_file)

    start_rec_index = checkpoint["rec_index"] if checkpoint else 0

    with tqdm(total=len(recovery_rates),
            desc=f"Simulating for network type {network_type} degree {average_degree}",
            unit="rec",
            initial=start_rec_index) as pbar:

        #Loop through alphas list
        for rec_index, rec in enumerate(
            recovery_rates[start_rec_index:],
            start=start_rec_index
            ):

            ds_rec = initialize_ds(t0_values_ascending,
                                        t0_values_descending,
                                        steps,
                                        n_simulations
                                        )

            rec_chance = rec

            ds_rec = parallel_simulations(steps,
                                            t0_values_ascending, t0_values_descending,
                                            inf_chance, rec_chance,
                                            g_type, n_simulations, ds_rec)

            # Save the final dataset for the current alpha
            final_filename = os.path.join(data_dir,
                                          f"final_simulation_recChance_{rec}.nc"
                                          )
            ds_rec.to_netcdf(final_filename)
            log_info(f"Final dataset for recChance={rec} saved as\n"
                    f"{final_filename}")

            pbar.refresh()

            del ds_rec

            save_checkpoint(checkpoint_file, {
            "rec_index": rec_index + 1
            })

            pbar.update(1)

    rec_files = []
    rec_files = sorted(
    [f for f in os.listdir(data_dir) if "final_simulation_recChance_" in f],
    key=lambda x: float(re.search(r"final_simulation_recChance_(\d+\.\d+)", x).group(1))
    )

    # Merge all saved datasets
    if len(rec_files) == len(recovery_rates):
        datasets = []

        for idx, f in enumerate(rec_files):
            ds = xr.open_dataset(os.path.join(data_dir, f))
            ds = ds.expand_dims("recovery_rate")
            ds = ds.assign_coords(recovery_rate=(["recovery_rate"],
                                                  [recovery_rates[idx]]))
            datasets.append(ds)

        final_ds = xr.concat(datasets, dim="recovery_rate")

        # Add attributes to the merged dataset
        if network_type in ["connected_watts_strogatz", "random_regular_graph"]:
            average_degree = average_degree
        else:
            average_degree = 99

        final_ds.attrs = {
            "n_simulations": n_simulations,
            "iterations": steps,
            "network_type": network_type,
            "average_degree": average_degree
        }

        merged_filename = os.path.join(dataset_dir, f"final_ds_{
            network_type}_degree{average_degree}.nc")
        final_ds.to_netcdf(merged_filename)
        log_info(f"Merged datasets saved as {merged_filename} in {data_dir}")
    else:
        log_error("Final merge failed. Too few data files.")
