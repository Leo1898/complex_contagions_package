"""Dataset creation."""
import json
import os
from multiprocessing import Manager, Pool

import numpy as np
import xarray as xr
from tqdm import tqdm

from complex_contagions_package.logging import log_error, log_info
from complex_contagions_package.simulator import run_hysteresis_simulation


def create_data_directory(base_dir=None, dataset_dir=None):
    """Create or return data, dataset and batch directory for storing results."""
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "data"
            ))

    if dataset_dir is None:
        dataset_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "datasets"
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
    ds_alpha = xr.Dataset(
        {
            "inflist_asc": (("simulation", "t0", "steps"), np.empty(
                (n_simulations, len(t0_values_ascending), steps)
                )),
            "inflist_desc": (("simulation", "t0", "steps"), np.empty(
                (n_simulations, len(t0_values_descending), steps)
                )),

            # Clusterdaten:
            "max_cluster_size_asc": (("simulation", "t0"), np.empty(
                (n_simulations, len(t0_values_ascending)))),
            "max_cluster_size_desc": (("simulation", "t0"), np.empty(
                (n_simulations, len(t0_values_descending)))),
            "mean_cluster_size_asc": (("simulation", "t0"), np.empty(
                (n_simulations, len(t0_values_ascending)))),
            "mean_cluster_size_desc": (("simulation", "t0"), np.empty(
                (n_simulations, len(t0_values_descending)))),
            "largest_cluster_fraction_asc": (("simulation", "t0"), np.empty(
                (n_simulations, len(t0_values_ascending)))),
            "largest_cluster_fraction_desc": (("simulation", "t0"), np.empty(
                (n_simulations, len(t0_values_descending)))),
            "n_clusters_asc": (("simulation", "t0"), np.empty(
                (n_simulations, len(t0_values_ascending)))),
            "n_clusters_desc": (("simulation", "t0"), np.empty(
                (n_simulations, len(t0_values_descending)))),
        },
        coords={
            "simulation": np.arange(1, n_simulations+1),
            "t0": t0_values_ascending,
            "steps": np.arange(1, steps+1),
        }
    )

    return ds_alpha

def run_simulation_wrapper(i, g_type, steps,
                           t0_values_ascending, t0_values_descending,
                           inf_chance, rec_chance,
                           #network_states
                           network_states_ascending, network_states_descending
                           ):
    """Wrapper für die Simulation, der den Index und die Ergebnisse zurückgibt."""
    inflist_ascending, _= run_hysteresis_simulation(
        i, g_type, steps,
        t0_values_ascending, t0_values_descending,
        inf_chance, rec_chance,
        #network_states
        network_states_ascending, network_states_descending
        )
    _, inflist_descending = run_hysteresis_simulation(
        i, g_type, steps,
        t0_values_ascending, t0_values_descending,
        inf_chance, rec_chance,
        #network_states
        network_states_ascending, network_states_descending
        )
    inflist_asc = np.array([x[2] for x in inflist_ascending])
    inflist_desc = np.array([x[2] for x in inflist_descending])[::-1]

    max_cluster_size_asc = np.array([x[3]["max_cluster_size"]
                                    for x in inflist_ascending])
    max_cluster_size_desc= np.array([x[3]["max_cluster_size"]
                                    for x in inflist_descending])[::-1]

    mean_cluster_asc = np.array([x[3]["mean_cluster_size"]
                                    for x in inflist_ascending])
    mean_cluster_desc = np.array([x[3]["mean_cluster_size"]
                                    for x in inflist_descending])[::-1]

    largest_cluster_fraction_asc = np.array([x[3]["largest_cluster_fraction"]
                                    for x in inflist_ascending])
    largest_cluster_fraction_desc = np.array([x[3]["largest_cluster_fraction"]
                                    for x in inflist_descending])[::-1]

    n_clusters_asc = np.array([x[3]["n_clusters"] for x in inflist_ascending])
    n_clusters_desc = np.array([x[3]["n_clusters"] for x in inflist_descending])[::-1]

    return i, inflist_asc, inflist_desc, \
           max_cluster_size_asc, max_cluster_size_desc, \
           mean_cluster_asc, mean_cluster_desc, \
           largest_cluster_fraction_asc, largest_cluster_fraction_desc, \
           n_clusters_asc, n_clusters_desc

def parallel_simulations(steps, t0_values_ascending, t0_values_descending,
                         inf_chance, rec_chance, g_type, n_simulations, ds_alpha):
    """Führt alle Simulationen für ein bestimmtes Alpha parallel aus."""
    with Manager() as manager:
        #network_states = manager.dict()  # Gemeinsames Dictionary für Netzwerkzustände
        network_states_ascending = manager.dict()
        network_states_descending = manager.dict()
        # Aufsteigende Simulationen ausführen und WARTEN, bis sie fertig sind

        for i in range(n_simulations):
            network_states_ascending[i] = manager.dict()
            network_states_descending[i] = manager.dict()

        with Pool(processes=os.cpu_count()) as pool:
            results = [
                pool.apply_async(
                    run_simulation_wrapper,
                    args=(i, g_type, steps, t0_values_ascending, t0_values_descending,
                          inf_chance, rec_chance,
                          #network_states)
                          network_states_ascending, network_states_descending)
                )
                for i in range(n_simulations)
            ]

            for result in results:
                (
                idx, inflist_asc, inflist_desc,
                max_cluster_asc, max_cluster_desc,
                mean_cluster_asc, mean_cluster_desc,
                frac_cluster_asc, frac_cluster_desc,
                n_clusters_asc, n_clusters_desc
                 ) = result.get()  # Warten, bis ALLE aufsteigenden fertig sind

                ds_alpha["inflist_asc"][idx, :, :] = inflist_asc
                ds_alpha["inflist_desc"][idx, :, :] = inflist_desc

                ds_alpha["max_cluster_size_asc"][idx, :] = max_cluster_asc
                ds_alpha["max_cluster_size_desc"][idx, :] = max_cluster_desc

                ds_alpha["largest_cluster_fraction_asc"][idx, :] = frac_cluster_asc
                ds_alpha["largest_cluster_fraction_desc"][idx, :] = frac_cluster_desc

                ds_alpha["mean_cluster_size_asc"][idx, :] = mean_cluster_asc
                ds_alpha["mean_cluster_size_desc"][idx, :] = mean_cluster_desc

                ds_alpha["n_clusters_asc"][idx, :] = n_clusters_asc
                ds_alpha["n_clusters_desc"][idx, :] = n_clusters_desc

    return ds_alpha


# def parallel_simulations(steps, t0_values_ascending, t0_values_descending,
#                          inf_chance, rec_chance, g_type, n_simulations, ds_alpha):
#     """Führt alle Simulationen für ein bestimmtes Alpha parallel aus."""
#     pool = Pool(processes=os.cpu_count())  # Verwendet alle verfügbaren CPU-Kerne

#     # Parallele Ausführung der Simulationen
#     results = [
#         pool.apply_async(
#             run_simulation_wrapper,
#             args=(i, g_type, steps,
#                   t0_values_ascending, t0_values_descending,
#                   inf_chance, rec_chance
#                   )
#         )
#         for i in range(n_simulations)
#     ]

#     for result in results:
#         idx, inflist_asc, inflist_desc = result.get()  # Ergebnis abrufen
#         ds_alpha["inflist_asc"][idx, :, :] = inflist_asc
#         ds_alpha["inflist_desc"][idx, :, :] = inflist_desc

#     pool.close()
#     pool.join()

#     return ds_alpha

def create_ds(network_type, average_degree, alphas, g_type,
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

    start_alpha_index = checkpoint["alpha_index"] if checkpoint else 0

    with tqdm(total=len(alphas),
            desc=f"Simulating for network type {network_type} degree {average_degree}",
            unit="alpha",
            initial=start_alpha_index) as pbar:

        #Loop through alphas list
        for alpha_index, alpha in enumerate(
            alphas[start_alpha_index:],
            start=start_alpha_index
            ):

            ds_alpha = initialize_ds(t0_values_ascending,
                                        t0_values_descending,
                                        steps,
                                        n_simulations
                                        )

            rec_chance = inf_chance / alpha

            ds_alpha = parallel_simulations(steps,
                                            t0_values_ascending, t0_values_descending,
                                            inf_chance, rec_chance,
                                            g_type, n_simulations, ds_alpha)

            # Save the final dataset for the current alpha
            final_filename = os.path.join(data_dir,
                                          f"final_simulation_alpha_{alpha}.nc"
                                          )
            ds_alpha.to_netcdf(final_filename)
            log_info(f"Final dataset for alpha={alpha} saved as\n"
                    f"{final_filename}")

            pbar.refresh()

            del ds_alpha

            save_checkpoint(checkpoint_file, {
            "alpha_index": alpha_index + 1
            })

            pbar.update(1)

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
            ds = ds.assign_coords(alpha=(["alpha"], [alphas[idx]]))
            datasets.append(ds)

        final_ds = xr.concat(datasets, dim="alpha")

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

        merged_filename = os.path.join(dataset_dir, f"cl_orange_gruen_final_ds_{network_type}_degree{
            average_degree}.nc")
        final_ds.to_netcdf(merged_filename)
        log_info(f"Merged datasets saved as {merged_filename} in {data_dir}")
    else:
        log_error("Final merge failed. Too few data files.")
