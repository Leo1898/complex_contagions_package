"""Command line interface."""
import json
import os

import click
import networkx as nx
import numpy as np

from complex_contagions_package.dataset_creator import create_data_directory, create_ds
from complex_contagions_package.logging import log_error, log_info

data_dir, batches_dir = create_data_directory()
checkpoint_file = os.path.join(data_dir, "checkpoint.json")
config_file = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "..", "..",
                                            "config.json"
                                            ))

def load_config():
    """Load config from json."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found.")

    with open(config_file) as f:
        config = json.load(f)
    return config

def save_config(config):
    """Save custom config to json."""
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)

def check_for_checkpoint():
    """Checks if a checkpoint exists."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            checkpoint_data = json.load(f)
            return checkpoint_data.get("alpha_index", 0) != 0 or checkpoint_data.get(
                "simulation_index", 0
                ) != 0
    return False

def reset_checkpoint():
    """Resets checkpoint to initial state."""
    with open(checkpoint_file, "w") as f:
        json.dump({"alpha_index": 0, "simulation_index": 0}, f)
    log_info("Checkpoint has been reset.")

def delete_simulation_files():
    """Deletes data and batch files, excluding checkpoint.json."""
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if file_name != "checkpoint.json":
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    log_info(f"Deleted file: {file_path}")
            except Exception as e:
                log_error(f"Failed to delete file {file_path}: {e}")

    if os.path.exists(batches_dir):
        for file_name in os.listdir(batches_dir):
            file_path = os.path.join(batches_dir, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    log_info(f"Deleted file in batches: {file_path}")
            except Exception as e:
                log_error(f"Failed to delete file in batches directory: {e}")
    else:
        log_info(f"Batches directory {batches_dir} does not exist or is already empty.")

def generate_t0_values(t0_config):
    """Converts t0 values from config into a list."""
    return np.linspace(
        t0_config["start"],
        t0_config["stop"],
        t0_config["steps"]
    ).tolist()

def generate_alphas(alphas_config):
    """Converts alphas from config into a list."""
    return np.linspace(
        alphas_config["start"],
        alphas_config["stop"],
        alphas_config["steps"]
    ).tolist()

def create_graph(network_type):
    """Creates network according to chosen network type."""
    if network_type == "connected_watts_strogatz":
        return nx.connected_watts_strogatz_graph(100, 4, 0.5)
    elif network_type == "grid_2d":
        return nx.grid_2d_graph(10, 10)
    elif network_type == "barabasi_albert":
        return nx.barabasi_albert_graph(100, 12)
    else:
        raise ValueError(f"Unknown network type: {network_type}")

def start_simulation(config):
    """Helper function to start the simulation."""
    config["t0_values_ascending"] = generate_t0_values(config["t0_values_ascending"])
    config["t0_values_descending"] = generate_t0_values(config["t0_values_descending"])
    config["alphas"] = generate_alphas(config["alphas"])
    config["g_type"] = create_graph(config["network_type"])

    create_ds(
        config["network_type"],
        config["hys_threshold"],
        config["alphas"],
        config["g_type"],
        config["t0_values_ascending"],
        config["t0_values_descending"],
        config["INF_CHANCE"],
        config["steps"],
        config["n_simulations"]
    )

@click.group()
def cli():
    """CLI-Tool."""
    pass

@click.command()
def start():
    """Starts a new simulation."""
    config = load_config()

    if check_for_checkpoint():
        continue_simulation = click.confirm(
            "A checkpoint was found. Do you want to resume the simulation?",
            default=True
        )

        if continue_simulation:
            log_info("Resuming simulation from checkpoint.")
            start_simulation(config)
            return

        reset = click.confirm(
            "Do you want to reset the checkpoint and delete saved files?",
            default=True
        )
        if reset:
            reset_checkpoint()
            delete_simulation_files()
            log_info("Starting a new simulation.")
        else:
            log_info("Exiting without starting a new simulation.")
            return

    use_defaults = click.confirm(
        f"The following settings from config.json will be applied:\n"
        f"Infection Chance: {config["INF_CHANCE"]}\n"
        f"Alpha Values: {config["alphas"]}\n"
        f"Number of Simulations: {config["n_simulations"]}\n"
        f"Steps per Simulation: {config["steps"]}\n"
        f"Network Type: {config["network_type"]}\n"
        f"Hysteresis Threshold: {config["hys_threshold"]}\n"
        f"Do you want to use these settings?",
        default=True
    )

    if use_defaults:
        log_info("User accepted settings from config.json.")
        start_simulation(config)
    else:
        log_info("User opted to provide custom settings.")

        config["network_type"] = click.prompt(
            "Enter the network type ('connected_watts_strogatz', 'grid_2d', "
            "'barabasi_albert')",
            default=config["network_type"]
        )
        config["hys_threshold"] = click.prompt(
            "Enter a new hysteresis threshold",
            default=config["hys_threshold"],
            type=int
        )
        config["n_simulations"] = click.prompt(
            "Enter the number of simulations",
            default=config["n_simulations"],
            type=int
        )
        config["steps"] = click.prompt(
            "Enter the number of steps per simulation",
            default=config["steps"],
            type=int
        )

        save_config(config)

        log_info("Starting simulation with custom settings.")
        start_simulation(config)

@click.command()
def resume():
    """Resumes a simulation based on an existing checkpoint."""
    config = load_config()

    if not check_for_checkpoint():
        click.echo("No checkpoint found. Please start a new simulation.")
        return

    log_info("Resuming simulation from checkpoint.")
    start_simulation(config)

@click.command()
def delete():
    """Clears data and batch folders."""
    if check_for_checkpoint():
        reset = click.confirm(
            "Do you want to reset the checkpoint and delete saved files?",
            default=True
        )
        if reset:
            reset_checkpoint()
            delete_simulation_files()
        else:
            log_info("Exiting.")
            return
    else:
        log_info("No checkpoint found.")

cli.add_command(start)
cli.add_command(resume)
cli.add_command(delete)

if __name__ == "__main__":
    cli()