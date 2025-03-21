"""Visualizer module."""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from complex_contagions_package.analyser2 import hysteresis_calc


def heatmap_generator(input_folder=None, output_folder=None, network = "random"):
    """Creates a heat map.

    Heatmap to visualize the distribution of hysteresis areas
    for each alpha value based on their frequency of occurrence.
    """
    input_folder = input_folder or os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "datasetsForRec"
            ))

    output_folder = output_folder or os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "heatmapsForRec"
            ))

    alpha_files = sorted(
    [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.startswith(
        f"final_ds_{network}")],
    key=lambda x: int(re.search(r"degree(\d+)", x).group(1))  # Degree-Zahl extrahieren
    )

    # Merge all saved datasets
    for i in alpha_files:
        ds = xr.open_dataset(i)

        simulation = ds.coords["simulation"].values
        rec_chances = ds.coords["recovery_rate"].values
        t0 = ds.coords["t0"].values
        network_type = ds.attrs.get("network_type", "Unknown")
        average_degree = ds.attrs.get("average_degree", "Unknown")

        hysteresis_df, _ = hysteresis_calc(simulation, ds, rec_chances, t0)

        # Prepare histogram data for heatmap
        bin_edges = np.linspace(-10, 60, 71)  # Fixed y-axis range from -10 to 60
        hysteresis_df['hysteresis_bin'] = pd.cut(hysteresis_df['hysteresis_area'],
                                                bins=bin_edges)

        # Frequency table
        heatmap_data = (
            hysteresis_df.groupby(['recovery_rate', 'hysteresis_bin'])
            .size()
            .unstack(fill_value=0)
        )

        # X- and Y-labels
        heatmap_data.columns = heatmap_data.columns.map(
            lambda x: f"{int(x.left)}-{int(x.right)}"
            )

        heatmap_data.index = [
            f"{recovery_rate:.2f}" for recovery_rate in heatmap_data.index
            ]

        # Heatmap
        plt.figure(figsize=(15, 10))
        sns.heatmap(
            heatmap_data.T,  # Transponierte Tabelle für Darstellung
            cmap="turbo",
            cbar_kws={'label': 'Frequency'},
            xticklabels=heatmap_data.index,
            yticklabels=False,
            vmin=0,
            vmax=100,
        )

        plt.yticks(ticks=np.arange(len(bin_edges)), labels=bin_edges.astype(int))

        # Fix y-axis range and invert it
        plt.ylim(0, len(bin_edges) - 1)  # Ensure all bins are shown

        plt.title(f"Heatmap of Hysteresis distribution for {network_type} (Degree: {
            average_degree})")
        plt.xlabel('Recovery chance')
        plt.ylabel('Hysteresis area')
        plt.tight_layout()
        #plt.show()

        frame_path = os.path.join(output_folder, f"final_ds_{network_type}_degree{
        average_degree}.png")
        plt.savefig(frame_path)
        plt.close()

class DiffusionPlotter:
    """Plotter class."""
    def __init__(self, dataset=None, dataset_path=None):
        """Initialization function."""
        if dataset_path:
            self.ds = xr.open_dataset(dataset_path)
        elif dataset is not None:
            self.ds = dataset
        else:
            raise ValueError("Either a dataset or a dataset_path must be provided.")

        self.available_simulation = self.ds.coords["simulation"].values
        self.available_t0 = self.ds.coords["t0"].values
        self.available_rec_chances= self.ds.coords["recovery_rate"].values

    def multiple_instances(self, dataset_paths):
        """Creates DiffusionPlotter instances based on the passed dataset paths."""
        # Sicherstellen, dass dataset_paths immer eine Liste ist
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]

        return [DiffusionPlotter(dataset_path=path) for path in dataset_paths]

    def hysteresis_boxplot(self):
        """Creates a boxplot.

        Boxplot for the distribution of hysteresis per alpha value.
        """
        simulation = self.available_simulation
        ds = self.ds
        rec_chances = self.available_rec_chances
        t0 = self.available_t0
        network_type = ds.attrs.get("network_type", "Unknown")
        average_degree = ds.attrs.get("average_degree", "Unknown")

        hysteresis_df, _ = hysteresis_calc(simulation, ds,
                                           rec_chances,
                                           t0)

        fig, ax_main = plt.subplots(figsize=(15, 8))

        hysteresis_df.boxplot(column='hysteresis_area',
                              by='recovery_rate',
                              ax=ax_main,
                                    grid=False
                                    )


        # Definiere die gewünschten Ticks im Bereich 0 bis 1
        x_ticks = np.arange(0, 1.01, 0.01)
        x_labels = [f"{tick:.2f}" for tick in x_ticks]

        # Setze die Ticks und Labels
        ax_main.set_xticks(np.arange(1, len(x_ticks) + 1))
        ax_main.set_xticklabels(x_labels, rotation=90)

        ax_main.set_title(f"Hysteresis boxplot per rec_chance for {
            network_type} (Degree: {average_degree})")

        ax_main.set_xlabel('rec_chance')
        ax_main.set_ylabel('Hysteresis')
        plt.suptitle('')
        plt.tight_layout()
        plt.show()

    def hysteresis_heatmap(self):
        """Creates a heat map.

        Heatmap to visualize the distribution of hysteresis areas
        for each recovery chance value based on their frequency of occurrence.
        """
        output_folder = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "heatmapsForRec"
            ))

        simulation = self.available_simulation
        ds = self.ds
        rec_chances = self.available_rec_chances
        t0 = self.available_t0
        network_type = ds.attrs.get("network_type", "Unknown")
        average_degree = ds.attrs.get("average_degree", "Unknown")

        hysteresis_df, _ = hysteresis_calc(simulation, ds, rec_chances, t0)

        # Prepare histogram data for heatmap
        bin_edges = np.linspace(-10, 60, 71)  # Fixed y-axis range from -10 to 60
        hysteresis_df['hysteresis_bin'] = pd.cut(hysteresis_df['hysteresis_area'],
                                                 bins=bin_edges)

        # Frequency table
        heatmap_data = (
            hysteresis_df.groupby(['recovery_rate', 'hysteresis_bin'])
            .size()
            .unstack(fill_value=0)
        )

        # X- and Y-labels
        heatmap_data.columns = heatmap_data.columns.map(
            lambda x: f"{int(x.left)}-{int(x.right)}"
            )

        heatmap_data.index = [
            f"{recovery_rate:.2f}" for recovery_rate in heatmap_data.index
            ]

        # Heatmap
        plt.figure(figsize=(15, 10))
        sns.heatmap(
            heatmap_data.T,  # Transponierte Tabelle für Darstellung
            cmap="turbo",
            cbar_kws={'label': 'Frequency'},
            xticklabels=heatmap_data.index,
            yticklabels=False,
            vmin=0,
            vmax=100,
        )

        plt.yticks(ticks=np.arange(len(bin_edges)), labels=bin_edges.astype(int))

        # Fix y-axis range and invert it
        plt.ylim(0, len(bin_edges) - 1)  # Ensure all bins are shown

        plt.title(f"Heatmap of Hysteresis distribution for {network_type} (Degree: {
            average_degree})")
        plt.xlabel('Recovery chance')
        plt.ylabel('Hysteresis area')
        plt.tight_layout()

        frame_path = os.path.join(output_folder, f"final_ds_{network_type}_degree{
        average_degree}.png")
        plt.savefig(frame_path)
        plt.show()
        plt.close()
