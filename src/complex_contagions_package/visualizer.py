"""Visualizer module."""
from itertools import cycle

import ipywidgets as widgets
import matplotlib.pyplot as plt
import xarray as xr
from IPython.display import display

from complex_contagions_package.analyser import (
    calculate_peak_and_t0,
    consolidate_data,
    hysteresis_calc,
)


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
        self.available_alphas = self.ds.coords["alpha"].values
        self.available_t0 = self.ds.coords["t0"].values

    def multiple_instances(self, dataset_paths):
        """Creates DiffusionPlotter instances based on the passed dataset paths."""
        # Sicherstellen, dass dataset_paths immer eine Liste ist
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]

        return [DiffusionPlotter(dataset_path=path) for path in dataset_paths]

    def plot_alpha_sim(self, alpha_input, alpha_slider,
                       simulation, direction, show_average
                       ):
        """Plots simulation data for specific alpha values from input or slider."""
        if alpha_input.strip():
            alphas = self.parse_alpha_input(alpha_input)
            use_slider = False
        else:
            alphas = [alpha_slider]
            use_slider = True

        if show_average:
            simulation = "Average"

        fig, ax_main = plt.subplots(figsize=(15, 5))
        ax2 = ax_main.twinx() if use_slider and direction != "Asc vs Desc" else None
        ds = self.ds

        for alpha in alphas:
            lists = consolidate_data(simulation, ds)
            inflist_asc_fin, inflist_desc_fin, inflist_asc_all, inflist_desc_all = lists

            inflist_asc_fin = inflist_asc_fin.sel(alpha=alpha)
            inflist_desc_fin = inflist_desc_fin.sel(alpha=alpha)
            inflist_asc_all = inflist_asc_all.sel(alpha=alpha)
            inflist_desc_all = inflist_desc_all.sel(alpha=alpha)

            if direction == "Asc vs Desc":
                inflist_asc_fin.plot(x="t0", ax=ax_main,
                                     label=f"Alpha {alpha} Ascending"
                                     )
                inflist_desc_fin.plot(x="t0", ax=ax_main,
                                      label=f"Alpha {alpha} Descending"
                                      )
                ax_main.legend()
                if ax2:
                    ax2.set_visible(False)

            elif direction == "Ascending":
                if use_slider:
                    inflist_asc_all.plot(x="t0", y="steps", ax=ax2, alpha=0.4)
                inflist_asc_fin.plot(x="t0", ax=ax_main, label=f"Alpha {alpha}")
                ax_main.legend()
                ax_main.set_title("")

            elif direction == "Descending":
                if use_slider:
                    inflist_desc_all.plot(x="t0", y="steps", ax=ax2, alpha=0.4)
                inflist_desc_fin.plot(x="t0", ax=ax_main, label=f"Alpha {alpha}")
                ax_main.legend()
                ax_main.set_title("")

        ax_main.set_xlabel("Threshold t0")
        ax_main.set_ylabel("Final number of infected nodes")

        plt.tight_layout()
        plt.title(f"Diffusion for {direction} t0 - Simulation {simulation}")
        plt.show()

    def parse_alpha_input(self, alpha_str):
        """Parses the alpha input from the text field into a list of integers."""
        try:
            return [int(alpha) for alpha in alpha_str.split(',')
                    if alpha.strip().isdigit()]
        except ValueError:
            return []

    def show_widgets(self):
        """Widgets zur Steuerung des Plots ohne doppelte Anzeige der Einzelwidgets."""
        direction_toggle = widgets.ToggleButtons(
            options=["Asc vs Desc", "Ascending", "Descending"],
            description="Direction:",
            disabled=False,
            button_style="info"
        )

        alpha_slider = widgets.IntSlider(
            value=self.available_alphas[-1],
            min=self.available_alphas.min(),
            max=self.available_alphas.max(),
            step=2,
            description="Alpha:",
            continuous_update=False
        )

        simulation_slider = widgets.IntSlider(
            value=self.available_simulation[0],
            min=self.available_simulation.min(),
            max=self.available_simulation.max(),
            step=1,
            description="Simulation:",
            continuous_update=False
        )

        average_checkbox = widgets.Checkbox(
            value=False,
            description="Show Average",
            disabled=False
        )

        alpha_input = widgets.Text(
            placeholder='e.g.: 100, 50, 10, 5',
            style={'description_width': '0px'},
            layout=widgets.Layout(width='150px')
        )

        alpha_label = widgets.Label(
            value='Choose alphas to be plotted together:',
            layout=widgets.Layout(width='auto')
        )

        alpha_box = widgets.VBox([alpha_slider, widgets.VBox(
            [alpha_label, alpha_input]
            )],
            layout=widgets.Layout(align_items='center',
                                  justify_content='center'
                                                       ))

        simulation_box = widgets.VBox([simulation_slider, average_checkbox])
        control_box = widgets.HBox([alpha_box, simulation_box],
                                   layout=widgets.Layout(align_items='center',
                                                         justify_content='center'
                                                         ))
        main_box = widgets.VBox([control_box, direction_toggle],
                                layout=widgets.Layout(align_items='center',
                                                      justify_content='center'
                                                      ))

        interactive_plot = widgets.interactive(
            self.plot_alpha_sim,
            alpha_input=alpha_input,
            alpha_slider=alpha_slider,
            simulation=simulation_slider,
            direction=direction_toggle,
            show_average=average_checkbox
        )

        display(main_box)
        display(interactive_plot.children[-1])

    def hysteresis_boxplot(self):
        """Creates a boxplot.

        Boxplot for the distribution of hysteresis per alpha value.
        """
        simulation = self.available_simulation
        ds = self.ds
        alphas = self.available_alphas
        t0 = self.available_t0
        network_type = ds.attrs.get("network_type", "Unknown")
        average_degree = ds.attrs.get("average_degree", "Unknown")

        hysteresis_df, _ = hysteresis_calc(simulation, ds, alphas, t0)

        #hysteresis_df['recovery_rate'] = 1 / hysteresis_df['alpha']

        fig, ax_main = plt.subplots(figsize=(15, 8))

        hysteresis_df.boxplot(column='hysteresis_area', by='alpha', ax=ax_main,
                                    grid=False
                                    )

        ax_main.set_xticks(range(1, len(alphas) + 1))
        ax_main.set_xticklabels([int(alpha) for alpha in alphas])

        ax_main.set_title(f"Hysteresis boxplot per alpha for {network_type} (Degree: {
            average_degree})")
        ax_main.set_xlabel('alpha')
        ax_main.set_ylabel('Hysteresis')
        plt.suptitle('')
        plt.tight_layout()
        plt.show()

        # Neue Funktion im visualizer Modul zum Scatter-Plot der kritischen t0-Werte
    def boxplot_max_peaks_per_alpha(self):
        """Scatterplot der Max-Peaks je Simulation pro Alpha."""
        simulation = self.available_simulation
        ds = self.ds
        alphas = self.available_alphas
        t0 = self.available_t0
        network_type = ds.attrs.get("network_type", "Unknown")
        average_degree = ds.attrs.get("average_degree", "Unknown")

        peak_data = calculate_peak_and_t0(simulation, ds, alphas, t0)

        fig, ax_main = plt.subplots(figsize=(15, 8))

        peak_data.boxplot(column='max_peak', by='alpha', ax=ax_main,
                                    grid=False
                                    )

        ax_main.set_xticks(range(1, len(alphas) + 1))
        ax_main.set_xticklabels([int(alpha) for alpha in alphas])

        ax_main.set_title(f"Max Peak boxplot per alpha for {network_type} (Degree: {
            average_degree})")
        ax_main.set_xlabel('alpha')
        ax_main.set_ylabel('Max Peak')
        plt.suptitle('')
        plt.tight_layout()
        plt.show()

    def boxplot_peakt0_per_alpha(self):
        """Scatterplot der Max-Peaks je Simulation pro Alpha."""
        simulation = self.available_simulation
        ds = self.ds
        alphas = self.available_alphas
        t0 = self.available_t0
        network_type = ds.attrs.get("network_type", "Unknown")
        average_degree = ds.attrs.get("average_degree", "Unknown")

        peak_data = calculate_peak_and_t0(simulation, ds, alphas, t0)

        fig, ax_main = plt.subplots(figsize=(15, 8))

        peak_data.boxplot(column='peak_t0', by='alpha', ax=ax_main,
                                    grid=False
                                    )

        ax_main.set_xticks(range(1, len(alphas) + 1))
        ax_main.set_xticklabels([int(alpha) for alpha in alphas])

        ax_main.set_title(f"Max Peak boxplot per alpha for {network_type} (Degree: {
            average_degree})")
        ax_main.set_xlabel('alpha')
        ax_main.set_ylabel('Max Peak')
        plt.suptitle('')
        plt.tight_layout()
        plt.show()

    def avg_hysteresis_per_alpha(self, dataset_paths):
        """Plots the average hysteresis area per alpha as a curve."""
        fig, ax_main = plt.subplots(figsize=(15, 8))

        # Zuordnungen für Farben und Linienstile
        color_map = {}  # Speichert Farben für jeden network_type
        linestyle_map = {}  # Speichert Linienstile für jeden average_degree
        colors = cycle(plt.cm.tab10.colors)  # Verwende einen vordefinierten Farbsatz
        linestyles = cycle(['-', '--', '-.', ':'])  # Definiere die Linienstile

        for dataset_path in dataset_paths:
            # Lade den Datensatz
            ds = xr.open_dataset(dataset_path)

            # Extrahiere die benötigten Parameter
            simulation = ds.simulation.values
            alphas = ds.alpha.values
            t0 = ds.t0.values

            # Berechne die Hysterese
            _, avg_hysteresis_df = hysteresis_calc(simulation, ds, alphas, t0)

            # Extrahiere Metadaten
            network_type = ds.attrs.get("network_type", "Unknown")
            average_degree = ds.attrs.get("average_degree", "Unknown")

            # Farbe auswählen (oder neu zuordnen)
            if average_degree not in color_map:
                color_map[average_degree] = next(colors)

            # Linienstil auswählen (oder neu zuordnen)
            if network_type not in linestyle_map:
                linestyle_map[network_type] = next(linestyles)

            #x = 1/avg_hysteresis_df['alpha']

            # Plotte die Daten
            ax_main.plot(avg_hysteresis_df['alpha'],
                        avg_hysteresis_df['avg_hysteresis_area'],
                        #marker='o',
                        linestyle=linestyle_map[network_type],
                        color=color_map[average_degree],
                        label=f"{network_type} (Degree: {average_degree})")

        # Beschriftung und Formatierung
        ax_main.set_title('Average Hysteresis per alpha')
        ax_main.set_xlabel('Alpha')
        ax_main.set_ylabel('Average Hysteresis Area')
        ax_main.legend()
        plt.tight_layout()
        plt.show()

    def avg_maxpeak_per_alpha(self, dataset_paths):
        """Plots the average hysteresis area per alpha as a curve."""
        fig, ax_main = plt.subplots(figsize=(15, 8))

        # Zuordnungen für Farben und Linienstile
        color_map = {}  # Speichert Farben für jeden network_type
        linestyle_map = {}  # Speichert Linienstile für jeden average_degree
        colors = cycle(plt.cm.tab10.colors)  # Verwende einen vordefinierten Farbsatz
        linestyles = cycle(['-', '--', '-.', ':'])  # Definiere die Linienstile

        for dataset_path in dataset_paths:
            # Lade den Datensatz
            ds = xr.open_dataset(dataset_path)

            # Extrahiere die benötigten Parameter
            simulation = ds.simulation.values
            alphas = ds.alpha.values
            t0 = ds.t0.values

            # Berechne die Hysterese
            _, mean_peak_data = calculate_peak_and_t0(simulation, ds, alphas, t0)

            # Extrahiere Metadaten
            network_type = ds.attrs.get("network_type", "Unknown")
            average_degree = ds.attrs.get("average_degree", "Unknown")

            # Farbe auswählen (oder neu zuordnen)
            if average_degree not in color_map:
                color_map[average_degree] = next(colors)

            # Linienstil auswählen (oder neu zuordnen)
            if network_type not in linestyle_map:
                linestyle_map[network_type] = next(linestyles)

            #x = 1/avg_hysteresis_df['alpha']

            # Plotte die Daten
            ax_main.plot(mean_peak_data['alpha'],
                        mean_peak_data['max_peak'],
                        #marker='o',
                        linestyle=linestyle_map[network_type],
                        color=color_map[average_degree],
                        label=f"{network_type} (Degree: {average_degree})")

        # Beschriftung und Formatierung
        ax_main.set_title('Average MaxPeak per alpha')
        ax_main.set_xlabel('Alpha')
        ax_main.set_ylabel('Average Max Peak')
        ax_main.legend()
        plt.tight_layout()
        plt.show()

    def avg_peakt0_per_alpha(self, dataset_paths):
        """Plots the average hysteresis area per alpha as a curve."""
        fig, ax_main = plt.subplots(figsize=(15, 8))

        # Zuordnungen für Farben und Linienstile
        color_map = {}  # Speichert Farben für jeden network_type
        linestyle_map = {}  # Speichert Linienstile für jeden average_degree
        colors = cycle(plt.cm.tab10.colors)  # Verwende einen vordefinierten Farbsatz
        linestyles = cycle(['-', '--', '-.', ':'])  # Definiere die Linienstile

        for dataset_path in dataset_paths:
            # Lade den Datensatz
            ds = xr.open_dataset(dataset_path)

            # Extrahiere die benötigten Parameter
            simulation = ds.simulation.values
            alphas = ds.alpha.values
            t0 = ds.t0.values

            # Berechne die Hysterese
            _, mean_peak_data = calculate_peak_and_t0(simulation, ds, alphas, t0)

            # Extrahiere Metadaten
            network_type = ds.attrs.get("network_type", "Unknown")
            average_degree = ds.attrs.get("average_degree", "Unknown")

            # Farbe auswählen (oder neu zuordnen)
            if average_degree not in color_map:
                color_map[average_degree] = next(colors)

            # Linienstil auswählen (oder neu zuordnen)
            if network_type not in linestyle_map:
                linestyle_map[network_type] = next(linestyles)

            #x = 1/avg_hysteresis_df['alpha']

            # Plotte die Daten
            ax_main.plot(mean_peak_data['alpha'],
                        mean_peak_data['peak_t0'],
                        #marker='o',
                        linestyle=linestyle_map[network_type],
                        color=color_map[average_degree],
                        label=f"{network_type} (Degree: {average_degree})")

        # Beschriftung und Formatierung
        ax_main.set_title('Average PeakT0 per alpha')
        ax_main.set_xlabel('Alpha')
        ax_main.set_ylabel('Average Peak t0')
        ax_main.legend()
        plt.tight_layout()
        plt.show()
