"""Visualizer module."""
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import display


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

    def consolidate_data(self, simulation):
        """Returns data for chosen alpha and simulation no. or average if selected."""
        ds = self.ds

        if isinstance(simulation, str) and simulation == "Average":
            inflist_asc_last_step = ds.inflist_asc.isel(steps=-1).mean(
                dim="simulation"
                )
            inflist_desc_last_step = ds.inflist_desc.isel(steps=-1).mean(
                dim="simulation"
                )
            inflist_asc_all_steps = ds.inflist_asc.mean(
                dim="simulation"
                )
            inflist_desc_all_steps = ds.inflist_desc.mean(
                dim="simulation"
                )
        else:
            simulation_index = simulation - 1
            inflist_asc_last_step = ds.inflist_asc.isel(
                simulation=simulation_index, steps=-1
                )
            inflist_desc_last_step = ds.inflist_desc.isel(
                simulation=simulation_index, steps=-1
                )
            inflist_asc_all_steps = ds.inflist_asc.isel(
                simulation=simulation_index
                )
            inflist_desc_all_steps = ds.inflist_desc.isel(
                simulation=simulation_index
                )

        return (inflist_asc_last_step, inflist_desc_last_step,
                inflist_asc_all_steps, inflist_desc_all_steps)

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

        for alpha in alphas:
            lists = self.consolidate_data(simulation)
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
            step=1,
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

        lists = self.consolidate_data(simulation)
        asc_curve, desc_curve, _, _ = lists

        hysteresis_areas = []
        for alpha in self.available_alphas:
            for sim in range(len(self.available_simulation)):
                asc_sim = asc_curve.sel(alpha=alpha).isel(simulation=sim)
                desc_sim = desc_curve.sel(alpha=alpha).isel(simulation=sim)

                area = np.trapz((asc_sim - desc_sim), x=self.available_t0)
                hysteresis_areas.append({"alpha": alpha,
                                         "simulation": sim,
                                         "hysteresis_area": area
                                         })

        hysteresis_df = pd.DataFrame(hysteresis_areas)

        fig, ax_main = plt.subplots(figsize=(15, 8))

        hysteresis_df.boxplot(column='hysteresis_area', by='alpha', ax=ax_main,
                                    grid=False
                                    )

        ax_main.set_title('Hysteresis boxplot per alpha')
        ax_main.set_xlabel('alpha')
        ax_main.set_ylabel('Hysteresis')
        plt.suptitle('')
        plt.tight_layout()
        plt.show()
