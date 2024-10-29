"""Visualizer module."""
import ipywidgets as widgets
import matplotlib.pyplot as plt
import xarray as xr
from IPython.display import display


class DiffusionPlotter:
    """Plotter class."""
    def __init__(self, dataset=None, dataset_path=None):
        """Initialization function."""
        if dataset_path:
            self.ds_alpha = xr.open_dataset(dataset_path)
        elif dataset is not None:
            self.ds_alpha = dataset
        else:
            raise ValueError("Either a dataset or a dataset_path must be provided.")

        self.available_simulation = self.ds_alpha.coords["simulation"].values
        self.available_alphas = self.ds_alpha.coords["alpha"].values

    def consolidate_data(self, simulation, alpha):
        """Returns data for chosen alpha and simulation no. or average if selected."""
        ds_filtered = self.ds_alpha.sel(alpha=alpha)

        if simulation == "Average":
            inflist_asc_last_step = ds_filtered.inflist_asc.isel(steps=-1).mean(
                dim="simulation"
                )
            inflist_desc_last_step = ds_filtered.inflist_desc.isel(steps=-1).mean(
                dim="simulation"
                )
            inflist_asc_all_steps = ds_filtered.inflist_asc.mean(
                dim="simulation"
                )
            inflist_desc_all_steps = ds_filtered.inflist_desc.mean(
                dim="simulation"
                )
        else:
            simulation_index = simulation - 1
            inflist_asc_last_step = ds_filtered.inflist_asc.isel(
                simulation=simulation_index, steps=-1
                )
            inflist_desc_last_step = ds_filtered.inflist_desc.isel(
                simulation=simulation_index, steps=-1
                )
            inflist_asc_all_steps = ds_filtered.inflist_asc.isel(
                simulation=simulation_index
                )
            inflist_desc_all_steps = ds_filtered.inflist_desc.isel(
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
            lists = self.consolidate_data(simulation, alpha)
            inflist_asc_fin, inflist_desc_fin, inflist_asc_all, inflist_desc_all = lists

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

    def plot_hysteresis_gaps_with_nan_count(self):
        """Creates a boxplot.

        Boxplot for the distribution of hysteresis gaps per alpha value and line
        showing the number of NaN values per alpha.
        """
        hysteresis_gaps_df = self.ds_alpha.hysteresis_gaps.to_dataframe().reset_index()

        nan_counts = hysteresis_gaps_df.groupby('alpha')['hysteresis_gaps'].apply(
            lambda x: x.isna().sum()
            )

        fig, ax_main = plt.subplots(figsize=(15, 8))  # Größe des Diagramms festlegen
        ax_secondary = ax_main.twinx()  # Sekundäre y-Achse hinzufügen

        hysteresis_gaps_df.boxplot(column='hysteresis_gaps', by='alpha', ax=ax_main,
                                    grid=False
                                    )

        ax_secondary.plot(nan_counts.index, nan_counts.values, marker='o',
                          linestyle='-', label='NaN count'
                          )
        ax_secondary.set_ylabel('Number of missing critical t0')
        ax_secondary.tick_params(axis='y')

        ax_main.set_title('Gaps boxplot and missing critical t0 per alpha')
        ax_main.set_xlabel('alpha')
        ax_main.set_ylabel('Hysteresis gaps')
        plt.suptitle('')

        ticks = ax_main.get_xticks()
        selected_ticks = [tick for tick in ticks if tick % 5 == 0]
        ax_main.set_xticks(selected_ticks)
        ax_main.set_xticklabels([str(int(tick)) for tick in selected_ticks])
        ax_main.xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax_main.tick_params(which='minor', length=4, color='gray', labelsize=8)

        plt.tight_layout()
        ax_secondary.legend(loc="upper right")
        plt.show()
