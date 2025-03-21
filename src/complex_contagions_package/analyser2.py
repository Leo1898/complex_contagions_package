"""Simulation analysis."""
import numpy as np
import pandas as pd


def consolidate_data(simulation, ds):
    """Returns data for chosen alpha and simulation no. or average if selected."""
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

def hysteresis_calc(simulation, ds,
                    recovery_rates,
                    t0):
    """Calculates hysteresis areas and averages per alpha."""
    lists = consolidate_data(simulation, ds)
    asc_curve, desc_curve, _, _ = lists

    hysteresis_areas = []
    avg_hysteresis_areas = []

    #for alpha in alphas:
    for recovery_rate in recovery_rates:
        areas_per_alpha = []
        for sim in range(len(simulation)):
            asc_sim = asc_curve.sel(recovery_rate=recovery_rate).isel(simulation=sim)
            desc_sim = desc_curve.sel(recovery_rate=recovery_rate).isel(simulation=sim)

            area = np.trapz((asc_sim - desc_sim), x=t0)
            hysteresis_areas.append({"recovery_rate": recovery_rate,
                                    "simulation": sim+1, #1-based
                                    "hysteresis_area": area
                                    })
            areas_per_alpha.append(area)

        avg_area = np.mean(areas_per_alpha)
        avg_hysteresis_areas.append({"recovery_rate": recovery_rate,
                                    "avg_hysteresis_area": avg_area
                                    })

    hysteresis_df = pd.DataFrame(hysteresis_areas)
    avg_hysteresis_df = pd.DataFrame(avg_hysteresis_areas)

    return hysteresis_df, avg_hysteresis_df

